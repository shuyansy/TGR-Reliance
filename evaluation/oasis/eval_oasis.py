"""Video-Oasis: accuracy and ReGAP in one pass (Qwen2.5-VL / Qwen3-VL and models trained from them).

For every question (official question list, official frames incl. the RTV-Bench time windows, one prompt):
  * accuracy request: greedy, the model's repetition_penalty, up to 16,384 tokens, official answer extraction
    (the official Video-Oasis protocol);
  * ReGAP: an anchor response (greedy, up to 256 tokens, top-64 logprobs) gives the answer a*, the predicted span and
    p_full; then the predicted span and K random blocks of the same number of frames are deleted (utils/deletion.py).
    When a deletion arm writes no option letter, the probabilities are read at its answer slot, or after forcing one
    "<answer>" token; arms that still have no value are left out (refusal="exclude" in regap.summarize).

  OASIS_REPO=/path/to/Video-Oasis python evaluation/oasis/eval_oasis.py --family qwen25 --model <dir> --prompt train \
      --out results/oasis/<name>_shard0of8.jsonl --num_shards 8 --shard_id 0
  python evaluation/oasis/summarize_oasis.py results/oasis/<name>_shard*of8.jsonl
"""
import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import official as OO  # noqa: E402
from utils import deletion as D  # noqa: E402
from utils import frames as F  # noqa: E402
from utils.parsing import agg_letters, letter_dist, parse_letter_simple, parse_span, slot_read  # noqa: E402
from utils.prompts import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["qwen25", "qwen3"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="train")
    ap.add_argument("--system", default=None)
    ap.add_argument("--suffix", default=None)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--uids", default="", help="file with one uid per line: evaluate only these questions")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--decode_workers", type=int, default=2)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--anchor_tokens", type=int, default=256)
    ap.add_argument("--acc_tokens", type=int, default=16384)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--max_model_len", type=int, default=0, help="0 = 40960 (Qwen3) / 69632 (Qwen2.5)")
    ap.add_argument("--hf_mrope", action="store_true", help="Qwen2.5: compute vLLM M-RoPE time positions exactly as HF (see mrope_patch.py)")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    system, suffix = PROMPTS[a.prompt]
    system, suffix = a.system or system, a.suffix or suffix
    qs = sorted(OO.questions(), key=lambda q: q["uid"])
    if a.uids:
        keep = {x.strip() for x in open(a.uids) if x.strip()}
        qs = [q for q in qs if q["uid"] in keep]
    qs = qs[a.shard_id::a.num_shards]
    done = set()
    if a.resume and os.path.exists(a.out):
        done = {json.loads(x)["id"] for x in open(a.out) if x.strip()}
    todo = [q for q in qs if q["uid"] not in done]
    print(f"[oasis] {len(todo)} questions to run ({len(done)} done), family={a.family}, prompt={a.prompt}", flush=True)
    if not todo:
        return

    if a.hf_mrope:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        import mrope_patch
        mrope_patch.apply()
    from transformers import AutoProcessor, AutoTokenizer, GenerationConfig
    from vllm import LLM, SamplingParams
    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, tensor_parallel_size=1, gpu_memory_utilization=0.85,
              max_model_len=a.max_model_len or (40960 if a.family == "qwen3" else 69632),
              limit_mm_per_prompt={"video": 1, "image": 0}, seed=0, max_logprobs=max(a.topk, 20))
    rep_pen = float(getattr(GenerationConfig.from_pretrained(a.model), "repetition_penalty", None) or 1.0)
    eos = processor.tokenizer.eos_token_id
    sp_acc = SamplingParams(temperature=0.0, max_tokens=a.acc_tokens, repetition_penalty=rep_pen, ignore_eos=True, stop_token_ids=[eos])
    sp_anc = SamplingParams(max_tokens=a.anchor_tokens, temperature=0, logprobs=a.topk)
    sp1 = SamplingParams(max_tokens=1, temperature=0, logprobs=a.topk)

    def chat(text):
        msg = [{"role": "system", "content": [{"type": "text", "text": system}]},
               {"role": "user", "content": [{"type": "video", "video": "placeholder"}, {"type": "text", "text": text}]}]
        try:
            return processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def req(prompt, frames, meta):
        return F.video_request(a.model, prompt, frames, meta)

    def dist_of(out, prompt, frames, meta, n, letter):
        """Letter distribution of a response: at the answer letter; else at the answer slot; else after forcing '<answer>'."""
        d = letter_dist(out, tok, n, hint=letter)
        if d:
            return d, "letter"
        o = out.outputs[0]
        probs, marker = slot_read(out, tok, n)
        src = "slot"
        if probs is None and o.finish_reason != "length":
            o1 = llm.generate([req(prompt + o.text + ("" if marker else "<answer>"), frames, meta)], sampling_params=sp1, use_tqdm=False)[0].outputs[0]
            if o1.logprobs:
                probs, src = agg_letters(o1.logprobs[0], tok, n), "forced"
        if probs is None:
            return None, "none"
        Z = sum(probs.values())
        return ({k: v / Z for k, v in probs.items()} if Z > 1e-12 else None), src

    def load(q):
        if not q["local_video"]:
            return q, None, None, "no_video"
        t0, t1 = OO.bounds(q)
        try:
            fr, me = F.sample_qwen(q["local_video"], a.family, None, t0, t1)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return q, None, None, f"decode_exception: {type(e).__name__}: {e}"
        if not fr or me is None:
            return q, None, None, "video_decode_error"
        return q, fr, me, None

    def gen(reqs, sps):
        try:
            return llm.generate(reqs, sampling_params=sps, use_tqdm=False)
        except Exception:  # noqa: BLE001  retry one by one; failures become error records
            traceback.print_exc()
            res = []
            for r_, s_ in zip(reqs, sps if isinstance(sps, list) else [sps] * len(reqs)):
                try:
                    res.append(llm.generate([r_], sampling_params=s_, use_tqdm=False)[0])
                except Exception as e:  # noqa: BLE001
                    traceback.print_exc()
                    res.append(e)
            return res

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fout = open(a.out, "a")
    pool = ThreadPoolExecutor(max_workers=a.decode_workers)
    chunks = [todo[i:i + a.batch] for i in range(0, len(todo), a.batch)]
    pending = pool.map(load, chunks[0])
    n_err, t_start = 0, time.time()
    for ci in range(len(chunks)):
        loaded = list(pending)
        if ci + 1 < len(chunks):
            pending = pool.map(load, chunks[ci + 1])
        items, full_reqs, full_sps, recs = [], [], [], []
        for q, fr, me, err in loaded:
            n = len(q["options"])
            rec = {"id": q["uid"], "db": q["db"], "category": q.get("oasis_category"), "gt_answer": q["answer"], "n_options": n,
                   "letter": None, "correct": False, "answer": None, "pred_spans": [], "p_full": None, "p_span_deleted": None,
                   "p_random_deleted": None, "p_random_blocks": [], "status": None}
            recs.append(rec)
            if err:
                rec["status"] = "no_video" if err == "no_video" else "video_decode_error"
                if err != "no_video":
                    rec["error"] = err
                    n_err += 1
                continue
            body = "Question: " + str(q["question"]).strip() + "\nOptions:\n" + "\n".join(str(o).strip() for o in q["options"]) + "\n" + suffix
            prompt = chat(body)
            items.append((q, fr, me, prompt, rec))
            r0 = req(prompt, fr, me)
            full_reqs += [r0, r0]
            full_sps += [sp_acc, sp_anc]
        outs = gen(full_reqs, full_sps) if full_reqs else []
        del_reqs, del_meta = [], []
        for j, (q, fr, me, prompt, rec) in enumerate(items):
            o_acc, o_anc = outs[2 * j], outs[2 * j + 1]
            if isinstance(o_acc, Exception) or isinstance(o_anc, Exception):
                e = o_acc if isinstance(o_acc, Exception) else o_anc
                rec.update(status="generation_error", error=f"{type(e).__name__}: {e}")
                n_err += 1
                continue
            n = rec["n_options"]
            letter = OO.extract(OO.postprocess(a.family, o_acc.outputs[0].text), n) or None
            rec.update(letter=letter, correct=(letter == q["answer"]), acc_text=o_acc.outputs[0].text[-2000:])
            raw1 = o_anc.outputs[0].text
            fl = parse_letter_simple(raw1, n)
            rec.update(answer=fl, anchor_text=raw1, n_frames=len(fr))
            fd, fsrc = dist_of(o_anc, prompt, fr, me, n, fl) if fl else (None, "none")
            rec.update(p_full=(fd.get(fl) if (fd and fl) else None), p_full_src=fsrc)
            if len(fr) < 4:
                rec["status"] = "video_too_short"
                continue
            span = parse_span(raw1, me["duration"])
            if span is None:
                rec["status"] = "no_span"
                continue
            rec["pred_spans"] = [span]
            pl = D.plan(me, len(fr), span, q["uid"], a.k, a.margin, a.seed)
            rec.update(deletion_mode=pl["mode"], n_cited_frames=pl["n_cited"])
            if pl["mode"] == "nodelete":
                rec.update(p_span_deleted=rec["p_full"], p_span_src="nodelete", p_random_blocks=[rec["p_full"]], p_random_deleted=rec["p_full"],
                           status="ok" if rec["p_full"] is not None else "no_answer")
                continue
            if pl["mode"] == "not_deletable":
                rec["status"] = "not_deletable"
                continue
            for kind, drop in [("span", pl["cited"])] + [("random", b) for b in pl["controls"]]:
                f2, m2 = F.drop_frames(fr, me, drop)
                del_reqs.append(req(prompt, f2, m2))
                del_meta.append((j, kind, f2, m2))
        if del_reqs:
            douts = gen(del_reqs, [sp_anc] * len(del_reqs))
            per = {}
            for (j, kind, f2, m2), o in zip(del_meta, douts):
                per.setdefault(j, []).append((kind, f2, m2, o))
            for j, lst in per.items():
                q, fr, me, prompt, rec = items[j]
                n, fl = rec["n_options"], rec["answer"]
                if any(isinstance(o, Exception) for *_, o in lst):
                    rec["status"] = "generation_error"
                    n_err += 1
                    continue
                pr = []
                for kind, f2, m2, o in lst:
                    la = parse_letter_simple(o.outputs[0].text, n)
                    d, src = dist_of(o, prompt, f2, m2, n, la) if fl else (None, "none")
                    p = d[fl] if d else None
                    if kind == "span":
                        rec.update(p_span_deleted=p, p_span_src=src, answer_span_deleted=la)
                    else:
                        pr.append(p)
                vals = [p for p in pr if p is not None]
                rec.update(p_random_blocks=pr, p_random_deleted=(sum(vals) / len(vals) if vals else None))
                rec["status"] = "no_answer" if (fl is None or rec["p_full"] is None) else ("ok" if vals else "not_deletable")
        for rec in recs:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        nd = sum(len(c) for c in chunks[:ci + 1])
        el = time.time() - t_start
        print(f"[oasis] {nd}/{len(todo)} questions, {n_err} errors, {el / 60:.1f} min, ~{(len(todo) - nd) * el / max(nd, 1) / 60:.1f} min left", flush=True)
    fout.close()
    print(f"[oasis] finished, {n_err} errors -> {a.out}", flush=True)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
