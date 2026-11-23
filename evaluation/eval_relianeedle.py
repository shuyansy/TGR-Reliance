"""Evaluate a video LLM on ReliaNeedle and record everything ReGAP needs (one JSON line per item).

For every item the model answers once with the full video (greedy). From that response we read its answer a*, its predicted
span and p_full = p(a* | full video) at the answer slot. We then delete the predicted span, and separately K random blocks
of the same number of frames, keep the prompt fixed, and read p(a*) again (utils/deletion.py describes the arms).
`python evaluation/compute_regap.py <out.jsonl>` turns the file into the Table 1 columns.

Example (our released Qwen2.5-VL-7B model, as in the paper):
  python evaluation/eval_relianeedle.py --model <dir> --sampler qwen25 --prompt train --max_tokens 256 \
      --annotations relianeedle/relianeedle.json --video_root <ReliaNeedle dir> --out results/ours-qwen2.5-vl-7b.jsonl
Backbones of Table 1:  --sampler grid --prompt tga --max_tokens 64
A new model: add its prompt with --system / --suffix (or --prompt), and make sure its output contains the span as
[start, end] in seconds inside <think></think> and the option letter inside <answer></answer>, or adapt utils/parsing.py.
"""
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import deletion as D  # noqa: E402
from utils import frames as F  # noqa: E402
from utils.parsing import letter_dist, parse_letter, parse_span  # noqa: E402
from utils.prompts import PROMPTS, user_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--video_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sampler", choices=["grid", "qwen25", "qwen3"], required=True)
    ap.add_argument("--max_frames", type=int, default=None, help="grid: default 64; qwen25/qwen3: default 128")
    ap.add_argument("--max_side", type=int, default=768, help="grid sampler only")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="train")
    ap.add_argument("--system", default=None, help="overrides the system prompt of --prompt")
    ap.add_argument("--suffix", default=None, help="overrides the final instruction of --prompt")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--k", type=int, default=3, help="number of random control blocks")
    ap.add_argument("--margin", type=float, default=1.0, help="seconds added on both sides of the predicted span")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    system, suffix = PROMPTS[a.prompt]
    system, suffix = a.system or system, a.suffix or suffix
    items = json.load(open(a.annotations))
    items = items[a.shard_id::a.num_shards]
    if a.limit:
        items = items[:a.limit]
    done = set()
    if a.resume and os.path.exists(a.out):
        done = {json.loads(x)["id"] for x in open(a.out) if x.strip()}
    todo = [it for it in items if it["id"] not in done]
    print(f"[relianeedle] {len(todo)} items to run ({len(done)} already done), sampler={a.sampler}, prompt={a.prompt}", flush=True)
    if not todo:
        return

    from transformers import AutoProcessor, AutoTokenizer
    from vllm import LLM, SamplingParams
    processor = AutoProcessor.from_pretrained(a.model, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    llm = LLM(model=a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_memory_utilization)
    sp = SamplingParams(max_tokens=a.max_tokens, temperature=0, logprobs=20)

    def chat(text):
        msg = [{"role": "system", "content": [{"type": "text", "text": system}]},
               {"role": "user", "content": [{"type": "video", "video": "placeholder"}, {"type": "text", "text": text}]}]
        return processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def gen(prompt, frames, meta):
        return llm.generate([F.video_request(a.model, prompt, frames, meta)], sampling_params=sp, use_tqdm=False)[0]

    def sample(path):
        if a.sampler == "grid":
            return F.sample_grid(path, 1.0, a.max_frames or 64, a.max_side)
        return F.sample_qwen(path, a.sampler, a.max_frames)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    n_err, t0 = 0, time.time()
    with open(a.out, "a") as fout:
        for i, it in enumerate(todo):
            n = len(it["options"])
            rec = {"id": it["id"], "subset": it.get("subset"), "gt_span": it.get("needle_span"), "gt_answer": it["answer"], "n_options": n,
                   "correct": False, "answer": None, "pred_spans": [], "p_full": None, "p_span_deleted": None,
                   "p_random_deleted": None, "p_random_blocks": [], "status": None}
            try:
                frames, meta = sample(os.path.join(a.video_root, it["video"]))
                if not frames:
                    raise RuntimeError(f"cannot read {it['video']}")
                prompt = chat(user_text(it["question"], it["options"], suffix))
                out = gen(prompt, frames, meta)
                text = out.outputs[0].text
                a_star = parse_letter(text, n)
                dist = letter_dist(out, tok, n, hint=a_star)
                rec.update(full_text=text, finish_reason=out.outputs[0].finish_reason, n_frames=len(frames))
                if a_star is None or dist is None:
                    rec["status"] = "no_answer"
                else:
                    rec.update(answer=a_star, correct=(a_star == it["answer"]), p_full=dist[a_star])
                    span = parse_span(text, meta["duration"])
                    if span is None:
                        rec["status"] = "no_span"
                    else:
                        rec["pred_spans"] = [span]
                        pl = D.plan(meta, len(frames), span, it["id"], a.k, a.margin, a.seed)
                        rec.update(deletion_mode=pl["mode"], n_cited_frames=pl["n_cited"])
                        if pl["mode"] == "nodelete":        # the span covers every frame: both arms keep the full video
                            rec.update(p_span_deleted=rec["p_full"], p_random_blocks=[rec["p_full"]], p_random_deleted=rec["p_full"], status="ok")
                        elif pl["mode"] == "not_deletable":
                            rec["status"] = "not_deletable"
                        else:
                            def arm(drop):
                                f2, m2 = F.drop_frames(frames, meta, drop)
                                o = gen(prompt, f2, m2)
                                la = parse_letter(o.outputs[0].text, n)
                                d = letter_dist(o, tok, n, hint=la)
                                return (d[a_star] if d else None), la
                            p_span, l_span = arm(pl["cited"])
                            blocks = [arm(b) for b in pl["controls"]]
                            ps = [p for p, _ in blocks if p is not None]
                            rec.update(p_span_deleted=p_span, answer_span_deleted=l_span, p_random_blocks=[p for p, _ in blocks],
                                       answers_random_deleted=[la for _, la in blocks], p_random_deleted=(sum(ps) / len(ps) if ps else None))
                            rec["status"] = ("ok" if p_span is not None else "refusal") if ps else "not_deletable"
            except Exception as e:  # noqa: BLE001  keep going, but report and exit non-zero at the end
                n_err += 1
                rec.update(status="error", error=f"{type(e).__name__}: {e}")
                print(f"!! {it['id']}:\n{traceback.format_exc()}", flush=True)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if (i + 1) % 20 == 0:
                print(f"[relianeedle] {i + 1}/{len(todo)} done, {n_err} errors, {(time.time() - t0) / (i + 1):.1f} s/item", flush=True)
    print(f"[relianeedle] finished {len(todo)} items, {n_err} errors -> {a.out}", flush=True)
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
