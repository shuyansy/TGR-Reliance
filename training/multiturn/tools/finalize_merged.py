"""Turn the HF folder written by Video-o3's tools/model_merger.py into a folder with the same format as the starting model
(in place, can be run again):
  1. tensor names. qwen25: transformers 4.52 writes the new names (model.language_model.X, model.visual.X); they are renamed back to the
     names of the Qwen2.5-VL checkpoints (model.X, visual.X), values untouched. qwen3: the names already match; only checked.
  2. the non-weight files written by the merger (config / processor / tokenizer / chat template / generation_config) are moved to
     _merger_saved/ and replaced by those of the starting model (same architecture; generation_config must be the original one, vLLM uses it
     as default sampling parameters while the merger writes the transformers defaults).
  3. a few tensors are compared with the starting model: if all are identical the run stops with an error (training had no effect).
  4. (qwen3 by default, qwen25 with --full_check) every tensor is compared with the starting model: same names and shapes, no NaN / Inf, and
     no element changed by more than VO3_FINALIZE_TOL_ABS (0.01) + VO3_FINALIZE_TOL_REL (2^-6) x |start|. A legitimate update at lr 1e-6 is
     far below this bound; it catches corrupted parameters (we once found a block of Qwen3 vision parameters overwritten, see the README).
Usage: python finalize_merged.py --family qwen25|qwen3 <merged dir .../actor/huggingface> <starting model> [--full_check]
       python finalize_merged.py --family qwen25|qwen3 <merged dir> <starting model> --check_only
           only step 4, nothing is changed; exit code 0 = no problem, 3 = problems found, 2 = the check itself failed
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

WEIGHT_INDEX = "model.safetensors.index.json"


def shards(d):
    idx = os.path.join(d, WEIGHT_INDEX)
    if os.path.exists(idx):
        return sorted(set(json.load(open(idx))["weight_map"].values()))
    return ["model.safetensors"]


def locate(d):
    m = {}
    for fn in shards(d):
        with safe_open(os.path.join(d, fn), framework="pt") as f:
            for k in f.keys():
                m[k] = fn
    return m


def legacy(k):
    if k.startswith("model.language_model."):
        return "model." + k[len("model.language_model."):]
    if k.startswith("model.visual."):
        return "visual." + k[len("model.visual."):]
    return k


def rename_q25(merged):
    renamed = 0
    for fn in shards(merged):
        fp = os.path.join(merged, fn)
        with safe_open(fp, framework="pt") as f:
            keys = list(f.keys())
        if not any(legacy(k) != k for k in keys):
            continue
        t = load_file(fp)
        t = {legacy(k): v for k, v in t.items()}
        assert len(t) == len(keys), f"{fn}: the number of tensors changed after renaming (name clash)"
        save_file(t, fp + ".tmp", metadata={"format": "pt"})
        os.replace(fp + ".tmp", fp)
        renamed += len(keys)
        del t
    idx = os.path.join(merged, WEIGHT_INDEX)
    if os.path.exists(idx):
        j = json.load(open(idx))
        j["weight_map"] = {legacy(k): v for k, v in j["weight_map"].items()}
        json.dump(j, open(idx + ".tmp", "w"), indent=2)
        os.replace(idx + ".tmp", idx)
    print(f"[finalize] 1. renamed {renamed} tensors to the Qwen2.5-VL names (0 = already renamed)", flush=True)


def probes_of(family, mm, bm):
    if family == "qwen25":
        L, V = "model.", "visual."
        keys = [L + "embed_tokens.weight", "lm_head.weight", L + "norm.weight", V + "patch_embed.proj.weight", V + "blocks.0.attn.qkv.weight",
                V + "merger.mlp.0.weight", L + "layers.0.self_attn.q_proj.weight", L + "layers.0.mlp.down_proj.weight"]
    else:
        L, V = "model.language_model.", "model.visual."
        keys = [L + "embed_tokens.weight", "lm_head.weight", L + "norm.weight", V + "patch_embed.proj.weight", V + "blocks.0.attn.qkv.weight",
                V + "merger.linear_fc1.weight", L + "layers.0.self_attn.q_proj.weight", L + "layers.0.mlp.down_proj.weight"]
    probes = [k for k in keys if k in mm and k in bm]
    pos = len(L.split("."))   # index of the layer number in the split name ("model.layers.N" -> 2, "model.language_model.layers.N" -> 3)
    last = max((int(k.split(".")[pos]) for k in mm if k.startswith(L + "layers.")), default=None)
    if last is not None:
        probes += [k for k in (f"{L}layers.{last}.self_attn.o_proj.weight", f"{L}layers.{last}.mlp.up_proj.weight") if k in mm and k in bm]
    return probes


def finalize(family, merged, base, full_check):
    # 1. names
    if family == "qwen25":
        rename_q25(merged)
    else:
        nm, nb = set(locate(merged)), set(locate(base))
        if nm != nb:
            print(f"!! tensor names differ from the starting model: extra {sorted(nm - nb)[:5]} ({len(nm - nb)}) / missing {sorted(nb - nm)[:5]} ({len(nb - nm)})", flush=True)
            sys.exit(1)
        print(f"[finalize] 1. tensor names match the starting model: {len(nm)}", flush=True)

    # 2. non-weight files from the starting model
    keep = os.path.join(merged, "_merger_saved")
    os.makedirs(keep, exist_ok=True)
    moved = []
    if not os.path.exists(os.path.join(keep, ".done")):   # on a second run do not move again (it would overwrite the saved originals)
        for fn in sorted(os.listdir(merged)):
            p = os.path.join(merged, fn)
            if os.path.isdir(p) or fn.endswith(".safetensors") or fn == WEIGHT_INDEX:
                continue
            shutil.move(p, os.path.join(keep, fn))
            moved.append(fn)
        open(os.path.join(keep, ".done"), "w").close()
    copied = []
    for fn in sorted(os.listdir(base)):
        p = os.path.join(base, fn)
        if os.path.isdir(p) or fn.endswith(".safetensors") or fn in (WEIGHT_INDEX, "README.md", ".gitattributes"):
            continue
        shutil.copy2(p, os.path.join(merged, fn))
        copied.append(fn)
    print(f"[finalize] 2. moved to _merger_saved/: {moved}; copied from the starting model: {copied}", flush=True)

    # 3. spot check that the weights changed
    mm, bm = locate(merged), locate(base)
    tot_diff = 0
    probes = probes_of(family, mm, bm)
    for k in probes:
        with safe_open(os.path.join(merged, mm[k]), framework="pt") as f:
            sl = f.get_slice(k)
            a = sl[: min(2048, sl.get_shape()[0])] if len(sl.get_shape()) else f.get_tensor(k)
        with safe_open(os.path.join(base, bm[k]), framework="pt") as f:
            sl = f.get_slice(k)
            b = sl[: min(2048, sl.get_shape()[0])] if len(sl.get_shape()) else f.get_tensor(k)
        nd = int((a != b).sum())
        tot_diff += nd
        print(f"[finalize] 3. {k}: {nd}/{a.numel()} elements differ from the starting model ({nd / a.numel():.2%}), "
              f"max |diff| {(a.float() - b.float()).abs().max().item():.3g}", flush=True)
    if not probes or tot_diff == 0:
        print("!! the checked tensors are identical to the starting model (or none could be compared): training had no effect", flush=True)
        sys.exit(1)

    # 4. full check
    if (family == "qwen3" or full_check) and check_all(merged, base, mm, bm):
        sys.exit(1)
    print("[finalize] done", flush=True)


def check_all(merged, base, mm=None, bm=None, summary_out=None):
    """Step 4: compare every tensor with the starting model; returns the list of problems (empty = ok). Problems: the name sets differ or are
    empty / a shape differs / NaN or Inf on either side / an element exceeds the bound (tested as not (|diff| <= bound), so NaN counts)."""
    mm = mm or locate(merged)
    bm = bm or locate(base)
    ta = float(os.environ.get("VO3_FINALIZE_TOL_ABS", "0.01"))
    tr = float(os.environ.get("VO3_FINALIZE_TOL_REL", str(2 ** -6)))
    problems, kinds = [], []

    def prob(kind, tensor, msg):
        problems.append(msg)
        kinds.append((kind, tensor))

    names_m, names_b = set(mm), set(bm)
    if not names_m or not names_b:
        prob("empty", None, f"empty tensor list: merged {len(names_m)} / start {len(names_b)}")
    only_m, only_b = sorted(names_m - names_b), sorted(names_b - names_m)
    if only_m or only_b:
        prob("names", None, f"tensor names differ: only in merged {len(only_m)} (first 5 {only_m[:5]}), only in start {len(only_b)} (first 5 {only_b[:5]})")
    fm, fb, n_all, n_nf, worst = {}, {}, 0, 0, 0.0

    def get(cache, root, fn, k):
        if fn not in cache:
            cache[fn] = safe_open(os.path.join(root, fn), framework="pt")
        return cache[fn].get_tensor(k)

    inf, CH = float("inf"), 1 << 24   # compare in chunks (converting a large tensor to fp32 at once takes several GB)
    for k in sorted(names_m & names_b):
        A, Bt = get(fm, merged, mm[k], k), get(fb, base, bm[k], k)
        if A.shape != Bt.shape:
            prob("shape", k, f"{k}: shape {tuple(A.shape)} vs start {tuple(Bt.shape)}")
            continue
        n_all += 1
        a1, b1 = A.reshape(-1), Bt.reshape(-1)
        nf_a = nf_b = n = 0
        dm, first, last = 0.0, None, None
        for off in range(0, a1.numel(), CH):
            a, b = a1[off:off + CH].float(), b1[off:off + CH].float()
            nf_a += int((~torch.isfinite(a)).sum())
            nf_b += int((~torch.isfinite(b)).sum())
            d = (a - b).abs()
            over = ~(d <= (ta + tr * b.abs()))   # NaN compares false -> counted as over the bound
            dm = max(dm, float(torch.nan_to_num(d, nan=inf, posinf=inf).max()))
            c = int(over.sum())
            if c:
                idx = over.nonzero().squeeze(1)
                first = off + int(idx[0]) if first is None else first
                last = off + int(idx[-1])
                n += c
            del a, b, d, over
        worst = max(worst, dm)
        if nf_a or nf_b:
            n_nf += 1
            prob("nonfinite", k, f"{k} {tuple(A.shape)}: non-finite values (NaN / Inf) merged {nf_a} / start {nf_b}")
        if n:
            prob("over", k, f"{k} {tuple(A.shape)}: {n} elements over the bound, max |diff| {dm:.4g}, flat indices {first}-{last}")
        del A, Bt, a1, b1
    if n_all == 0:
        prob("nocompare", None, "no tensor could be compared")
    print(f"[finalize] 4. merged {len(names_m)} / start {len(names_b)} tensors, {n_all} compared element-wise (bound {ta:g} + {tr:g} x |start|): "
          f"{len(problems)} problems ({n_nf} tensors with non-finite values); largest |diff| {worst:.4g}", flush=True)
    if problems:
        for l in problems[:20]:
            print(f"[finalize] 4.   {l}", flush=True)
        print(f"!! 4. {len(problems)} problems (missing tensors / non-finite values / changes far beyond a legitimate update)", flush=True)
    if summary_out is not None:
        summary_out.update(n_merged=len(names_m), n_base=len(names_b), n_compared=n_all, worst=worst,
                           problems=[{"kind": k, "tensor": t} for k, t in kinds])
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["qwen25", "qwen3"])
    ap.add_argument("merged")
    ap.add_argument("start")
    ap.add_argument("--full_check", action="store_true", help="also run step 4 for qwen25")
    ap.add_argument("--check_only", action="store_true", help="only step 4, nothing is changed")
    a = ap.parse_args()
    if a.check_only:
        try:
            summ = {}
            probs = check_all(a.merged, a.start, summary_out=summ)
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"!! 4. the check failed to run: {sys.exc_info()[0].__name__}", flush=True)
            sys.exit(2)
        print(f"[finalize] 4. result {json.dumps(summ)}", flush=True)
        sys.exit(3 if probs else 0)
    finalize(a.family, a.merged, a.start, a.full_check)


if __name__ == "__main__":
    main()
