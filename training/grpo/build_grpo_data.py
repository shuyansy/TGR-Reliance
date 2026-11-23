"""EviCe (LLaMA-Factory format) -> verl parquet files for GRPO.

Each row keeps the system prompt and user text of the SFT data verbatim; the video field is the official Qwen visual dict
(video path + official sampling parameters), so frames are sampled during rollout exactly as in SFT and evaluation;
the ground truth (answer letter + evidence spans) is parsed from the SFT target.

  python training/grpo/build_grpo_data.py --evice evice.json --media_dir /path/to/evice_videos --family qwen25 --out_dir data/grpo
  -> data/grpo/train.parquet, data/grpo/val.parquet (--val_size items, random with --seed)
Use --ids to restrict the training pool to a list of EviCe ids (one per line), e.g. the items not used for SFT.
"""
import argparse
import json
import os
import random
import re

OFFICIAL = {   # official lmms-eval Qwen wrapper defaults (max_frames=128 as in the official launch scripts)
    "qwen25": {"min_pixels": 100352, "max_pixels": 602112, "total_pixels": 90316800},
    "qwen3": {"min_pixels": 16384, "max_pixels": 786432, "total_pixels": 16777216},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evice", required=True)
    ap.add_argument("--media_dir", required=True)
    ap.add_argument("--family", choices=sorted(OFFICIAL), required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--val_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ids", default=None)
    a = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = json.load(open(a.evice))
    if a.ids:
        keep = {x.strip() for x in open(a.ids) if x.strip()}
        rows = [r for r in rows if r["id"] in keep]
    d = OFFICIAL[a.family]
    out = []
    for i, r in enumerate(rows):
        system, user, target = (m["content"] for m in r["messages"])
        answer = re.search(r"<answer>\s*([A-Z])\s*</answer>", target).group(1)
        spans = [[float(x), float(y)] for x, y in re.findall(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", target.split("</think>")[0])]
        n_opt = len(re.findall(r"^[A-Z]\. ", user, re.M))
        video = {"type": "video", "video": os.path.join(a.media_dir, r["videos"][0]), "fps": 1.0, "min_frames": 4, "max_frames": 128, **d}
        out.append({"data_source": "tga_rl", "prompt": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "videos": [video], "ability": "video_mcq_grounding",
                    "reward_model": {"style": "rule", "ground_truth": json.dumps({"answer": answer, "spans": spans})},
                    "extra_info": {"id": r["id"], "index": i, "duration": 0.0, "n_options": n_opt, "n_frames": 0}})
    random.Random(a.seed).shuffle(out)
    val, train = out[:a.val_size], out[a.val_size:]
    os.makedirs(a.out_dir, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(train), os.path.join(a.out_dir, "train.parquet"))
    pq.write_table(pa.Table.from_pylist(val), os.path.join(a.out_dir, "val.parquet"))
    print(f"train {len(train)} / val {len(val)} -> {a.out_dir}")


if __name__ == "__main__":
    main()
