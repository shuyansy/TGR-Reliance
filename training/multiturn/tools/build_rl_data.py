"""EviCe -> RL data of the Video-o3 RL code (one json list per split).

  python build_rl_data.py <out.json> --evice EviCe.json --media_dir <EviCe folder> [--ids ids.txt] [--n N --seed S]
      [--exclude ids.txt] [--val_out val.json --val_n N] [--dur_cache cache.json]

Row format (fields of the Video-o3 RL json files):
  {"doc_id", "video": frame folder relative to BASE_IMAGE_DIR (= the video path relative to --media_dir without extension, e.g. videos/evice_00001),
   "question", "options": ["A. ...", ...], "solution": {"answer": "C", "clue": [{"timestamp": [s, e], "text": ""}, ...]},
   "duration": seconds or null (filled by finalize_rl_data.py after frame extraction), "data_source": "tool_reward_without_round_penalty",
   "src_video": absolute path of the source video (used by extract_frames_4fps.py; ignored by the RL code)}
data_source selects the reward function of Video-o3; tool_reward_without_round_penalty is the one Video-o3 uses for all multiple-choice
questions that need the crop tool (with VO3_REWARD=rg the patched function computes our grounding reward instead, see README).
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_sft_data import parse  # noqa: E402  (same EviCe parser as the SFT data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--evice", required=True)
    ap.add_argument("--media_dir", required=True, help="the EviCe folder (the 'videos' paths of EviCe.json are relative to it)")
    ap.add_argument("--ids", default=None, help="file with EviCe ids to include (one per line); default all")
    ap.add_argument("--exclude", default=None, help="file with EviCe ids to leave out (e.g. the SFT questions)")
    ap.add_argument("--n", type=int, default=0, help="keep N random questions (with --seed); 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_out", default=None, help="also write a validation split here")
    ap.add_argument("--val_n", type=int, default=0, help="size of the validation split (drawn before --n, disjoint from the training split)")
    ap.add_argument("--dur_cache", default=None, help="optional json {absolute video path: duration in s}; missing durations are filled later")
    a = ap.parse_args()
    rows = json.load(open(a.evice))
    if a.ids:
        keep = {x.strip() for x in open(a.ids) if x.strip()}
        rows = [r for r in rows if r["id"] in keep]
    if a.exclude:
        drop = {x.strip() for x in open(a.exclude) if x.strip()}
        rows = [r for r in rows if r["id"] not in drop]
    dur = json.load(open(a.dur_cache)) if a.dur_cache else {}
    out, bad = [], {}
    for r in rows:
        p, why = parse(r, a.media_dir)
        if p is None:
            bad[why] = bad.get(why, 0) + 1
            continue
        q, opts, ans, spans, video = p
        rel = r["videos"][0]
        src = os.path.abspath(video)
        out.append({
            "doc_id": r["id"],
            "video": os.path.splitext(rel)[0].lstrip("/"),
            "question": q,
            "options": [f"{chr(65 + i)}. {o}" for i, o in enumerate(opts)],
            "solution": {"answer": ans, "clue": [{"timestamp": [s, e], "text": ""} for s, e in spans]},
            "duration": dur.get(src),
            "data_source": "tool_reward_without_round_penalty",
            "src_video": src,
        })
    rng = random.Random(a.seed)
    order = list(range(len(out)))
    rng.shuffle(order)
    val = []
    if a.val_out and a.val_n:
        val = [out[i] for i in sorted(order[:a.val_n])]
        order = order[a.val_n:]
    train = [out[i] for i in sorted(order[:a.n] if a.n else order)]
    json.dump(train, open(a.out, "w"), ensure_ascii=False)
    msg = f"[build_rl_data] {len(rows)} rows -> train {len(train)} ({a.out})"
    if a.val_out and a.val_n:
        json.dump(val, open(a.val_out, "w"), ensure_ascii=False)
        msg += f", val {len(val)} ({a.val_out})"
    print(msg + f"; dropped {bad}; multi-span questions {sum(len(o['solution']['clue']) > 1 for o in train)}")
    return 1 if not train else 0


if __name__ == "__main__":
    sys.exit(main())
