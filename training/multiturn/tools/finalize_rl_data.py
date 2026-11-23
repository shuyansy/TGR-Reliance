"""After frame extraction: write the RL data file actually used for training.
  - checks every frame folder: <frames_root>/<row.video>/_meta.json exists, was written with the k/4 s rule, and the folder holds as many .jpg
    files as recorded (>= 2)
  - sets duration to the duration measured at extraction (total frames / average fps); the Video-o3 code does not read it (its time axis is
    frames / 4 fps), but a batch that mixes null and numbers is not collated reliably, so every row gets a number
  - rows with an incomplete frame folder are dropped and listed; more than --max_drop dropped rows -> exit code 1
Usage: python finalize_rl_data.py <frames_root> <in_json> <out_json> [--max_drop K] [--head N]
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_root")
    ap.add_argument("in_json")
    ap.add_argument("out_json")
    ap.add_argument("--max_drop", type=int, default=20)
    ap.add_argument("--head", type=int, default=0, help="only the first N rows (small smoke-test splits)")
    a = ap.parse_args()
    rows = json.load(open(a.in_json))
    if a.head:
        rows = rows[:a.head]
    out, dropped, dur_diff, checked = [], [], [], {}
    for r in rows:
        d = os.path.join(a.frames_root, r["video"])
        if d not in checked:
            why = None
            try:
                m = json.load(open(os.path.join(d, "_meta.json")))
                n = sum(1 for f in os.listdir(d) if f.endswith(".jpg"))
                if m.get("index_rule") != "k/4s":
                    why = f"old extraction rule {m.get('index_rule')} (frame k must be at k/4 s)"
                elif n != m["n_extracted"] or n < 2:
                    why = f"frame count mismatch: folder {n} / recorded {m['n_extracted']}"
            except Exception as e:
                m, why = None, f"no complete frame folder: {e!r}"
            checked[d] = (m, why)
        m, why = checked[d]
        if why:
            dropped.append((r["doc_id"], r["src_video"], why))
            continue
        if r.get("duration"):
            dur_diff.append(abs(r["duration"] - m["duration"]))
        out.append(dict(r, duration=round(float(m["duration"]), 3)))
    json.dump(out, open(a.out_json, "w"), ensure_ascii=False)
    dur_diff.sort()
    print(f"[rl-data] {a.in_json} -> {a.out_json}: kept {len(out)} / {len(rows)}, dropped {len(dropped)} (incomplete frame folders); "
          f"{len(checked)} videos; |given duration - measured| median {dur_diff[len(dur_diff) // 2] if dur_diff else 0:.2f} s, "
          f"max {dur_diff[-1] if dur_diff else 0:.2f} s (n={len(dur_diff)})", flush=True)
    for x in dropped:
        print(f"[rl-data-drop] {x[0]} {x[1]}: {x[2]}", flush=True)
    if len(dropped) > a.max_drop:
        print(f"!! {len(dropped)} rows dropped > limit {a.max_drop}, stopping", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
