"""Video-Oasis accuracy (official micro-average) and ReGAP from the output of eval_oasis.py.

  python evaluation/oasis/summarize_oasis.py results/oasis/<name>_shard*of8.jsonl [--expect 11031]
Accuracy: all questions except those whose video is unavailable locally (no_video); decoding / generation errors count as
wrong. ReGAP: items with p_full, a predicted span, p_span_deleted and p_random_deleted (refusal="exclude").
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import regap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--expect", type=int, default=11031, help="number of official questions (0 = do not check)")
    a = ap.parse_args()
    recs = [json.loads(x) for f in a.files for x in open(f) if x.strip()]
    cnt = collections.Counter(r["id"] for r in recs)
    dup = [k for k, v in cnt.items() if v > 1]
    if dup or (a.expect and len(cnt) != a.expect):
        print(f"!! {len(cnt)} distinct questions (expected {a.expect}), {len(dup)} duplicated: finish or rerun the missing shards")
    scored = [r for r in recs if r["status"] != "no_video"]
    acc = lambda rs: 100.0 * sum(bool(r["correct"]) for r in rs) / max(len(rs), 1)
    print(f"Video-Oasis accuracy {acc(scored):.2f}  (scored {len(scored)}, no_video {len(recs) - len(scored)}, "
          f"status {dict(collections.Counter(r['status'] for r in recs))})")
    by = collections.defaultdict(list)
    for r in scored:
        by[r["db"]].append(r)
    for db in sorted(by):
        print(f"  {db:<18} {acc(by[db]):6.2f}  n={len(by[db])}")
    g = regap.regap(scored, refusal="exclude")
    print(f"Video-Oasis ReGAP {g['regap']:.3f}  (measurable {g['n_measurable']} / {len(scored)}, tau {g['tau']:.3f})")


if __name__ == "__main__":
    main()
