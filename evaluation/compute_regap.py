"""Table 1 columns from per-item result files (output of eval_relianeedle.py, or the files in examples/).

  python evaluation/compute_regap.py results/model_a.jsonl results/model_b.jsonl [--bootstrap 1000]
Acc. is over all items; R@0.3 and ReGAP (All / Hit / Miss) are over the items where ReGAP is defined (see regap/metric.py).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import regap  # noqa: E402


def load(path):
    if path.endswith(".jsonl"):
        return [json.loads(x) for x in open(path) if x.strip()]
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--bootstrap", type=int, default=0, help="number of bootstrap resamples for a 95%% CI of ReGAP (tau re-estimated)")
    ap.add_argument("--expect", type=int, default=725, help="number of items in the benchmark (0 = do not check)")
    a = ap.parse_args()
    bad = 0
    print(f"{'model':32s} {'Acc.':>6s} {'R@0.3':>6s} {'ReGAP':>7s} {'Hit':>7s} {'Miss':>7s} {'meas.':>6s}" + ("   95% CI" if a.bootstrap else ""))
    for f in a.files:
        items = load(f)
        err = [it["id"] for it in items if it.get("status") == "error"]
        if (a.expect and len(items) != a.expect) or err:
            print(f"!! {f}: {len(items)} items, {len(err)} with errors (e.g. {err[:3]}) -- finish or rerun them before reporting")
            bad += 1
        s = regap.summarize(items)
        line = f"{os.path.basename(f)[:32]:32s} {s['acc']:6.1f} {s['r_at_0.3']:6.1f} {s['regap']:7.3f} {s['regap_hit']:7.3f} {s['regap_miss']:7.3f} {s['n_measurable']:6d}"
        if a.bootstrap:
            b = regap.bootstrap(items, a.bootstrap)
            line += f"   [{b['ci_low']:.3f}, {b['ci_high']:.3f}]"
        print(line)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
