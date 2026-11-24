"""Reliance Gap (ReGAP), Eq. (1)-(2) of the paper.

Per item i (measurable items only):
    c_i = logit( E_k p(v \\ r_k) ) - logit( p(v \\ s_hat) )
    ReGAP = mean_i(c_i) / tau,   tau = mean_i |logit p(v)|
where p(.) is the model's probability for its own full-video answer, renormalised over the answer options,
s_hat its predicted span and r_k (k = 1..K) random contiguous blocks of the same length outside s_hat.

Conventions (identical to the paper's tables):
  * probabilities are clipped to [1e-4, 1 - 1e-4] before the logit;
  * random blocks after which the model gives no option letter are left out of the average over k;
    if no block gives a letter, the item is not measurable;
  * if the model gives no option letter after its span is deleted ("refusal"), p(v \\ s_hat) = 1 / n_options
    (ReliaNeedle, refusal="chance"); on Video-Oasis the evaluator re-reads the probabilities at the answer slot and items
    that remain without a value are left out (refusal="exclude");
  * items without a full-video answer or without a predicted span are not measurable (they still count for accuracy);
  * Hit / Miss: union IoU between the predicted spans and the ground-truth span >= 0.3 / < 0.3.

Item format (one dict per item, as written by evaluation/eval_relianeedle.py):
    {"id": str, "gt_span": [s, e] | None, "pred_spans": [[s, e], ...], "correct": bool,
     "n_options": int, "p_full": float | None, "p_span_deleted": float | None,
     "p_random_deleted": float | None   (or "p_random_blocks": [float | None, ...]),
     "status": optional, recomputed when missing}
"""
import math
import random

CLIP = 1e-4
HIT_IOU = 0.3


def logit(p, clip=CLIP):
    p = min(max(float(p), clip), 1.0 - clip)
    return math.log(p / (1.0 - p))


def mean_random(blocks):
    """Average over the K random blocks; blocks without an option letter (None) are left out."""
    vals = [float(p) for p in (blocks or []) if p is not None]
    return sum(vals) / len(vals) if vals else None


def _p_random(item):
    if item.get("p_random_deleted") is not None:
        return float(item["p_random_deleted"])
    return mean_random(item.get("p_random_blocks"))


def item_status(item):
    """ok / refusal (measurable) or no_answer / no_span / not_deletable (not measurable)."""
    if item.get("p_full") is None:
        return "no_answer"
    if not item.get("pred_spans"):
        return "no_span"
    if _p_random(item) is None:
        return "not_deletable"
    return "ok" if item.get("p_span_deleted") is not None else "refusal"


def item_gap(item, clip=CLIP, refusal="chance"):
    """c_i of Eq. (1), or None when the item is not measurable."""
    st = item_status(item)
    if st not in ("ok", "refusal") or (st == "refusal" and refusal == "exclude"):
        return None
    k = int(item.get("n_options") or 4)
    p_span = float(item["p_span_deleted"]) if st == "ok" else 1.0 / k
    return logit(_p_random(item), clip) - logit(p_span, clip)


def _spans(x):
    if not x:
        return []
    if isinstance(x[0], (int, float)):
        x = [x]
    return [[float(a), float(b)] for a, b in x if float(b) > float(a)]


def span_iou(pred_spans, gt_span):
    """IoU between the union of the predicted spans and the ground-truth span."""
    cs = sorted(_spans(pred_spans))
    if not cs or not gt_span:
        return 0.0
    merged = [cs[0][:]]
    for a, b in cs[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    g0, g1 = float(gt_span[0]), float(gt_span[1])
    inter = sum(max(0.0, min(b, g1) - max(a, g0)) for a, b in merged)
    union = sum(b - a for a, b in merged) + (g1 - g0) - inter
    return inter / union if union > 0 else 0.0


def accuracy(items):
    return 100.0 * sum(bool(it.get("correct")) for it in items) / len(items)


def regap(items, clip=CLIP, tau=None, refusal="chance"):
    """ReGAP over the measurable items. Returns dict(regap, tau, n_measurable); tau can be fixed by the caller."""
    meas = [(it, item_gap(it, clip, refusal)) for it in items]
    meas = [(it, c) for it, c in meas if c is not None]
    if not meas:
        return {"regap": float("nan"), "tau": float("nan"), "n_measurable": 0}
    if tau is None:
        tau = sum(abs(logit(it["p_full"], clip)) for it, _ in meas) / len(meas)
    return {"regap": sum(c for _, c in meas) / len(meas) / tau, "tau": tau, "n_measurable": len(meas)}


def summarize(items, clip=CLIP, hit_iou=HIT_IOU, refusal="chance"):
    """The columns of Table 1: Acc. (all items), R@0.3 and ReGAP All / Hit / Miss (measurable items)."""
    r = regap(items, clip, refusal=refusal)
    tau = r["tau"]
    meas = [it for it in items if item_gap(it, clip, refusal) is not None]
    hit = [it for it in meas if it.get("gt_span") and span_iou(it.get("pred_spans"), it["gt_span"]) >= hit_iou]
    miss = [it for it in meas if it not in hit]
    part = lambda xs: (sum(item_gap(it, clip, refusal) for it in xs) / len(xs) / tau) if xs else float("nan")
    return {"n": len(items), "acc": accuracy(items), "n_measurable": len(meas),
            "r_at_0.3": 100.0 * len(hit) / len(meas) if meas else float("nan"),
            "regap": r["regap"], "regap_hit": part(hit), "regap_miss": part(miss), "tau": tau, "n_hit": len(hit), "n_miss": len(miss)}


def bootstrap(items, n_boot=1000, seed=0, clip=CLIP, alpha=0.05, refusal="chance"):
    """Item-level bootstrap CI of ReGAP; tau is re-estimated in every resample."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [items[rng.randrange(len(items))] for _ in items]
        v = regap(s, clip, refusal=refusal)["regap"]
        if v == v:
            vals.append(v)
    vals.sort()
    lo, hi = vals[int(alpha / 2 * len(vals))], vals[int((1 - alpha / 2) * len(vals)) - 1]
    return {"regap": regap(items, clip, refusal=refusal)["regap"], "ci_low": lo, "ci_high": hi, "n_boot": len(vals)}
