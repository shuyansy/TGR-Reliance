"""GRPO rewards of the paper (Sec. 4.4 and Appendix H), for verl's custom reward function.

The response must have the single-turn format  <think>[s1, e1], [s2, e2]</think><answer>X</answer>:
the think region contains only time spans (seconds) and the answer is one option letter. A response in any other format gets 0.
For well-formed responses:
    R_ans       = 1[answer correct] + 0.1
    R_ans+IoU   = 1[answer correct] + 0.1 + 0.5 * IoU
    R_IoU       = 0.1 + 0.5 * IoU          (grounding only; the answer is never rewarded; recommended)
IoU is the temporal IoU between the union of the predicted spans and the union of the ground-truth evidence spans.
ground_truth is a JSON string {"answer": "B", "spans": [[s, e], ...]} (see build_grpo_data.py).

verl: reward.custom_reward_function.path=training/grpo/reward.py
      reward.custom_reward_function.name=compute_score_iou | compute_score_ans_iou | compute_score_ans
"""
import json
import re

_ANS = re.compile(r"<answer>\s*\(?([A-Z])\)?\s*</answer>", re.S)
_SPAN = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")


def _answer(text):
    m = _ANS.findall(text or "")
    return m[-1] if m else None


def _union(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def temporal_iou(pred, gt):
    P, G = _union(pred), _union(gt)
    if not P or not G:
        return 0.0
    inter = sum(max(0.0, min(b1, b2) - max(a1, a2)) for a1, b1 in P for a2, b2 in G)
    union = sum(b - a for a, b in P) + sum(b - a for a, b in G) - inter
    return inter / union if union > 0 else 0.0


def think_region(text):
    """Text before the first </think> (the <think> tag removed); None unless an <answer> follows it."""
    text = text or ""
    i = text.find("</think>")
    if i < 0 or "<answer>" not in text[i:]:
        return None
    return text[:i].replace("<think>", " ")


def _spans(region):
    sp = [(float(a), float(b)) for a, b in _SPAN.findall(region or "")]
    return [(a, b) for a, b in sp if b > a]


def well_formed(text):
    region = think_region(text)
    if region is None or _answer(text) is None or not _spans(region):
        return False
    return re.sub(r"[\s,;，、；]", "", _SPAN.sub(" ", region)) == ""   # only spans (and separators) inside <think>


def score(solution_str, ground_truth, reward="iou"):
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    ans = _answer(solution_str)
    spans = _spans(think_region(solution_str) or "")
    fmt = well_formed(solution_str)
    acc = float(ans is not None and ans == str(gt["answer"]).strip().upper())
    iou = temporal_iou(spans, gt["spans"]) if spans else 0.0
    if not fmt:
        s = 0.0
    elif reward == "iou":
        s = 0.1 + 0.5 * iou
    elif reward == "ans_iou":
        s = acc + 0.1 + 0.5 * iou
    elif reward == "ans":
        s = acc + 0.1
    else:
        raise ValueError(reward)
    return {"score": s, "acc": acc, "fmt": float(fmt), "iou": iou, "span_width": sum(b - a for a, b in _union(spans)) if spans else 0.0}


def compute_score_iou(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return score(solution_str, ground_truth, "iou")


def compute_score_ans_iou(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return score(solution_str, ground_truth, "ans_iou")


def compute_score_ans(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return score(solution_str, ground_truth, "ans")
