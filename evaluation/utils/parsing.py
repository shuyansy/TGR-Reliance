"""Parsing of the model output: answer letter, predicted span, and the probability distribution over the option letters
at the answer slot (read from the top-20 logprobs of the generated answer token)."""
import math
import re
import string

LETTERS = list(string.ascii_uppercase)
THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
IV_RE = re.compile(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)\s*\]")
LET_RE = re.compile(r"[^A-Za-z0-9]*([A-Z])(?![A-Za-z0-9])")


def parse_letter(text, n):
    """Option letter of a response: the letter inside <answer>, else the first standalone letter (after </think> if present)."""
    t = text or ""
    if "<think>" in t and "</think>" not in t:
        return None
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[-1]
    hi = LETTERS[n - 1]
    m = re.search(rf"<answer>\s*([A-{hi}])\b", t, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(rf"\b([A-{hi}])\b", t)
    return m.group(1).upper() if m else None


def parse_span(text, duration):
    """Predicted evidence: the envelope of all [start, end] pairs written inside <think> (clipped to the video)."""
    src = THINK_RE.search(text or "")
    spans = []
    for a, b in IV_RE.findall(src.group(1) if src else (text or "")):
        try:
            s, e = float(a), float(b)
        except ValueError:
            continue
        if e < s:
            s, e = e, s
        s, e = max(0.0, min(s, duration)), max(0.0, min(e, duration))
        if e > s:
            spans.append([s, e])
    if not spans:
        return None
    return [min(s for s, _ in spans), max(e for _, e in spans)]


def _piece_letter(piece):
    m = LET_RE.match(piece)
    return m.group(1) if m else None


def letter_dist(output, tokenizer, n, hint=None):
    """Renormalised probabilities of the option letters at the answer slot of a vLLM output (needs logprobs=20).
    Answer slot: first letter token inside <answer>...</answer>; else the first letter right after the last \\boxed{;
    else the last token equal to `hint` (the parsed answer letter). None if there is no such token."""
    o = output.outputs[0]
    if o.logprobs is None:
        return None
    pieces = [tokenizer.decode([t]) for t in o.token_ids]
    text, cands, starts = "", [], []
    for i, p in enumerate(pieces):
        starts.append(len(text))
        text += p
        L = _piece_letter(p)
        if L:
            cands.append((i, L))
    ends = [starts[i] + len(pieces[i]) for i in range(len(pieces))]
    target, low = None, text.lower()
    a0 = low.find("<answer>")
    if a0 >= 0:
        a1 = low.find("</answer>", a0 + 8)
        a1 = len(text) if a1 < 0 else a1
        inside = [i for i, _ in cands if ends[i] > a0 + 8 and starts[i] < a1]
        if not inside:
            return None
        target = inside[0]
    else:
        b0 = text.rfind("\\boxed{")
        if b0 >= 0:
            after = [i for i, _ in cands if ends[i] > b0 + 7 and re.fullmatch(r"[\s(]*", text[b0 + 7:max(b0 + 7, starts[i])] or "")]
            target = after[0] if after else None
        if target is None and hint:
            hits = [i for i, L in cands if L == hint]
            target = hits[-1] if hits else None
    if target is None or target >= len(o.logprobs):
        return None
    probs = {L: 0.0 for L in LETTERS[:n]}
    for tid, info in o.logprobs[target].items():
        L = _piece_letter(tokenizer.decode([tid]))
        if L and L in probs:
            probs[L] += math.exp(info.logprob)
    Z = sum(probs.values())
    return {k: v / Z for k, v in probs.items()} if Z > 1e-9 else None
