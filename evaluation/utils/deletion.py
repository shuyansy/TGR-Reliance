"""Deletion arms of ReGAP on the sampled-frame grid (Sec. 3.2 and Appendix G of the paper).

  * cited arm: delete the sampled frames inside the predicted span widened by +-margin (1 s). If the widened span contains
    no sampled frame (long videos, short spans), delete the 1-2 frames whose "cells" (up to the midpoints to their neighbours)
    intersect it.
  * control arm: K (3) contiguous blocks with the same number of frames, outside the cited frames, placed uniformly at random
    with a per-item seed. If the grid has fewer than K such positions, the frames outside the span are concatenated and the
    blocks are taken from that sequence.
  * wide spans: if the cited frames outnumber the remaining ones, the cited arm deletes an evenly spaced subset of the cited
    frames as large as the remainder and the control arm deletes the whole remainder (one block); if nothing remains,
    both arms keep the full video.
"""
import random

from .frames import uniform_take


def cited_frames(meta, span, margin=1.0):
    s, e = span[0] - margin, span[1] + margin
    ts = [fi / meta["fps"] for fi in meta["frames_indices"]]
    S = {i for i, t in enumerate(ts) if s <= t <= e}
    if S or not ts:
        return S
    b = [(ts[i] + ts[i + 1]) / 2 for i in range(len(ts) - 1)]   # cell boundaries
    return {i for i in range(len(ts)) if (b[i - 1] if i > 0 else float("-inf")) <= e and s < (b[i] if i < len(b) else float("inf"))}


def random_blocks(n_frames, excl, d, k, seed_key):
    starts = [i for i in range(0, n_frames - d + 1) if all(j not in excl for j in range(i, i + d))]
    if len(starts) >= k:
        rng = random.Random(seed_key)
        rng.shuffle(starts)
        return [set(range(i, i + d)) for i in starts[:k]], "grid"
    C = [i for i in range(n_frames) if i not in excl]
    pos = list(range(0, len(C) - d + 1))
    if d < 1 or not pos:
        return [], "none"
    rng = random.Random(f"{seed_key}:concat")
    rng.shuffle(pos)
    return [set(C[p:p + d]) for p in pos[:k]], "concat"


def plan(meta, n_frames, span, item_id, k=3, margin=1.0, seed=42, min_kept=1):
    """-> dict(mode, cited=set, controls=[set, ...]) or dict(mode="not_deletable"|"nodelete")."""
    S = cited_frames(meta, span, margin)
    ds, comp = len(S), n_frames - len(S)
    if ds < 1:
        return {"mode": "not_deletable", "n_cited": ds}
    if ds > comp:
        if comp < 1:
            return {"mode": "nodelete", "n_cited": ds}
        if n_frames - comp < min_kept:
            return {"mode": "not_deletable", "n_cited": ds}
        return {"mode": "clamp", "n_cited": ds, "cited": set(uniform_take(sorted(S), comp)), "controls": [set(range(n_frames)) - S]}
    if n_frames - ds < min_kept:
        return {"mode": "not_deletable", "n_cited": ds}
    blocks, how = random_blocks(n_frames, S, ds, k, f"{seed}:{item_id}")
    if not blocks:
        return {"mode": "not_deletable", "n_cited": ds}
    return {"mode": how, "n_cited": ds, "cited": S, "controls": blocks}
