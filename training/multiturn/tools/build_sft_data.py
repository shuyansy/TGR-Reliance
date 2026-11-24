"""EviCe (single-turn: video + multiple-choice question + "<think>[s, e], ...</think><answer>X</answer>") -> multi-turn SFT data in the
Video-o3 tool format, for the Video-o3 SFT code (its LLaMA-Factory fork) followed by Video-o3 RL.

  python build_sft_data.py <out.json> --evice EviCe.json --media_dir <EviCe folder> --video_o3 <Video-o3 repo> --family qwen25|qwen3
      [--think spans|template] [--ids ids.txt] [--n N --seed S] [--check_duration --dur_cache cache.json] [--show K]

Each EviCe sample -> one multi-turn sample (only video / question / evidence spans / answer are used, no generated descriptions):
  system  = Video-o3 SYSTEM_PROMPT_MULTI_ROUND_MC
  user 1  = QUESTION_TEMPLATE_MULTI_ROUND_MC (full <video>, question, options = str(option list); qwen3: options keep their "A. " prefix,
            as in the RL data and in evaluation)
  for every evidence span k (in time order): assistant = <think>...</think>\n<grounding>{"temporal_segment": [s, e], "sampling_strategy": "medium"}</grounding>\n
                                               user = TOOL_CALL_CROP_VIDEO_MULTI_TRUN_PROMPT_TIME_INSTRUCTION (crop k as <video>)
  last assistant = <think>...</think>\n<answer>X</answer>\n
  videos = [{"url": video, "crop": [-1, -1], ...} (full), {"url": video, "crop": [s, e], "sample": "medium"}, ...]; the SFT code samples the
           frames of each interval from the source video and appends the sampling note after <video>.
Crop intervals = the evidence spans, changed only when needed: the end is clipped to the RL grid duration (grid frames / 4); spans shorter than
1 s are extended backwards on the 0.25 s grid (the tool parser requires start <= end - 1); overlapping spans are merged; at most 5 crops (the
engine allows 6 rounds = 5 crops + answer): beyond that the two closest neighbours are merged repeatedly (no question is dropped).
Prompts are read from Video-o3's RL/verl/utils/dataset/task_prompt.py (identical to RL and evaluation).
Thinking text: spans (default, used in the paper: crop rounds write "[s, e]", the answer round writes all spans, matching the single-turn format)
or template (short template sentences).
"""
import argparse
import collections
import importlib.util
import json
import math
import os
import random
import re
import sys

VIDEO_TOKENS = "<|vision_start|><|video_pad|><|vision_end|>"
MAX_CROPS, MIN_LEN, STRATEGY, GRID = 5, 1.0, "medium", 4.0
RE_USER = re.compile(r"^<video>Question: (.*?)\nOptions:\n(.*?)\nReturn the relevant time segment", re.S)
RE_OPT = re.compile(r"^([A-Z])\. (.*)$")
RE_ASSIST = re.compile(r"^<think>(.*?)</think><answer>([A-Z])</answer>$", re.S)
RE_SPAN = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]")


def load_prompts(video_o3):
    spec = importlib.util.spec_from_file_location("vo3_task_prompt", os.path.join(video_o3, "RL/verl/utils/dataset/task_prompt.py"))
    tp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tp)
    q1, tool = tp.QUESTION_TEMPLATE_MULTI_ROUND_MC, tp.TOOL_CALL_CROP_VIDEO_MULTI_TRUN_PROMPT_TIME_INSTRUCTION
    assert q1.count("<video>") == 1 and tool.count(VIDEO_TOKENS) == 1
    return tp.SYSTEM_PROMPT_MULTI_ROUND_MC, q1, tool


def parse(row, media_dir):
    """EviCe row -> ((question, options, answer, spans, video path), None) or (None, reason)."""
    msgs = {m["role"]: m["content"] for m in row["messages"]}
    mu, ma = RE_USER.match(msgs.get("user", "")), RE_ASSIST.match(msgs.get("assistant", ""))
    if not mu:
        return None, "question not parsable"
    if not ma:
        return None, "answer not parsable"
    opts = []
    for i, line in enumerate(mu.group(2).split("\n")):
        mo = RE_OPT.match(line)
        if not mo or mo.group(1) != chr(65 + i):
            return None, "options not parsable"
        opts.append(mo.group(2))
    ans = ma.group(2)
    if ord(ans) - 65 >= len(opts):
        return None, "answer out of range"
    spans = [(float(a), float(b)) for a, b in RE_SPAN.findall(ma.group(1))]
    if not spans:
        return None, "no evidence span"
    if len(row.get("videos") or []) != 1:
        return None, "not exactly one video"
    return (mu.group(1), opts, ans, spans, os.path.join(media_dir, row["videos"][0])), None


def grid_len(dur):
    """Video duration as seen by the RL tool = 4 fps grid frames / 4 (grid frames = max(2, floor(duration * 4)), as extract_frames_4fps).
    The tool parser clips the end to this length and requires start <= end - 1."""
    return max(2, math.floor(dur * GRID)) / GRID


def crops_of(spans, dur=None):
    """Evidence spans -> crop intervals. Original values are kept and changed only when needed:
    (1) an end beyond the grid duration grid_len(dur) is clipped (multiple of 0.25); (2) sorted and overlapping spans merged; more than 5 crops:
    the two closest neighbours are merged repeatedly; (3) shorter than 1 s (tested with the parser's own float expression a <= b - 1): the start
    moves back to floor(4 (b - 1)) / 4 on the 0.25 s grid; if still too short (end < 1 s) the end moves to a + 1.
    Returns None if the grid duration is shorter than 1 s (the tool cannot crop; the caller drops the sample). dur=None: no clipping."""
    L = grid_len(dur) if dur is not None else None
    out = []
    for a, b in sorted(spans):
        if b < a:
            a, b = b, a
        a = max(0.0, a)
        if L is not None and b > L:
            b = L
            a = min(a, b)
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    while len(out) > MAX_CROPS:   # engine limit (5 crops): merge the two closest neighbours (the merged crop covers both and the gap)
        k = min(range(len(out) - 1), key=lambda i: out[i + 1][0] - out[i][1])
        out[k:k + 2] = [(out[k][0], out[k + 1][1])]
    fixed = []
    for a, b in out:
        if not a <= b - MIN_LEN:   # same expression as the parser (t_start <= t_end - 1)
            a = max(0.0, math.floor((b - MIN_LEN) * GRID) / GRID)
        if not a <= b - MIN_LEN:
            b = a + MIN_LEN if L is None else min(L, a + MIN_LEN)
        if not a <= b - MIN_LEN:
            return None
        if fixed and a <= fixed[-1][1]:   # overlaps the previous crop after moving the start: merge
            fixed[-1] = (fixed[-1][0], max(fixed[-1][1], b))
        else:
            fixed.append((a, b))
    return fixed


def fmt(x):
    return f"{x:.2f}"


def thinks(mode, crops, ans):
    """Thinking before every crop call and before the answer."""
    if mode == "spans":
        pre = [f"[{fmt(a)}, {fmt(b)}]" for a, b in crops]
        return pre, ", ".join(pre)
    pre = []
    for k, (a, b) in enumerate(crops):
        if k == 0:
            pre.append(f"To answer the question, I need to locate the part of the video that shows the relevant evidence. "
                       f"It appears around {fmt(a)} to {fmt(b)} seconds, so I will examine this segment in detail.")
        else:
            pre.append(f"The previous segment provides part of the evidence. I also need to examine {fmt(a)} to {fmt(b)} seconds.")
    seg = "segment contains" if len(crops) == 1 else "segments contain"
    return pre, f"The observed {seg} the evidence needed to answer the question, so the answer is option {ans}."


def build(rid, parsed, mode, prompts, letter_options, dur=None):
    system, q1, tool = prompts
    q, opts, ans, spans, video = parsed
    crops = crops_of(spans, dur)
    if crops is None:
        return None, "grid duration shorter than 1 s"
    pre, final = thinks(mode, crops, ans)
    shown = [f"{chr(65 + i)}. {o}" for i, o in enumerate(opts)] if letter_options else opts
    msgs = [{"role": "user", "content": q1.format(question=q, options=str(shown))}]
    videos = [{"url": video, "crop": [-1.0, -1.0], "sample": STRATEGY}]   # full video also as a dict (datasets needs one column type); crop [-1, -1] = no crop
    for k, ((a, b), t) in enumerate(zip(crops, pre)):
        g = json.dumps({"temporal_segment": [a, b], "sampling_strategy": STRATEGY})
        msgs.append({"role": "assistant", "content": f"<think>{t}</think>\n<grounding>{g}</grounding>\n"})
        msgs.append({"role": "user", "content": tool.format(action_turn=k + 1, observation_turn=k + 1, time_instruction="").replace(VIDEO_TOKENS, "<video>")})
        videos.append({"url": video, "crop": [a, b], "sample": STRATEGY})
    msgs.append({"role": "assistant", "content": f"<think>{final}</think>\n<answer>{ans}</answer>\n"})
    return {"id": rid, "system": system, "messages": msgs, "videos": videos, "meta": {"gt_spans": spans, "crops": crops, "answer": ans}}, None


DUR = {}


def duration_of(path):
    """Video duration (decord frames / average fps, the duration used by extract_frames_4fps and the RL grid); cached in DUR."""
    if path not in DUR:
        import decord
        vr = decord.VideoReader(path, num_threads=1)
        DUR[path] = len(vr) / float(vr.get_avg_fps())
        del vr
    return DUR[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--evice", required=True)
    ap.add_argument("--media_dir", required=True, help="the EviCe folder (the 'videos' paths of EviCe.json are relative to it)")
    ap.add_argument("--video_o3", required=True, help="Video-o3 repository (patched), for RL/verl/utils/dataset/task_prompt.py")
    ap.add_argument("--family", required=True, choices=["qwen25", "qwen3"], help="qwen3: options keep their letter prefix (as in RL / evaluation)")
    ap.add_argument("--think", default="spans", choices=["spans", "template"])
    ap.add_argument("--ids", default=None, help="file with EviCe ids to include (one per line); default all")
    ap.add_argument("--n", type=int, default=0, help="only the first N rows (or N random rows with --seed); 0 = all")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--check_duration", action="store_true", help="read video durations and clip crops to the video (opens every video)")
    ap.add_argument("--dur_cache", default=None, help="json cache of video durations (read if present, written at the end)")
    ap.add_argument("--show", type=int, default=0, help="print the first K samples")
    a = ap.parse_args()
    prompts = load_prompts(a.video_o3)
    if a.dur_cache and os.path.exists(a.dur_cache):
        DUR.update(json.load(open(a.dur_cache)))
    rows = json.load(open(a.evice))
    if a.ids:
        keep = {x.strip() for x in open(a.ids) if x.strip()}
        rows = [r for r in rows if r["id"] in keep]
    idxs = list(range(len(rows)))
    if a.n:
        idxs = sorted(random.Random(a.seed).sample(idxs, a.n)) if a.seed is not None else idxs[:a.n]
    out, bad, ncrop = [], collections.Counter(), collections.Counter()
    for i in idxs:
        p, why = parse(rows[i], a.media_dir)
        if p is None:
            bad[why] += 1
            continue
        dur = None
        if a.check_duration:
            try:
                dur = duration_of(p[4])
            except Exception as e:
                bad[f"cannot open video ({type(e).__name__})"] += 1
                continue
            if min(s for s, _ in p[3]) >= dur:
                bad["evidence outside the video"] += 1
                continue
        r, why = build(rows[i]["id"], p, a.think, prompts, a.family == "qwen3", dur)
        if r is None:
            bad[why] += 1
            continue
        ncrop[len(r["meta"]["crops"])] += 1
        out.append(r)
    json.dump(out, open(a.out, "w"), ensure_ascii=False, indent=1 if len(out) <= 50 else None)
    if a.dur_cache:
        json.dump(DUR, open(a.dur_cache, "w"))
    print(f"[build_sft_data] {len(idxs)} rows -> {len(out)} samples ({a.out}); dropped {dict(bad)}; crops per sample {dict(sorted(ncrop.items()))}")
    for r in out[:a.show]:
        print("\n" + "=" * 100 + f"\n[{r['id']}] evidence {r['meta']['gt_spans']} -> crops {r['meta']['crops']}, answer {r['meta']['answer']}")
        print(f"videos = {json.dumps(r['videos'], ensure_ascii=False)}\n--- system ---\n{r['system']}")
        for m in r["messages"]:
            print(f"--- {m['role']} ---\n{m['content']}")
    return 1 if not out else 0


if __name__ == "__main__":
    sys.exit(main())
