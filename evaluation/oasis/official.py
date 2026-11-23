"""Thin loader for the official Video-Oasis code (sejong-rcv/Video-Oasis @ adaa458): the question list, the official Qwen
wrapper defaults, frame sampler, prompts and answer extraction are read from the official repository, not re-implemented.

  OASIS_REPO    clone of https://github.com/sejong-rcv/Video-Oasis checked out at adaa458
  OASIS_VIDEOS  folder with the benchmark videos in the official layout <benchmark>/videos/... (default: $OASIS_REPO/data/benchmarks/videos)
"""
import ast
import importlib.util
import json
import os
import sys
import types

COMMIT = "adaa458"
REPO = os.environ.get("OASIS_REPO", "")
if not REPO or not os.path.isdir(os.path.join(REPO, "src")):
    raise SystemExit("set OASIS_REPO to a clone of sejong-rcv/Video-Oasis at commit adaa458")
SRC = os.path.join(REPO, "src")
VROOT = os.environ.get("OASIS_VIDEOS", os.path.join(REPO, "data", "benchmarks", "videos"))


def _git_head(repo):
    head = open(os.path.join(repo, ".git", "HEAD")).read().strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(" ", 1)[1]
    if os.path.exists(os.path.join(repo, ".git", ref)):
        return open(os.path.join(repo, ".git", ref)).read().strip()
    for line in open(os.path.join(repo, ".git", "packed-refs")):
        if line.strip().endswith(ref):
            return line.split()[0]
    raise RuntimeError(f"cannot resolve {ref}")


if os.path.isdir(os.path.join(REPO, ".git")):
    assert _git_head(REPO).startswith(COMMIT), f"Video-Oasis must be at commit {COMMIT}"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def _pkg(name):
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m


VP = _load("oasis_official_vision_process", f"{SRC}/lmms_eval/models/model_utils/qwen/vision_process.py")
VI = _load("oasis_official_video_input", f"{SRC}/lmms_eval/models/model_utils/video_input.py")
_EU = _VO = None


def _task_modules():
    global _EU, _VO
    if _EU is None:
        for p in ("lmms_eval", "lmms_eval.tasks", "lmms_eval.tasks._task_utils"):
            _pkg(p)
        _EU = _load("lmms_eval.tasks._task_utils.eval_utils", f"{SRC}/lmms_eval/tasks/_task_utils/eval_utils.py")
        if "lmms_eval.tasks._task_utils.file_utils" not in sys.modules:   # imported by utils.py, unused here
            fu = types.ModuleType("lmms_eval.tasks._task_utils.file_utils")
            fu.generate_submission_file = None
            sys.modules[fu.__name__] = fu
        _VO = _load("oasis_official_v_oasis_utils", f"{SRC}/lmms_eval/tasks/v_oasis/utils.py")
    return _EU, _VO


_WRAPPER_FILE = {"qwen3": f"{SRC}/lmms_eval/models/simple/qwen3_vl.py", "qwen25": f"{SRC}/lmms_eval/models/simple/qwen2_5_vl.py"}
SCRIPT_MAX_FRAMES = 128   # official launch scripts: --model_args max_frames=128
POST_PROMPT = "Put your final answer in \\boxed{}."


def wrapper_defaults(family):
    """Default arguments of the official wrapper __init__ (parsed from the source) with max_frames=128."""
    tree = ast.parse(open(_WRAPPER_FILE[family]).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            args = node.args.args[-len(node.args.defaults):]
            d = {}
            for arg, v in zip(args, node.args.defaults):
                try:
                    d[arg.arg] = eval(compile(ast.Expression(v), "<default>", "eval"), {"__builtins__": {}})
                except Exception:
                    pass
            d["max_frames"] = SCRIPT_MAX_FRAMES
            return d
    raise RuntimeError(f"{family}: wrapper __init__ not found")


def visual_dict(family, path, t0=None, t1=None):
    w = wrapper_defaults(family)
    spec = VI.parse_video_input((path, t0, t1) if t1 is not None else path)
    d = {"type": "video", "video": spec.path, "min_pixels": w["video_min_pixels"], "max_pixels": w["video_max_pixels"],
         "total_pixels": w["video_total_pixels"], "min_frames": w["min_frames"], "max_frames": w["max_frames"], "fps": w["fps"]}
    if w.get("nframes") is not None:
        d["nframes"] = w["nframes"]
        d.pop("fps")
    VI.add_video_bounds(d, spec)
    return d


def system_prompt(family):
    return wrapper_defaults(family)["system_prompt"]


def doc_to_text(doc):
    return _task_modules()[1].v_oasis_doc_to_text(doc, {"pre_prompt": "", "post_prompt": POST_PROMPT})


def postprocess(family, raw):
    """Wrapper post-processing: Qwen3 keeps the text after the last </think>."""
    return raw.split("</think>")[-1].strip() if family == "qwen3" else raw


def extract(text, n_options):
    """Official answer extraction (extract_answer_letter)."""
    return _task_modules()[0].extract_answer_letter(text, n_options)


def bounds(doc):
    """RTV-Bench items are asked at end_time: (start_time, end_time); other items (None, None)."""
    if doc["db"] == "RTV-Bench":
        return doc["start_time"], doc["end_time"]
    return None, None


_BIDX = None


def _basename_index():
    global _BIDX
    if _BIDX is None:
        import glob
        _BIDX = {}
        for bdir in glob.glob(f"{VROOT}/*/videos"):
            bench = bdir.split("/")[-2]
            for f in glob.glob(f"{bdir}/**/*.*", recursive=True):
                _BIDX.setdefault(bench, {}).setdefault(os.path.basename(f), []).append(f)
    return _BIDX


def local_video(official_rel):
    """Official video_path (../data/benchmarks/videos/<rel>) -> local file; falls back to the unique file with the same
    name inside the same benchmark folder (some archives are flattened); None if missing or ambiguous."""
    rel = official_rel.replace("../data/benchmarks/videos/", "")
    p = os.path.join(VROOT, rel)
    if os.path.exists(p):
        return p, "exact"
    cands = _basename_index().get(rel.split("/")[0], {}).get(os.path.basename(rel), [])
    if len(cands) == 1:
        return cands[0], "basename"
    return None, ("ambiguous" if cands else "missing")


def questions():
    """The official 11,031 questions, with uid = '<benchmark>::<qid>' and the local video path (None if unavailable)."""
    rows = json.load(open(f"{SRC}/lmms_eval/video_oasis.json"))
    for x in rows:
        x["uid"] = f"{x['db']}::{x['qid']}"
        x["local_video"], x["video_resolve"] = local_video(x["video_path"])
    return rows
