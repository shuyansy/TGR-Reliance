"""Frame samplers used in the paper, and helpers that build vLLM video requests.

Every sampler returns (frames, meta) where meta["frames_indices"][i] / meta["fps"] is the source timestamp (s) of frames[i].
Deleting frames keeps the timestamps of the remaining frames.

  grid    1 fps, at most `max_frames` (64) frames spread uniformly over the video, long side resized to `max_side` (768 px), OpenCV.
          Used for the backbones and the single-turn TGR models of Table 1.
  qwen25 / qwen3
          The official sampler of the Qwen2.5-VL / Qwen3-VL evaluation (1 fps, 4..128 frames, official pixel limits, decord),
          identical to Video-Oasis and to the LLaMA-Factory training pipeline. Used for our trained models.
"""
import copy
import json
import os

import numpy as np

# official Qwen wrapper defaults (Video-Oasis lmms_eval qwen2_5_vl.py / qwen3_vl.py __init__ + max_frames=128 of the launch scripts)
QWEN_DEFAULTS = {
    "qwen25": {"fps": 1.0, "min_frames": 4, "max_frames": 128, "min_pixels": 100352, "max_pixels": 602112, "total_pixels": 90316800},
    "qwen3": {"fps": 1.0, "min_frames": 4, "max_frames": 128, "min_pixels": 16384, "max_pixels": 786432, "total_pixels": 16777216},
}


def uniform_take(items, k):
    if k <= 0 or len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int).tolist()
    return [items[i] for i in idx]


def sample_grid(path, fps=1.0, max_frames=64, max_side=768, t0=None, t1=None):
    """t0 / t1 (seconds, optional): sample only inside [t0, t1); timestamps are then relative to t0."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(path)
    total, vfps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), float(cap.get(cv2.CAP_PROP_FPS))
    if total <= 0 or vfps <= 0:
        cap.release()
        return [], None
    step = vfps / min(fps, vfps)
    f0 = 0 if t0 is None else min(total - 1, max(0, int(round(t0 * vfps))))
    f1 = total if t1 is None else min(total, max(f0 + 1, int(round(t1 * vfps))))
    cand = [f0 + int(i * step) for i in range(int((f1 - f0) / step))] or [f0]
    if t1 is not None and len(cand) < 2 and f1 - f0 >= 2:
        cand = [f0, f1 - 1]   # window shorter than one sampling step: its first and last frame
    frames, kept = [], []
    for i in uniform_take(cand, max_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        h0, w0 = fr.shape[:2]
        if max_side and max(h0, w0) > max_side:
            sc = max_side / max(h0, w0)
            fr = cv2.resize(fr, (round(w0 * sc), round(h0 * sc)), interpolation=cv2.INTER_AREA)
        frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
        kept.append(i)
    cap.release()
    meta = {"fps": vfps, "duration": (f1 - f0) / vfps, "total_num_frames": len(frames), "frames_indices": [i - f0 for i in kept],
            "video_backend": "opencv", "do_sample_frames": False}
    return frames, meta


def sample_qwen(path, family, max_frames=None, t0=None, t1=None):
    """Official Qwen2.5-VL / Qwen3-VL sampler (process_vision_info of the vendored vision_process.py).
    t0 / t1: optional time window (seconds), passed as the official video_start / video_end (RTV-Bench items of Video-Oasis)."""
    from . import vision_process as VP
    d = dict(QWEN_DEFAULTS[family])
    if max_frames:
        d["max_frames"] = max_frames
    vd = {"type": "video", "video": path, "min_pixels": d["min_pixels"], "max_pixels": d["max_pixels"], "total_pixels": d["total_pixels"],
          "min_frames": d["min_frames"], "max_frames": d["max_frames"], "fps": d["fps"]}
    if t1 is not None:
        vd["video_start"], vd["video_end"] = float(t0), float(t1)
    kw = {"image_patch_size": 16} if family == "qwen3" else {}
    _, vids, vkw = VP.process_vision_info([[{"role": "user", "content": [vd, {"type": "text", "text": "x"}]}]],
                                          return_video_kwargs=True, return_video_metadata=True, **kw)
    if not vids or vids[0] is None:
        return [], None
    video, vm = vids[0]
    n, src_fps, idx = int(video.shape[0]), float(vm["fps"]), [int(i) for i in vm["frames_indices"]]
    if family == "qwen3":     # absolute source-frame indices; the model sees timestamps from the metadata
        okw, drop_idx, dur = dict(vkw), idx, (idx[-1] + 1) / src_fps
    else:                     # Qwen2.5: frames are positioned sequentially from the first sampled frame
        okw = {"do_sample_frames": False, "fps": [n / max(vm["total_num_frames"], 1e-6) * src_fps]}
        drop_idx, dur = [i - idx[0] for i in idx], (idx[-1] - idx[0] + 1) / src_fps
    meta = {"fps": src_fps, "duration": dur, "total_num_frames": n, "frames_indices": drop_idx, "video_backend": "decord",
            "do_sample_frames": False, "_official": {"family": family, "meta": dict(vm), "kwargs": okw}}
    return [video[i] for i in range(n)], meta


def drop_frames(frames, meta, idx_set):
    """Remove the frames whose positions are in idx_set; the remaining frames keep their source timestamps."""
    keep = [i for i in range(len(frames)) if i not in idx_set]
    if not keep:
        return None, None
    m = copy.deepcopy(meta)
    m["frames_indices"] = [meta["frames_indices"][i] for i in keep]
    m["total_num_frames"] = len(keep)
    return [frames[i] for i in keep], m


def _hf_truncated_second_per_grid(fps, temporal_patch_size=2):
    """HF transformers (4.57) truncates second_per_grid_t = temporal_patch_size / fps to an integer for Qwen2.5-VL;
    vLLM 0.11 uses the float. We pass the fps that makes both identical."""
    import torch
    return int(torch.as_tensor(torch.tensor(temporal_patch_size / fps, dtype=torch.float32), dtype=torch.int64).item())


_IS_QWEN2 = {}


def video_request(model_path, prompt, frames, meta):
    """vLLM request for one video + prompt (same construction as the paper's evaluation)."""
    if "_official" in meta:
        import torch
        o = meta["_official"]
        video = torch.stack(frames)
        if o["family"] == "qwen3":
            m = dict(o["meta"])
            m["frames_indices"] = list(meta["frames_indices"])
            m["do_sample_frames"] = False
            return {"prompt": prompt, "multi_modal_data": {"video": [(video, m)]},
                    "mm_processor_kwargs": {"do_resize": False, **o["kwargs"]}}
        kw = dict(o["kwargs"])
        k = _hf_truncated_second_per_grid(float(kw["fps"][0]))
        kw["fps"] = [2 / k if k >= 1 else 1e6]
        return {"prompt": prompt, "multi_modal_data": {"video": [video]}, "mm_processor_kwargs": kw}
    # grid sampler: pass the effective fps of this request (sampled frames / video duration) so that the model's time axis
    # matches the source video, also after frames have been deleted
    fps_eff = meta["total_num_frames"] / max(meta.get("duration", meta["total_num_frames"]), 1e-6)
    item = (np.stack([np.array(f) for f in frames]), copy.deepcopy(meta))
    return {"prompt": prompt, "multi_modal_data": {"video": [item]}, "mm_processor_kwargs": {"fps": float(fps_eff)}}
