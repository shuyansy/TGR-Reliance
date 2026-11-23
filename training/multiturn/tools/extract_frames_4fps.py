"""Extract the 4 fps frame folders read by the Video-o3 RL code (my_qwen_vl_utils._read_video_frame reads the .jpg files of a folder
sorted by name, with source_frames_fps=4.0). The official extraction script of Video-o3 is not public; this implementation follows the
convention the RL code assumes:
  duration = total frames / average fps; n = max(2, floor(duration * 4)) frames; frame k is the source frame at k/4 s:
  index = round(k / 4 * average fps) (at most the last frame). Video-o3 treats frame k as time k/4 s for the time notes and crop intervals.
  Files %06d.jpg (zero padded, lexicographic = temporal order), JPEG quality 95; long side > 1280 resized to 1280 (the model caps a frame at
  768x28x28 ~ 0.6 MP, so this does not change model inputs and saves storage).
  Every video gets a _meta.json (native fps / total frames / duration / extracted frames / size); complete folders are skipped (resumable).
  Occasional EOF errors at the end of a video: DECORD_EOF_RETRY_MAX=40960, then a single-threaded retry; remaining failures are listed and the
  script exits non-zero.
Usage: python extract_frames_4fps.py <frame_root> <n_proc> <data_json> [<data_json> ...]
  data_json: output of build_rl_data.py; the frames of src_video go to <frame_root>/<video> (video = the row's frame folder key).
"""
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

os.environ.setdefault("DECORD_EOF_RETRY_MAX", "40960")
FPS, LONG_SIDE, CHUNK = 4.0, 1280, 64
INDEX_RULE = "k/4s"   # frame k = source video at k/4 s


def extract(args):
    out_root, src, key = args
    import cv2
    import decord
    import numpy as np
    cv2.setNumThreads(1)   # avoid oversubscribing OpenCV's thread pool across processes
    d = os.path.join(out_root, key)
    meta_p = os.path.join(d, "_meta.json")
    if os.path.exists(meta_p):
        try:
            m = json.load(open(meta_p))
            if m.get("index_rule") == INDEX_RULE and len([f for f in os.listdir(d) if f.endswith(".jpg")]) == m["n_extracted"]:
                return src, m, None
        except Exception:
            pass
    t0 = time.time()
    try:
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):   # before re-extracting (incomplete folder) remove leftover frames and metadata
            if f.endswith(".jpg") or f == "_meta.json":
                os.remove(os.path.join(d, f))
        for nt in (2, 1):   # decord with 2 decoding threads (the default 0 = all cores oversubscribes with many processes), then single-threaded
            try:
                vr = decord.VideoReader(src, num_threads=nt)
                total, fps = len(vr), float(vr.get_avg_fps())
                dur = total / fps
                n = max(2, int(np.floor(dur * FPS)))
                idx = np.minimum(np.round(np.arange(n) / FPS * fps).astype(int), total - 1).tolist()
                k = 0
                for s in range(0, n, CHUNK):
                    frames = vr.get_batch(idx[s:s + CHUNK]).asnumpy()
                    for fr in frames:
                        h, w = fr.shape[:2]
                        if max(h, w) > LONG_SIDE:
                            sc = LONG_SIDE / max(h, w)
                            fr = cv2.resize(fr, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
                        cv2.imwrite(os.path.join(d, f"{k:06d}.jpg"), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
                        k += 1
                hh, ww = fr.shape[:2]
                break
            except Exception:
                if nt == 1:
                    raise
                print(f"[vo3-extract-retry] {src}: multi-threaded decoding failed, retrying single-threaded\n{traceback.format_exc()}", flush=True)
        m = {"src": src, "index_rule": INDEX_RULE, "native_fps": fps, "total_frames": total, "duration": dur, "n_extracted": k, "width": ww, "height": hh,
             "sec": round(time.time() - t0, 1)}
        json.dump(m, open(meta_p, "w"))
        return src, m, None
    except Exception:
        return src, None, traceback.format_exc()


def main():
    out_root, n_proc, files = sys.argv[1], int(sys.argv[2]), sys.argv[3:]
    pairs = sorted({(r["src_video"], r["video"]) for f in files for r in json.load(open(f))})
    keys = {}
    for src, key in pairs:   # one frame folder per source video
        assert keys.setdefault(key, src) == src, f"frame folder {key} is used by two videos: {keys[key]} / {src}"
    srcs = [src for src, _ in pairs]
    print(f"[vo3-extract] {len(srcs)} videos -> {out_root}, {n_proc} processes, {FPS} fps", flush=True)
    done, fails, t0 = 0, [], time.time()
    with Pool(n_proc) as pool:
        for src, m, err in pool.imap_unordered(extract, [(out_root, s, k) for s, k in pairs], chunksize=4):
            done += 1
            if err:
                fails.append(src)
                print(f"[vo3-extract-error] {src}\n{err}", flush=True)
            if done % 200 == 0 or done == len(srcs):
                print(f"[vo3-extract] {done}/{len(srcs)}, {len(fails)} failed, {time.time() - t0:.0f} s", flush=True)
    json.dump(fails, open(os.path.join(out_root, "_failed.json"), "w"), indent=1)
    print(f"[vo3-extract] done: {len(srcs) - len(fails)} ok / {len(fails)} failed (list in {out_root}/_failed.json)", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
