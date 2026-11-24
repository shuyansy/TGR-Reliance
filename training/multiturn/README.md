# Multi-turn format (Video-o3)

In the multi-turn format the model first watches the whole video, then calls a crop tool on the evidence span(s) and answers after
viewing the cropped clips. We build on the official code of [Video-o3](https://github.com/MCG-NJU/Video-o3):

| Backbone | Video-o3 code | Our patch |
|---|---|---|
| Qwen2.5-VL-7B-Instruct | `main` @ `e93f563` | `patches/video_o3_qwen25.patch` (+ `patches/transformers_4.52_video_utils.patch` for the RL environment) |
| Qwen3-VL-8B-Instruct | `qwen3vl` @ `9f1df68` | `patches/video_o3_qwen3.patch` |

The Video-o3 repository has no license file, so we only release patches against it. The training recipe is the same as for the
single-turn format: SFT on EviCe, then GRPO with the grounding-only reward `R_IoU`.

## Setup

```bash
git clone https://github.com/MCG-NJU/Video-o3.git Video-o3 && cd Video-o3 && git checkout e93f563
git apply /path/to/TGR-Reliance/training/multiturn/patches/video_o3_qwen25.patch
# Qwen3-VL:
git clone -b qwen3vl https://github.com/MCG-NJU/Video-o3.git Video-o3-qwen3 && cd Video-o3-qwen3 && git checkout 9f1df68
git apply /path/to/TGR-Reliance/training/multiturn/patches/video_o3_qwen3.patch
```

Environments we used (`decord==0.6.0` everywhere):

| | SFT (both backbones) | RL, Qwen2.5-VL | RL, Qwen3-VL |
|---|---|---|---|
| torch | 2.8.0 | 2.7.0 | 2.8.0 |
| transformers | 4.57.1 | 4.52.0 + `transformers_4.52_video_utils.patch` | 4.57.1 |
| vLLM | - | 0.9.2 | 0.11.0 |
| other | deepspeed 0.16.9, flash-attn 2.8.3 | flash-attn 2.7.4.post1, ray 2.46.0 | flash-attn 2.8.3, ray 2.51.1, tensordict 0.10.0 |

The RL versions are those pinned by the respective Video-o3 branch. transformers 4.52.0 stacks videos of equal size but different
frame counts and crashes on an overview plus crops of a short, low-resolution video; the transformers patch backports the upstream fix:

```bash
patch -p1 -d $(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))") < patches/transformers_4.52_video_utils.patch
```

## Data

All data are built from EviCe (`EviCe.json` and the `videos/` folder of the EviCe dataset).

**SFT.** `tools/build_sft_data.py` rewrites every EviCe sample (video, question, evidence spans, answer) as a tool-use dialogue, with the
prompts of Video-o3 (read from `RL/verl/utils/dataset/task_prompt.py`, so SFT, RL and evaluation use the same text):

```
system     Video-o3 multi-round system prompt
user       <video> (full video) + question + options
assistant  <think>[12.40, 18.90]</think>
           <grounding>{"temporal_segment": [12.4, 18.9], "sampling_strategy": "medium"}</grounding>
user       <video> (the crop)                                   ... one crop call per evidence span ...
assistant  <think>[12.40, 18.90]</think>
           <answer>B</answer>
```

Crop intervals are the evidence spans, changed only when needed: clipped to the video, extended to at least 1 s (the tool rejects shorter
crops), overlapping spans merged, at most 5 crops (Video-o3 allows 6 rounds). As in the single-turn format, the thinking contains only the
spans (`--think spans`, default). For Qwen3-VL (`--family qwen3`) the options keep their letters (`A. ...`), as in the RL data and in
evaluation; `--family qwen25` reproduces the data of our Qwen2.5-VL model, whose options were written without letters.

```bash
python tools/build_sft_data.py data/evice_multiturn_qwen25.json --evice /path/to/EviCe/EviCe.json --media_dir /path/to/EviCe \
    --video_o3 /path/to/Video-o3 --family qwen25 --check_duration [--ids sft_ids.txt | --n 4800 --seed 0]
```

**RL.** `tools/build_rl_data.py` writes the json files read by the Video-o3 RL code (question, `A. ...` options, answer, evidence spans
as clues). In the paper, SFT and GRPO use disjoint questions (`--exclude` the SFT ids):

```bash
python tools/build_rl_data.py data/rl_train.json --evice /path/to/EviCe/EviCe.json --media_dir /path/to/EviCe \
    --exclude sft_ids.txt --val_out data/rl_val.json --val_n 256
```

The RL code reads videos as folders of frames at 4 fps (frame k = time k/4 s). `run_grpo.sh` extracts them with
`tools/extract_frames_4fps.py` (resumable) and checks them with `tools/finalize_rl_data.py`.

## SFT

```bash
cd Video-o3/SFT    # Video-o3-qwen3/SFT for Qwen3-VL
FAMILY=qwen25 CONFIG=/path/to/TGR-Reliance/training/multiturn/sft/qwen2_5vl_7b_multiturn.yaml bash /path/to/TGR-Reliance/training/multiturn/run_sft.sh
python /path/to/TGR-Reliance/training/multiturn/tools/make_view.py --family qwen25 \
    saves/qwen2_5vl_7b_multiturn_sft Qwen/Qwen2.5-VL-7B-Instruct saves/qwen2_5vl_7b_multiturn_sft_view
```

Full fine-tuning with the vision tower frozen (Qwen2.5-VL: also the projector; Qwen3-VL: the merger is trained), learning rate 1e-5, cosine
schedule, warmup 0.05, global batch 16, 292 steps (we used 4,800 questions), DeepSpeed ZeRO-3, Video-o3's video budgets (overview at
2 fps, at most 768 frames and 16,384 tokens; crops 2,048 / 4,096 / 6,144 tokens for coarse / medium / fine). For Qwen3-VL, `run_sft.sh` sets
`VO3_Q3_GRID_FRAMES=1`: frames, resizing and timestamps then follow the RL engine exactly.

`make_view.py` builds the starting model for RL: the SFT weights plus the config, processor, chat template and generation config of the
base model (the SFT environment saves these files in a newer format than the Qwen2.5-VL RL environment can read).

## GRPO

```bash
FAMILY=qwen25 VIDEO_O3=/path/to/Video-o3 MODEL_PATH=/path/to/qwen2_5vl_7b_multiturn_sft_view \
    TRAIN=data/rl_train.json VAL=data/rl_val.json FRAMES=/path/to/frames_4fps OUT=outputs/multiturn_qwen25 \
    REWARD=rg bash /path/to/TGR-Reliance/training/multiturn/run_grpo.sh
```

| `REWARD` | Reward of a rollout |
|---|---|
| `rg` (`R_IoU`, recommended) | well-formed: 0.1 + 0.5 · tIoU(union of the executed crops, union of the evidence spans); otherwise 0. Well-formed = Video-o3's structure check passes, the thinking of every round contains only spans, every crop request is accepted by the tool, and the last round gives an answer letter. Correctness is only logged. |
| `official` | Video-o3's reward: format score + 1[correct] · (1 + b), where b = (overlap + 1) / 2 if the answer is correct, the format is perfect and at least one crop was made (overlap = (2 · IoU + IoP + IoG) / 4 of crops vs. evidence), else b = 0. We used it for GRPO without cold start. |

Settings (those of Video-o3, unchanged): 32 questions per update, 16 rollouts per question, groups with identical rewards filtered out
(dynamic sampling), learning rate 1e-6, no KL loss, clip ratios 0.2 / 0.3, at most 6 rounds, one pass over the RL questions, one node with
8 GPUs (141 GB each). For Qwen3-VL the timestamps are text tokens, so the prompt / context / batch token limits are 4,096 larger and the
vLLM memory share is 0.80. The final checkpoint is merged into a Hugging Face folder with the same format as the starting model
(`tools/finalize_merged.py`).

## What the patches change

Both backbones:
- `R_IoU` for the multi-turn format (`VO3_REWARD=rg` in `reward_score/tool_reward_without_round_penalty.py`; the engine passes the duration
  of the frame grid to the reward so that crop requests are checked with the tool's own parser).
- Training loop (`ray_trainer.py`): one complete pass over the data (the original re-created the data iterator inside the loop), final
  validation and checkpoint when the data run out, correct checkpoint step numbers and resuming, old checkpoints deleted only after the new
  one is written (`KEEP_CKPTS`), `TRAINING_DONE.json`; optional dumps of all rollouts (`VO3_DUMP_ROLLOUTS_DIR`), tool-call statistics,
  timing summed over all generation batches of an update, host-memory metrics.
- Host memory: the rollout engine copies containers instead of video tensors, and the dataloader prefetches less (`VO3_PREFETCH_FACTOR=1`);
  by our estimate the original copies and prefetching held about 420 GB of video tensors, and one of our runs was killed.
- Failed video reads (the dataset then silently substitutes another question) and tool errors are logged (`[vo3-read-error]`,
  `[vo3-tool-infra-error]`).

Qwen2.5-VL only: SFT samples each crop at its own fps (`_vo3_crop_fps` in `mm_plugin.py`); context variants (`VO3_CTX=A/B/D`) for an
ablation that is not in the paper; the paper uses the original context (`C`, default).

Qwen3-VL only:
- The training side expands the video placeholders exactly as vLLM does (a timestamp per block of two frames) and computes position ids for
  all video grids; every round is checked token by token against the prompt vLLM actually processed.
- Crops carry absolute timestamps (the original training engine started them at 0, its evaluation used absolute times).
- SFT samples frames with the rules of the RL engine (`VO3_Q3_GRID_FRAMES=1`, `vo3_q3_frames.py`).
- Parameter guard (`VO3_PARAM_GUARD=1`, default): after every update the actor parameters and optimizer state are copied to host memory and
  compared byte by byte before the next update and before saving; any difference stops all ranks (rerun to resume from the last
  checkpoint). In one of our runs a block of vision parameters was overwritten between rollout and update; we did not find the cause, so we
  also disable vLLM cascade attention (`VO3_DISABLE_CASCADE=1`). `finalize_merged.py` additionally compares every merged tensor with the
  starting model.
