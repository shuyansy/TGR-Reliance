# Training

Our recipe (Sec. 4.4 of the paper): supervised fine-tuning on EviCe, then GRPO rewarded **only for temporal grounding**
(`R_IoU`). Both stages use the single-turn format `<think>[s, e], ...</think><answer>X</answer>` and the official
Qwen frame sampler (1 fps, at most 128 frames), so training inputs are identical to evaluation inputs.

| Stage | Framework | Folder |
|---|---|---|
| SFT | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) @ `3140c242f0` + `sft/llamafactory.patch` | `sft/` |
| GRPO | [verl](https://github.com/volcengine/verl) @ `f400fb7654b2` + `grpo/verl.patch` | `grpo/` |

Tested with `torch==2.8.0`, `transformers==4.57.1`, `vllm==0.11.0`, `decord==0.6.0`.

## 1. SFT

```bash
cd LLaMA-Factory   # patched, see sft/run_sft.sh
FAMILY=qwen25 CONFIG=/path/to/TGR-Reliance/training/sft/qwen2_5vl_7b_evice.yaml bash /path/to/TGR-Reliance/training/sft/run_sft.sh
FAMILY=qwen3  CONFIG=/path/to/TGR-Reliance/training/sft/qwen3vl_8b_evice.yaml   bash /path/to/TGR-Reliance/training/sft/run_sft.sh
```

Full fine-tuning of the language model (Qwen3-VL: also the merger), vision tower frozen, learning rate 1e-5, cosine schedule,
global batch 16, bf16, DeepSpeed ZeRO-3. When SFT is followed by GRPO, we stop SFT early (about one third of an epoch,
e.g. `max_steps` in the yaml) and keep the questions used for GRPO out of the SFT data.

## 2. GRPO

```bash
python training/grpo/build_grpo_data.py --evice evice.json --media_dir /path/to/evice_videos --family qwen25 --out_dir data/grpo_qwen25
cd verl   # patched, see grpo/run_grpo.sh
FAMILY=qwen25 INIT_MODEL=/path/to/sft_checkpoint DATA=data/grpo_qwen25 OUT=outputs/grpo_qwen25 \
    REWARD=compute_score_iou bash /path/to/TGR-Reliance/training/grpo/run_grpo.sh
```

Rewards (`grpo/reward.py`); a response that does not follow the format gets 0:

| Name in the paper | `REWARD` | Value for a well-formed response |
|---|---|---|
| `R_ans` | `compute_score_ans` | 1[correct] + 0.1 |
| `R_ans+IoU` | `compute_score_ans_iou` | 1[correct] + 0.1 + 0.5 · IoU |
| `R_IoU` (recommended) | `compute_score_iou` | 0.1 + 0.5 · IoU |

GRPO settings: 8 rollouts per question, global batch 16, learning rate 1e-6, no KL term, at most 256 response tokens,
FSDP2, vLLM rollout; we trained 323 steps on 2 nodes x 8 GPUs.

## 3. Multi-turn format

Coming soon (based on [Video-o3](https://github.com/MCG-NJU/Video-o3)).
