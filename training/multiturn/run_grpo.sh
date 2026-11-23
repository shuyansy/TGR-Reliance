#!/bin/bash
# Multi-turn GRPO with the RL code of Video-o3 (example; one node with 8 GPUs, the settings of the paper).
#
# One-time setup: patched Video-o3 checkout and environment, see README.md. Inputs:
#   MODEL_PATH  starting model: the SFT view (tools/make_view.py), or the base model for RL only
#   TRAIN, VAL  RL data written by tools/build_rl_data.py
#   FRAMES      folder for the 4 fps frame folders (filled here, resumable; can be shared between runs)
#   VIDEO_O3    the patched Video-o3 checkout (main @ e93f563 for qwen25, qwen3vl @ 9f1df68 for qwen3)
# Usage:
#   FAMILY=qwen25 VIDEO_O3=/path/to/Video-o3 MODEL_PATH=saves/qwen2_5vl_7b_multiturn_sft_view TRAIN=data/rl_train.json VAL=data/rl_val.json \
#       FRAMES=/path/to/frames_4fps OUT=outputs/multiturn_qwen25 bash run_grpo.sh
#   REWARD=rg (default; ours: format gate x (0.1 + 0.5 tIoU(crops, evidence)), no answer reward) | official (Video-o3 reward)
# One pass over TRAIN (each question sampled once; groups whose 16 rollouts all get the same reward are filtered out, so the number of updates
# depends on the data). The final checkpoint is merged into a Hugging Face folder: $OUT/global_step_<N>/actor/huggingface.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FAMILY=${FAMILY:?qwen25 or qwen3}; VIDEO_O3=${VIDEO_O3:?patched Video-o3 checkout}; MODEL_PATH=${MODEL_PATH:?starting model}
TRAIN=${TRAIN:?training json}; VAL=${VAL:?validation json}; FRAMES=${FRAMES:?frame folder root}; OUT=${OUT:?output folder}
CODE=$(cd "$VIDEO_O3/RL" && pwd); PY=${PY:-python}
export VO3_REWARD=${REWARD:-rg} VO3_CTX=C
mkdir -p "$OUT/data" "$OUT/logs" "$OUT/wandb"
if [ "$FAMILY" = qwen25 ]; then   # Video-o3 (main) pins transformers 4.52.0, which needs patches/transformers_4.52_video_utils.patch
  "$PY" -c "import inspect, transformers; from transformers import video_utils as V; assert 'num_frames = video.shape[-4]' in inspect.getsource(V.group_videos_by_shape)" \
    || { echo "apply patches/transformers_4.52_video_utils.patch to transformers first (see README)"; exit 1; }
fi

echo "== 1. frames (4 fps) and data"
NPROC=${EXTRACT_PROCS:-16}
"$PY" "$HERE/tools/extract_frames_4fps.py" "$FRAMES" "$NPROC" "$TRAIN" "$VAL" > "$OUT/extract.log" 2>&1 || { echo "frame extraction failed for some videos, see $OUT/extract.log"; }
"$PY" "$HERE/tools/finalize_rl_data.py" "$FRAMES" "$TRAIN" "$OUT/data/train.json" --max_drop ${MAX_DROP:-20}
"$PY" "$HERE/tools/finalize_rl_data.py" "$FRAMES" "$VAL" "$OUT/data/val.json" --max_drop ${MAX_DROP_VAL:-3}

export PYTHONUNBUFFERED=1 PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export DATA_MODE=video VLLM_USE_V1=1 WANDB_MODE=offline HYDRA_FULL_ERROR=1 RAY_DEDUP_LOGS=0
export BASE_IMAGE_DIR="$FRAMES" LOG_SAVE_DIR="$OUT/logs" WANDB_DIR="$OUT/wandb"
export KEEP_CKPTS=${KEEP_CKPTS:-2} VO3_DUMP_ROLLOUTS_DIR="$OUT/rollouts" DECORD_EOF_RETRY_MAX=40960 VO3_PREFETCH_FACTOR=1
unset VO3_SMOKE_FORCE_TOOL VO3_SMOKE_FORCE_BAD
if [ "$FAMILY" = qwen25 ]; then
  MAX_PROMPT=18432; MAX_BATCHED=34816; GPU_UTIL=0.85
  EXTRA=()
else
  # Qwen3-VL: timestamps are text tokens (+4,096 on prompt / context / batch limits); vLLM memory share 0.80 (0.85 ran out of memory on
  # wake-up on 141 GB GPUs); cascade attention off and the parameter guard on (see README)
  export VISION_PATCH_SIZE=16 VO3_DISABLE_CASCADE=1 VO3_PARAM_GUARD=${VO3_PARAM_GUARD:-1} VO3_MEMCHECK=0 VO3_MEMCHECK_DIR="$OUT/memcheck" VO3_THINK_PREFILL=${VO3_THINK_PREFILL:-0}
  MAX_PROMPT=22528; MAX_BATCHED=38912; GPU_UTIL=0.80
  EXTRA=(+data.patch_size=16 +actor_rollout_ref.actor.skip_overlong_prompt=True actor_rollout_ref.rollout.max_total_response_length=36864
         +actor_rollout_ref.rollout.crop_frames_sample_fps=2.0 +actor_rollout_ref.rollout.crop_min_tokens=512
         +actor_rollout_ref.rollout.crop_max_tokens_coarse=2048 +actor_rollout_ref.rollout.crop_max_tokens_medium=4096
         +actor_rollout_ref.rollout.crop_max_tokens_fine=6144 +actor_rollout_ref.rollout.crop_max_tokens_per_frame_cap=768)
fi

echo "== 2. GRPO (reward $VO3_REWARD)"
# start the Ray head first (vLLM V1 engine processes must find the same cluster), with this environment's python
"$PY" -m ray.scripts.scripts stop --force > /dev/null 2>&1 || true
"$PY" -m ray.scripts.scripts start --head --disable-usage-stats --include-dashboard=false > "$OUT/logs/ray_start.log" 2>&1
VBT=True; [ -f "$OUT/latest_checkpointed_iteration.txt" ] && VBT=False   # resuming: no validation before training
( cd "$CODE" && "$PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    hydra.run.dir="$OUT/logs/hydra_outputs" \
    trainer.val_before_train=$VBT \
    data.system_prompt="tool_crop" \
    data.train_files=["$OUT/data/train.json"] \
    data.val_files=["$OUT/data/val.json"] \
    data.train_batch_size=32 \
    data.max_prompt_length=$MAX_PROMPT \
    data.max_response_length=8192 \
    data.image_key=images \
    data.video_key=video \
    data.answer_key=solution \
    data.mask_blank=False \
    data.acc_reward_weight=1.0 \
    data.format_reward_weight=1.0 \
    data.decay_penalty_weight=0.05 \
    data.general_qa_reward_fn="general_qa_tool" \
    data.gpt_extract_answer=True \
    data.extract_answer_tags="strict" \
    data.return_raw_chat=True \
    data.gpt_threads=16 \
    data.tool_call="crop" \
    data.use_tgt_size=False \
    data.max_pixels=16384 \
    data.min_pixels=512 \
    reward_model.reward_manager=naive_multithreads_tool \
    actor_rollout_ref.actor.ignore_exceed=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED \
    actor_rollout_ref.rollout.name=vllm_multi_turn_tool_call \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UTIL \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_generation_round=6 \
    'actor_rollout_ref.rollout.limit_mm_per_prompt={'video': 12}' \
    actor_rollout_ref.rollout.val_max_generation_round=6 \
    'actor_rollout_ref.rollout.val_limit_mm_per_prompt={'video': 12}' \
    actor_rollout_ref.rollout.use_raw_image=True \
    actor_rollout_ref.rollout.multi_turn_prompt_type="v2" \
    actor_rollout_ref.rollout.vllm_infer_batch_size=32 \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.rollout.use_relative_coordinates=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='video_o3_multiturn' \
    trainer.experiment_name="$(basename "$OUT")" \
    trainer.val_generations_to_log_to_wandb=512 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-5} \
    trainer.default_local_dir="$OUT" \
    trainer.test_freq=${TEST_FREQ:-25} \
    trainer.total_epochs=1 \
    trainer.log_training_rollouts_freq=5 \
    trainer.train_generations_to_log_to_wandb=256 \
    trainer.use_3drope=True \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=1 \
    ${EXTRA[@]+"${EXTRA[@]}"} ) 2>&1 | tee -a "$OUT/train.log" || echo "the training process exited with an error"
# TRAINING_DONE.json is written after the final validation and checkpoint (an error after that point does not matter)
[ -f "$OUT/TRAINING_DONE.json" ] || { echo "training did not finish (no TRAINING_DONE.json); rerun the same command to resume from the last checkpoint"; exit 1; }
"$PY" -m ray.scripts.scripts stop --force > /dev/null 2>&1 || true

echo "== 3. merge the final checkpoint"
FINAL=$("$PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['final_ckpt_step'])" "$OUT/TRAINING_DONE.json")
ACT="$OUT/global_step_$FINAL/actor"
( cd "$CODE" && "$PY" tools/model_merger.py --local_dir "$ACT" ) > "$OUT/merge.log" 2>&1
"$PY" "$HERE/tools/finalize_merged.py" --family "$FAMILY" "$ACT/huggingface" "$MODEL_PATH" | tee -a "$OUT/merge.log"
echo "done: $ACT/huggingface"
