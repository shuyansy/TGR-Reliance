#!/bin/bash
# GRPO with verl from an SFT checkpoint (example; the hyperparameters are those of the paper).
#
# One-time setup:
#   git clone https://github.com/volcengine/verl.git && cd verl && git checkout f400fb7654b2
#   git apply /path/to/TGR-Reliance/training/grpo/verl.patch
#   pip install -e . vllm==0.11.0 transformers==4.57.1 qwen-vl-utils decord
# The patch makes rollouts and actor log-probs see the same video inputs as SFT and evaluation:
#   Qwen2.5-VL: real sampling fps and integer second_per_grid_t (TGA_Q25_FPS_TRUNC=1);
#   Qwen3-VL:   no resizing of the official frames (TGA_Q3_NO_RESIZE=1) and vLLM-compatible video placeholders.
#
# Usage:
#   FAMILY=qwen25 INIT_MODEL=/path/to/sft_checkpoint DATA=data/grpo OUT=outputs/grpo_qwen25 bash run_grpo.sh
#   FAMILY=qwen3  INIT_MODEL=/path/to/sft_checkpoint DATA=data/grpo OUT=outputs/grpo_qwen3  bash run_grpo.sh
#   REWARD=compute_score_iou (default, grounding only) | compute_score_ans_iou | compute_score_ans
# We trained on 2 nodes x 8 GPUs (NNODES=2 after starting a ray cluster), global batch 16, 8 rollouts per question,
# 323 steps (1 epoch over a 5,168-question pool); the batch size does not depend on the number of GPUs.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
FAMILY=${FAMILY:?qwen25 or qwen3}; INIT_MODEL=${INIT_MODEL:?SFT checkpoint}; DATA=${DATA:?folder with train.parquet / val.parquet}
OUT=${OUT:?output folder}; REWARD=${REWARD:-compute_score_iou}; NNODES=${NNODES:-1}
if [ "$FAMILY" = qwen25 ]; then PATCH=14; MAX_PROMPT=51200; PPO_MAX_TOK=25856; MAX_BATCHED_TOK=51712; SP=2
else PATCH=16; MAX_PROMPT=24576; PPO_MAX_TOK=32768; MAX_BATCHED_TOK=32768; SP=1; fi
MAX_RESP=256
export FORCE_QWENVL_VIDEO_READER=decord TGA_Q25_FPS_TRUNC=1 TGA_Q3_NO_RESIZE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 HYDRA_FULL_ERROR=1
mkdir -p "$OUT"
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo algorithm.use_kl_in_reward=False \
  data.train_files="$DATA/train.parquet" data.val_files="$DATA/val.parquet" \
  data.train_batch_size=16 data.max_prompt_length=$MAX_PROMPT data.max_response_length=$MAX_RESP \
  data.image_key=images data.video_key=videos data.image_patch_size=$PATCH \
  data.filter_overlong_prompts=False data.truncation=error data.return_raw_chat=True \
  reward.custom_reward_function.path="$HERE/reward.py" reward.custom_reward_function.name=$REWARD \
  actor_rollout_ref.model.path="$INIT_MODEL" \
  actor_rollout_ref.model.use_remove_padding=True actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SP actor_rollout_ref.ref.ulysses_sequence_parallel_size=$SP \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.use_dynamic_bsz=True actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOK \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT + MAX_RESP)) \
  actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_BATCHED_TOK actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$PPO_MAX_TOK \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$PPO_MAX_TOK \
  trainer.project_name=tgr_grpo trainer.experiment_name="$(basename "$OUT")" \
  trainer.n_gpus_per_node=8 trainer.nnodes=$NNODES trainer.default_local_dir="$OUT" \
  trainer.save_freq=81 trainer.test_freq=81 trainer.total_epochs=1 \
  trainer.val_before_train=True trainer.balance_batch=True trainer.resume_mode=auto \
  trainer.logger='["console","tensorboard"]' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.FORCE_QWENVL_VIDEO_READER="decord"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.TGA_Q25_FPS_TRUNC="1"' \
  '+ray_kwargs.ray_init.runtime_env.env_vars.TGA_Q3_NO_RESIZE="1"' 2>&1 | tee "$OUT/train.log"
