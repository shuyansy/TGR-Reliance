#!/bin/bash
# SFT on EviCe with LLaMA-Factory (example).
#
# One-time setup:
#   git clone https://github.com/hiyouga/LLaMA-Factory.git && cd LLaMA-Factory
#   git checkout 3140c242f0             # the version we used (0.9.4.dev0)
#   git apply /path/to/TGR-Reliance/training/sft/llamafactory.patch
#   pip install -e ".[torch,metrics,deepspeed]" transformers==4.57.1 decord
#   cp /path/to/EviCe/EviCe.json data/  # EviCe annotations (see the EviCe dataset card)
#   merge training/sft/dataset_info.json into data/dataset_info.json
# and set media_dir in the yaml to the downloaded EviCe folder (it holds videos/).
#
# The patch adds LF_VIDEO_SAMPLING=official_oasis: frames are sampled by the official Qwen2.5-VL / Qwen3-VL sampler
# (1 fps, at most 128 frames, official resolution), so that training inputs match our evaluation exactly.
#
# Usage (from the LLaMA-Factory folder):
#   FAMILY=qwen25 CONFIG=/path/to/qwen2_5vl_7b_evice.yaml bash run_sft.sh
#   FAMILY=qwen3  CONFIG=/path/to/qwen3vl_8b_evice.yaml   bash run_sft.sh
# Multi-node: also set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT (standard LLaMA-Factory variables).
set -euo pipefail
FAMILY=${FAMILY:?qwen25 or qwen3}; CONFIG=${CONFIG:?path to the yaml}
export LF_VIDEO_SAMPLING=official_oasis LF_OASIS_FAMILY=$FAMILY
export LF_OASIS_DROP_LOG=${LF_OASIS_DROP_LOG:-dropped_videos.txt}   # videos the official sampler cannot read are dropped and listed here
FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG"
