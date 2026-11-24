#!/bin/bash
# Multi-turn SFT with the SFT code of Video-o3 (example).
#
# One-time setup: patched Video-o3 checkout (see README.md), then from Video-o3/SFT:  pip install -e .  (or rely on PYTHONPATH below).
# Data (see README.md):
#   python tools/build_sft_data.py data/evice_multiturn_qwen25.json --evice /path/to/EviCe/EviCe.json --media_dir /path/to/EviCe \
#       --video_o3 /path/to/Video-o3 --family qwen25 --check_duration
#   cp sft/dataset_info.json data/      # the yaml's dataset_dir
#
# Usage (from Video-o3/SFT):
#   FAMILY=qwen25 CONFIG=/path/to/TGR-Reliance/training/multiturn/sft/qwen2_5vl_7b_multiturn.yaml bash run_sft.sh
#   FAMILY=qwen3  CONFIG=/path/to/TGR-Reliance/training/multiturn/sft/qwen3vl_8b_multiturn.yaml   bash run_sft.sh
# Multi-node: also set NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT (standard LLaMA-Factory variables).
# Afterwards build the starting model for RL (SFT weights + base model files):
#   python tools/make_view.py --family qwen25 saves/qwen2_5vl_7b_multiturn_sft Qwen/Qwen2.5-VL-7B-Instruct saves/qwen2_5vl_7b_multiturn_sft_view
set -euo pipefail
FAMILY=${FAMILY:?qwen25 or qwen3}; CONFIG=${CONFIG:?path to the yaml}
[ -d src/llamafactory ] || { echo "run from the SFT folder of the patched Video-o3 checkout"; exit 1; }
export PYTHONPATH=$(pwd)/src${PYTHONPATH:+:$PYTHONPATH} DISABLE_VERSION_CHECK=1 DECORD_EOF_RETRY_MAX=40960
# Qwen3-VL: sample and resize videos with the rules of the RL engine (4 fps grid, absolute timestamps; see the Qwen3 patch)
if [ "$FAMILY" = qwen3 ]; then export VO3_Q3_GRID_FRAMES=1 VISION_PATCH_SIZE=16; fi
FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG"
