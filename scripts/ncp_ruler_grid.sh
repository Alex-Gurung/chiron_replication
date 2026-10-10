#!/bin/bash
# One shard of chiron/ncp_ruler_grid.py (the NCP writer ruler with the plot and character slots swapped) on this job's GPU,
# set up like eaiexp/runners/ncp_score_q4_v2_20260908.sh: env3 python, the frozen ncp_eval code first on PYTHONPATH,
# STAGE2_STYLE=soft, GPU utilisation from gpu_fit_util.py (0.76 cap for the canonical Qwen3-4B, 0.90 for the 27B judge).
# Usage: bash scripts/ncp_ruler_grid.sh K N TAG EVERY MODEL "COND" ...
set -euo pipefail
R=/home/toolkit/chiron_replication
K=$1; N=$2; TAG=$3; EVERY=$4; MODEL=$5; shift 5
export STAGE2_STYLE=soft
export PYTHONPATH=/home/toolkit/ncp_q4_v2_20260908/code:/home/toolkit/diversity/diversity
export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1
S=/tmp/chiron_${JOB_NAME:-local}
export VLLM_CACHE_ROOT=$S/vllm TRITON_CACHE_DIR=$S/triton
trap 'rm -rf $S' EXIT
MAXU=0.76; [ "$MODEL" = "Qwen/Qwen3-4B-Instruct-2507" ] || MAXU=0.90
UTIL=$(/home/toolkit/eaiexp/env3/bin/python /home/toolkit/eaiexp/runners/gpu_fit_util.py --max $MAXU)
/home/toolkit/eaiexp/env3/bin/python -u $R/chiron/ncp_ruler_grid.py --shard "$K" --nshards "$N" --tag "$TAG" --every "$EVERY" --model "$MODEL" \
  --gpu-memory-utilization "$UTIL" --conds "$@"
