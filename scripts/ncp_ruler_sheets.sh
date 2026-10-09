#!/bin/bash
# One shard of chiron/ncp_ruler_sheets.py (the NCP writer ruler with only the character sheets swapped) on this job's GPU,
# set up like eaiexp/runners/ncp_score_q4_v2_20260908.sh: env3 python, the frozen ncp_eval code first on PYTHONPATH,
# STAGE2_STYLE=soft, GPU utilisation from gpu_fit_util.py (capped at 0.76).
# Usage: bash scripts/ncp_ruler_sheets.sh K N [extra ncp_ruler_sheets.py args]
set -euo pipefail
R=/home/toolkit/chiron_replication
export STAGE2_STYLE=soft
export PYTHONPATH=/home/toolkit/ncp_q4_v2_20260908/code:/home/toolkit/diversity/diversity
export HF_HOME=/home/toolkit/.cache/huggingface HF_HUB_OFFLINE=1
S=/tmp/chiron_${JOB_NAME:-local}
export VLLM_CACHE_ROOT=$S/vllm TRITON_CACHE_DIR=$S/triton
trap 'rm -rf $S' EXIT
UTIL=$(/home/toolkit/eaiexp/env3/bin/python /home/toolkit/eaiexp/runners/gpu_fit_util.py)
/home/toolkit/eaiexp/env3/bin/python -u $R/chiron/ncp_ruler_sheets.py --shard "$1" --nshards "$2" --gpu-memory-utilization "$UTIL" "${@:3}"
