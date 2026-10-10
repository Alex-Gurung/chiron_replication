#!/bin/bash
# Launch one eaiexp worker pod that serves ONLY the "chiron" queue lane.
# Usage: bash scripts/runner_chiron_worker.sh <worker-number> [gpus] [lanes]   (lanes default: chiron)
# Jobs are added to /home/toolkit/eaiexp/state/queue with lane=chiron (scripts/queue_jobs.py).
set -euo pipefail
N="${1:?usage: runner_chiron_worker.sh <worker-number> [gpus]}"
G="${2:-8}"
L="${3:-chiron}"
STAMP="$(date -u +%m%d%H%M%S)"
eai job new --gpu "$G" --cpu $(( G * 4 )) --mem $(( G * 32 )) \
  --data snow.home.alex_gurung:/home/toolkit:rw \
  --data snow.research.rlar.transformers_cache:/transformers_cache:ro \
  --preemptable --restartable --name "chiron_w${N}_${STAMP}" --tag proj=chiron \
  -e WORKER_ID="chiron_w${N}_${STAMP}" -e LANES="$L" -e CPU_SLOTS=0 -e IDLE_EXIT_SEC=3600 \
  -i pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel \
  -- /bin/bash -lc '/home/toolkit/eaiexp/worker.sh'
