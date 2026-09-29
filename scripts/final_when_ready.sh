#!/bin/bash
# Wait until no summary, charmem or pronoun job is queued or running, then build representations for
# all splits and queue the final evaluation. Usage: bash scripts/final_when_ready.sh
set -uo pipefail
Q=/home/toolkit/eaiexp/state/queue
cd /home/toolkit/chiron_replication
while ls $Q/queued $Q/running | grep -q "chiron_sum_\|chiron_charmem\|chiron_pronouns"; do sleep 60; done
ls $Q/failed | grep chiron && echo "WARNING: failed chiron jobs above"
for s in test val train; do (cd chiron && ../.venv/bin/python build_reps.py --split $s 2>&1 | grep -v -i warn); done
python3 scripts/queue_jobs.py final test val train | tail -3
