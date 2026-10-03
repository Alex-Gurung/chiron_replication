#!/bin/bash
# Once every chiron_ptest_* job (scripts/prompt_test.sh) has left the queue: rebuild the sheet reps and queue the 27B
# (thinking off) evals of each prompt variant's personality/appearance (sec0) and speech (sec3) sections.
# Usage: bash scripts/ptest_pipeline.sh
R=/home/toolkit/chiron_replication
Q=/home/toolkit/eaiexp/state/queue
while ls $Q/queued $Q/running | grep -q "^chiron_ptest_"; do sleep 60; done
for s in test val train; do $R/.venv/bin/python $R/chiron/build_reps.py --split $s --sheets-only > /dev/null; done
for v in low brief brieflow; do python3 $R/scripts/queue_jobs.py eval27w pt_$v 0 3 legacy_gptoss_${v}_x_sec0 legacy_gptoss_${v}_x_sec3 > /dev/null; done
echo "$(date -u +%H:%M) prompt variants: $(cat $R/outputs/legacy_gptoss_{low,brief,brieflow}_x/*.jsonl | wc -l) answers filtered; 27B evals queued"
