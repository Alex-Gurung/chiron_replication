#!/bin/bash
# Once a summary variant's jobs (chiron_sheet_<V>_*) have left the queue: rebuild the sheet reps and queue the 27B
# (thinking off) evals of sheet_<V>_flat alone and with the last 1,000 words.
# Usage: bash scripts/sum_pipeline.sh VARIANT
V=$1
R=/home/toolkit/chiron_replication
Q=/home/toolkit/eaiexp/state/queue
while ls $Q/queued $Q/running | grep -q "^chiron_sheet_${V}_[0-9]*_"; do sleep 60; done
for s in test val train; do $R/.venv/bin/python $R/chiron/build_reps.py --split $s --sheets-only > /dev/null; done
python3 $R/scripts/queue_jobs.py eval27w s_$V 0 3 sheet_${V}_flat "sheet_${V}_flat&book_last1000" > /dev/null
echo "$(date -u +%H:%M) $V: $(cat $R/outputs/sheets/$V/*.jsonl | wc -l) summaries; 27B evals queued"
