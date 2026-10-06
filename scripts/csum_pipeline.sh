#!/bin/bash
# Once every chiron_csum_* job (chiron/gen_csum.py, the CHIRON paper's Character-Summary baseline) has left the queue:
# rebuild the sheet reps and queue its evals: 27B thinking off (alone and with the last 1,000 words), 9B base and 4B.
# Usage: bash scripts/csum_pipeline.sh
R=/home/toolkit/chiron_replication
Q=/home/toolkit/eaiexp/state/queue
while ls $Q/queued $Q/running | grep -q "^chiron_csum_"; do sleep 60; done
for s in test val train; do $R/.venv/bin/python $R/chiron/build_reps.py --split $s --sheets-only > /dev/null; done
python3 $R/scripts/queue_jobs.py eval27w csum 0 3 sheet_csum_flat sheet_csum_f_flat "sheet_csum_flat&book_last1000" "sheet_csum_f_flat&book_last1000" > /dev/null
python3 $R/scripts/queue_jobs.py evalsmall csum 0 sheet_csum_flat sheet_csum_f_flat > /dev/null
echo "$(date -u +%H:%M) csum: $(cat $R/outputs/sheets/csum_f/*.jsonl | wc -l) summaries; evals queued"
