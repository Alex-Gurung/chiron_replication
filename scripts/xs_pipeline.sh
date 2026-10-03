#!/bin/bash
# Once the legsum_xs_500 summary jobs have left the queue: rebuild the sheet reps and queue the 27B (thinking off) evals of
# the ~500-word summary of the gpt-oss notes with reworded speech answers, alone and with the last 1,000 words.
# Usage: bash scripts/xs_pipeline.sh
R=/home/toolkit/chiron_replication
Q=/home/toolkit/eaiexp/state/queue
while ls $Q/queued $Q/running | grep -q "^chiron_sheet_legsum_xs_500_"; do sleep 60; done
for s in test val train; do $R/.venv/bin/python $R/chiron/build_reps.py --split $s --sheets-only > /dev/null; done
python3 $R/scripts/queue_jobs.py eval27w xssum 0 3 sheet_legsum_xs_500_flat "sheet_legsum_xs_500_flat&book_last1000" > /dev/null
echo "$(date -u +%H:%M) legsum_xs_500: $(cat $R/outputs/sheets/legsum_xs_500/*.jsonl | wc -l) summaries; 27B evals queued"
