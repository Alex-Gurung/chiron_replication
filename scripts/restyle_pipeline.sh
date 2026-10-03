#!/bin/bash
# Once all four restyle jobs (chiron_sheet_restyle_*) have left the queue: rebuild the sheet reps once and queue the 27B
# (thinking off) evals of each rewritten summary, alone and with the last 1,000 words.
# Usage: bash scripts/restyle_pipeline.sh
R=/home/toolkit/chiron_replication
Q=/home/toolkit/eaiexp/state/queue
while ls $Q/queued $Q/running | grep -q "^chiron_sheet_restyle_"; do sleep 60; done
for s in test val train; do $R/.venv/bin/python $R/chiron/build_reps.py --split $s --sheets-only > /dev/null; done
for v in restyle_leg_prose restyle_leg_bullets restyle_lsx_prose restyle_lsx_bullets; do
  python3 $R/scripts/queue_jobs.py eval27w s_$v 0 3 sheet_${v}_flat "sheet_${v}_flat&book_last1000" > /dev/null
done
echo "$(date -u +%H:%M) restyled summaries: $(cat $R/outputs/sheets/restyle_*/*.jsonl | wc -l); 27B evals queued"
