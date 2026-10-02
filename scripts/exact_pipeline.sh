#!/bin/bash
# Once every chiron_exact_* job (scripts/legacy_exact.sh) has left the queue: rebuild the sheet reps and queue the 27B
# (thinking off) evals of the exactly-replicated gpt-oss notes (legacy_gptoss_x) and their ~500-word summary
# (sheet_legsum_x_500_flat), alone, with the last 1,000 words and with Story Information, each with and without +pre.
# Usage: bash scripts/exact_pipeline.sh
R=/home/toolkit/chiron_replication
Q=/home/toolkit/eaiexp/state/queue
while ls $Q/queued $Q/running | grep -q "^chiron_exact_"; do sleep 60; done
for s in test val train; do $R/.venv/bin/python $R/chiron/build_reps.py --split $s --sheets-only > /dev/null; done
X=legacy_gptoss_x; S=sheet_legsum_x_500_flat
python3 $R/scripts/queue_jobs.py eval27w xnotes 1 3 $X $X+pre > /dev/null
python3 $R/scripts/queue_jobs.py eval27w xnotes_l1k 1 3 "$X&book_last1000" "$X&book_last1000+pre" > /dev/null
python3 $R/scripts/queue_jobs.py eval27w xnotes_si 1 3 "$X&si-ncp-ch2" "$X&si-ncp-ch2+pre" > /dev/null
python3 $R/scripts/queue_jobs.py eval27w xsum 0 2 $S $S+pre "$S&book_last1000" "$S&book_last1000+pre" "$S&si-ncp-ch2" "$S&si-ncp-ch2+pre" > /dev/null
echo "$(date -u +%H:%M) exact notes: $(cat $R/outputs/legacy_gptoss_x/*.jsonl | wc -l) answers, $(cat $R/outputs/sheets/legsum_x_500/*.jsonl | wc -l) summaries; 27B evals queued"
