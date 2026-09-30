#!/bin/bash
# After the with-headings generation (queue_jobs.py headings_gen) finishes: rebuild reps, export plot summaries,
# queue their evals. Run in the background from the login node: bash scripts/after_headings.sh
set -uo pipefail
Q=/home/toolkit/eaiexp/state/queue
R=/home/toolkit/chiron_replication
while [ "$(ls $Q/queued $Q/running | grep -cE '^chiron_(narrator|h)_')" -gt 0 ]; do sleep 60; done
for s in test val train; do $R/.venv/bin/python $R/chiron/build_reps.py --split $s | grep -E "^(summary_h|chapnotes_h|chiron_h)"; done
python3 $R/chiron/gen_plot.py export
echo "headings evals queued: $(python3 $R/scripts/queue_jobs.py headings_eval | grep -c queued) $(date -u +%H:%M)"
