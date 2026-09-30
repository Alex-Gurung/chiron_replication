#!/bin/bash
# Chain the steps after gpt-oss generation finishes (run in the background from the login node):
#   chapter notes done -> rebuild reps -> queue their evals;
#   plot summaries done -> requeue the plot jobs once (resumable: fills failed targets) -> export -> queue their evals.
# Usage: bash scripts/after_gen.sh
set -uo pipefail
Q=/home/toolkit/eaiexp/state/queue
R=/home/toolkit/chiron_replication
left() { ls $Q/queued $Q/running | grep -cE "$1"; }
wait_for() { while [ "$(left "$1")" -gt 0 ]; do sleep 60; done; }

chapnotes() {
  wait_for '^chiron_chapnotes_[0-9]'
  for s in test val train; do $R/.venv/bin/python $R/chiron/build_reps.py --split $s | grep -E "^chapnotes"; done
  echo "chapnotes evals queued: $(python3 $R/scripts/queue_jobs.py chapnotes_eval | grep -c queued) $(date -u +%H:%M)"
}

plots() {
  wait_for '^chiron_plot_(global|hier)_'
  python3 $R/scripts/queue_jobs.py plot > /dev/null
  sleep 120
  wait_for '^chiron_plot_(global|hier)_'
  python3 $R/chiron/gen_plot.py export
  echo "plot evals queued: $(python3 $R/scripts/queue_jobs.py plot_eval | grep -c queued) $(date -u +%H:%M)"
}

chapnotes &
plots &
wait
