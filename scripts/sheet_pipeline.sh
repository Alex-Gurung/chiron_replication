#!/bin/bash
# Queue the 27B (thinking off) eval of each sheet variant as soon as its generation jobs have left the queue.
# Usage: bash scripts/sheet_pipeline.sh VARIANT...   (evaluates sheet_<V> and sheet_<V>+pre; prints one line per variant)
R=/home/toolkit/chiron_replication
Q=/home/toolkit/eaiexp/state/queue
left=("$@")
sleep 30
while [ ${#left[@]} -gt 0 ]; do
  ready=(); rest=()
  for v in "${left[@]}"; do
    if ls $Q/queued $Q/running | grep -qE "^chiron_sheet_${v}_[0-9]+_"; then rest+=("$v"); else ready+=("$v"); fi
  done
  if [ ${#ready[@]} -gt 0 ]; then
    for s in test val train; do $R/.venv/bin/python $R/chiron/build_reps.py --split $s --sheets-only > /dev/null; done
    for v in "${ready[@]}"; do
      n=$(cat $R/outputs/sheets/$v/*.jsonl | wc -l)
      python3 $R/scripts/queue_jobs.py eval27 $v 0 sheet_$v sheet_$v+pre > /dev/null
      echo "$(date -u +%H:%M) $v: $n sheets, 27B eval queued"
    done
  fi
  left=("${rest[@]}")
  if [ ${#left[@]} -gt 0 ]; then sleep 60; fi
done
