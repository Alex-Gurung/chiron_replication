#!/bin/bash
# Once every chiron_exactL_* job (gen_legacy_exact.py --source llama) has left the queue: rebuild the sheet reps and queue
# the 27B (thinking off) eval of the Llama notes re-rated by the gpt-oss judge (legacy_full_xf), train in 9 shards.
# Usage: bash scripts/exactL_pipeline.sh
R=/home/toolkit/chiron_replication
Q=/home/toolkit/eaiexp/state/queue
while ls $Q/queued $Q/running | grep -q "^chiron_exactL_"; do sleep 60; done
for s in test val train; do $R/.venv/bin/python $R/chiron/build_reps.py --split $s --sheets-only > /dev/null; done
python3 $R/scripts/queue_jobs.py eval27w xf 1 9 legacy_full_xf legacy_full_xf+pre > /dev/null
echo "$(date -u +%H:%M) Llama notes re-rated: $(cat $R/outputs/legacy_llama_xf/*.jsonl | wc -l) answers; 27B eval queued"
