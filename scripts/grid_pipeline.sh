#!/bin/bash
# The same-model grid after generation: when a generator's jobs (chiron_gen_<gen>_*) have left the queue, export its plot
# summaries, rebuild the sheet reps and the NCP grid inputs, and queue its evaluations (scripts/queue_grid.py) on the lane
# in CHIRON_LANE: character identification on the 27B, 9B base and 4B, then the NCP ruler's second wave (canonical judge).
# Usage: CHIRON_LANE=chiron_l70 bash scripts/grid_pipeline.sh
R=/home/toolkit/chiron_replication
Q=/home/toolkit/eaiexp/state/queue
rebuild() {
  CHIRON_GEN=gptoss python3 $R/chiron/gen_plot.py export > /dev/null
  for s in test val train; do $R/.venv/bin/python $R/chiron/build_reps.py --split $s --sheets-only > /dev/null; done
  python3 $R/chiron/ncp_ruler_grid.py --prepare | tail -1
}
while ls $Q/queued $Q/running | grep -q "^chiron_gen_l70_"; do sleep 60; done
rebuild
python3 $R/scripts/queue_grid.py charid l70 --judges 27 small > /dev/null
echo "$(date -u +%H:%M) l70 generation done: character-identification evals queued"
while ls $Q/queued $Q/running | grep -q "^chiron_gen_q4b_"; do sleep 60; done
rebuild
python3 $R/scripts/queue_grid.py charid q4b --judges 27 small > /dev/null
python3 $R/scripts/queue_grid.py charid gptoss --judges 27 small > /dev/null
python3 $R/scripts/queue_grid.py ncp w2 2 Qwen/Qwen3-4B-Instruct-2507 --lean q4b l70 > /dev/null
echo "$(date -u +%H:%M) q4b generation done: q4b + gpt-oss evals and NCP wave 2 queued"
