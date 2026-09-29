#!/bin/bash
# Orchestrate the end of generation, then the final evaluation.
#  1. wait for the first-wave charmem jobs (old harness) to leave the queue;
#  2. requeue every book without a clean completed_books marker, one job per book (current harness:
#     second synthesis round with validator-driven repair);
#  3. wait until no summary, charmem or pronoun job is queued or running;
#  4. build representations for all splits and queue the final evaluation.
# Usage: bash scripts/final_when_ready.sh
set -uo pipefail
Q=/home/toolkit/eaiexp/state/queue
R=/home/toolkit/ncp_charmem_gptoss120b_reviewed_20260908
cd /home/toolkit/chiron_replication
busy() { ls $Q/queued $Q/running | grep -q "$1"; }

while busy "chiron_charmem_[0-9]_\|chiron_charmem_fix1"; do sleep 60; done
python3 - <<'PY'
import json, sys, time
sys.path.insert(0, "/home/toolkit/eaiexp"); import jobqueue as q
from pathlib import Path
R = Path("/home/toolkit/ncp_charmem_gptoss120b_reviewed_20260908")
REPO = "/home/toolkit/chiron_replication"
books = json.load(open(f"{REPO}/data/principals.json"))
todo = [b for b in books if not (R / "completed_books" / f"{b}.json").exists()
        or json.load(open(R / "completed_books" / f"{b}.json")).get("missing_boundaries", [None])]
stamp = time.strftime("%m%d%H%M", time.gmtime())
for b in todo:
    cmd = ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "131072", "--",
           "python3", "-u", f"{REPO}/chiron/charmem_finish.py", "--books", b, "--width", "24"]
    q.add(q.conn(), f"chiron_charmem_fix2_{b}_{stamp}", cmd, gpus=2, lane="chiron", priority=0)
print("charmem fix2 queued for", todo, flush=True)
PY

while busy "chiron_sum_\|chiron_charmem\|chiron_pronouns"; do sleep 60; done
ls $Q/failed | grep chiron && echo "WARNING: failed chiron jobs above"
for s in test val train; do (cd chiron && ../.venv/bin/python build_reps.py --split $s 2>&1 | grep -v -i warn); done
python3 scripts/queue_jobs.py final test val train | tail -3
