#!/bin/bash
# After the per-book charmem reruns finish: rebuild representations and requeue the charmem conditions
# (already-scored items are skipped by eval_mcp). Usage: bash scripts/charmem_evals_when_ready.sh
set -uo pipefail
Q=/home/toolkit/eaiexp/state/queue
cd /home/toolkit/chiron_replication
busy() { ls $Q/queued $Q/running | grep "$1" > /dev/null; }
while busy "chiron_charmem"; do sleep 60; done
ls /home/toolkit/ncp_charmem_gptoss120b_reviewed_20260908/completed_books | wc -l
for s in test val train; do (cd chiron && ../.venv/bin/python build_reps.py --split $s 2>&1 | grep "^charmem"); done
python3 - <<'PY'
import sys, time
sys.path.insert(0, "/home/toolkit/eaiexp"); import jobqueue as q
REPO = "/home/toolkit/chiron_replication"; stamp = time.strftime("%m%d%H%M", time.gmtime())
serve = lambda model, n: ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", model, "--max-model-len", str(n), "--"]
ev = lambda split, stem, c: ["python3", "-u", f"{REPO}/chiron/eval_mcp.py", "--split", split, "--items", stem,
                             "--conditions", c, "--workers", "128"]
jobs = []
for split in ("test", "val", "train"):
    for c in ("charmem", "swap:charmem"):
        jobs.append((f"{split}_{c}", serve("qwen4b", 65536) + ev(split, f"items_{split}", c)))
        jobs.append((f"{split}_two_{c}", serve("qwen4b", 65536) + ev(split, f"items_{split}_two", c)))
    jobs.append((f"{split}_pron_charmem", serve("qwen4b", 65536) + ev(split, f"items_{split}_pron", "charmem")))
    jobs.append((f"{split}_mistral_charmem", ["env", "CHIRON_ANSWER_PREFIX=[CHAR "] + serve("mistral", 32768) + ev(split, f"items_{split}", "charmem")))
for name, cmd in jobs:
    q.add(q.conn(), f"chiron_cm_{name.replace(':', '_')}_{stamp}", cmd, gpus=1, lane="chiron", priority=0)
print("queued", len(jobs), "charmem eval jobs", flush=True)
PY
