#!/bin/bash
# After oracle generation: export per-passage oracle reps and queue their evals on every model.
# Usage: bash scripts/oracle_evals_when_ready.sh
set -uo pipefail
Q=/home/toolkit/eaiexp/state/queue
cd /home/toolkit/chiron_replication
busy() { ls $Q/queued $Q/running | grep "$1" > /dev/null; }
while busy "chiron_oracle_"; do sleep 60; done
ls $Q/failed | grep chiron_oracle_ && echo "WARNING: failed oracle jobs"
(cd chiron && python3 oracle_reps.py export)
python3 - <<'PY'
import sys, time, json
sys.path.insert(0, "/home/toolkit/eaiexp"); import jobqueue as q
REPO = "/home/toolkit/chiron_replication"; stamp = time.strftime("%m%d%H%M", time.gmtime()); n = 0
srv = lambda model, ml: ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", model, "--max-model-len", str(ml), "--"]
for split in ("test", "val", "train"):
    for c in ("oracle_passage", "oracle_prior", "swapname:oracle_prior"):
        t = c.replace(":", "_")
        ev = ["python3", "-u", f"{REPO}/chiron/eval_mcp.py", "--split", split, "--items", f"items_{split}", "--conditions", c, "--workers", "128", "--rotations"]
        for name, env, model in (("q27", ["CHIRON_THINKING=0"], "qwen27"), ("q9b", ["CHIRON_BASE=1"], "qwen9base"), ("q4", [], "qwen4b")):
            n += q.add(q.conn(), f"chiron_or_{name}_{split}_{t}_{stamp}", ["env", *env] + srv(model, 65536) + ev, gpus=1, lane="chiron", priority=0)
        n += q.add(q.conn(), f"chiron_or_q27think_{split}_{t}_{stamp}", ["env", "CHIRON_THINKING=1"] + srv("qwen27", 65536) +
                   ["python3", "-u", f"{REPO}/chiron/eval_reason.py", "--split", split, "--items", f"items_{split}", "--conditions", c, "--workers", "64"],
                   gpus=1, lane="chiron", priority=0)
print("queued oracle evals", n, flush=True)
PY
