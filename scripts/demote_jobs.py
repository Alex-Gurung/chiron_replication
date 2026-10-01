"""Stop running chiron queue jobs and requeue them at a lower priority (their finished work is skipped on restart).

  python3 scripts/demote_jobs.py PRIORITY NAME...
Kills each job's process tree on its worker pod (eaiexp runners/kill_zombies.sh), requeues the same command under a
fresh name with the given priority, and moves the old record from failed/ to failed_archive/.
"""
import json
import subprocess
import sys
import time

from hang_watchdog import QUEUE, workers

sys.path.insert(0, "/home/toolkit/eaiexp")
import jobqueue as q  # noqa: E402

prio, names = int(sys.argv[1]), sys.argv[2:]
pods = workers()
for name in names:
    rec = json.load(open(QUEUE / "running" / f"{name}.json"))
    subprocess.run(["eai", "job", "exec", pods[rec["claimed_by"]], "--", "bash", "/home/toolkit/eaiexp/runners/kill_zombies.sh", name],
                   stdin=subprocess.DEVNULL, capture_output=True)
    for _ in range(60):
        if (QUEUE / "failed" / f"{name}.json").exists():
            (QUEUE / "failed" / f"{name}.json").rename(QUEUE / "failed_archive" / f"{name}.json")
            break
        time.sleep(5)
    short = name.rsplit("_", 1)[0].replace("chiron_", "", 1)
    q.add(q.conn(), f"chiron_{short}_{time.strftime('%m%d%H%M', time.gmtime())}", rec["cmd"], gpus=rec["gpus"], lane="chiron", priority=prio)
    print("demoted", name, flush=True)
