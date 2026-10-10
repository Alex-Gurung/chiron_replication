"""Stop running chiron queue jobs cleanly and (optionally) requeue them.

  python3 scripts/stop_jobs.py REGEX [--requeue PRIORITY]
For every running job whose name matches REGEX: SIGKILL its process tree on its worker pod (eaiexp
runners/kill_zombies.sh), then SIGKILL every VLLM::EngineCore whose parent is init on that pod: a killed vLLM job leaves
its engine holding the GPU's memory, and every job started on that GPU afterwards dies in engine init ("Free memory on
device ... is less than desired"). Live engines keep their job as parent and are not touched. The failed/ records are
archived, and with --requeue the same commands are queued again under fresh names BEFORE the kill, so freed GPUs go to
the queue in priority order. Prints GPU memory per pod at the end.
"""
import json
import re
import subprocess
import sys
import time

sys.path.insert(0, "/home/toolkit/chiron_replication/scripts")
sys.path.insert(0, "/home/toolkit/eaiexp")
import jobqueue as q  # noqa: E402
from hang_watchdog import QUEUE, workers  # noqa: E402

SWEEP = ('for p in $(ps -eo pid,ppid,cmd | grep "VLLM::EngineCore" | grep -v grep | awk "\\$2==1 {print \\$1}"); do kill -9 $p 2>/dev/null; done; '
         'sleep 6; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr "\\n" ";"')


def main():
    rx = re.compile(sys.argv[1])
    prio = int(sys.argv[sys.argv.index("--requeue") + 1]) if "--requeue" in sys.argv else None
    run = [r for r in (json.load(open(f)) for f in QUEUE.glob("running/chiron_*.json")) if rx.search(r["name"])]
    pods = workers()
    stamp = time.strftime("%m%d%H%M%S", time.gmtime())
    if prio is not None:
        for r in run:
            q.add(q.conn(), r["name"].rsplit("_", 1)[0] + "_" + stamp, r["cmd"], gpus=r["gpus"], lane=r["lane"], priority=prio)
    ex = lambda pod, *cmd: subprocess.run(["eai", "job", "exec", pod, "--", *cmd], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    for r in run:
        ex(pods[r["claimed_by"]], "bash", "/home/toolkit/eaiexp/runners/kill_zombies.sh", r["name"])
    time.sleep(10)
    for w in sorted({r["claimed_by"] for r in run}):
        print(w, ex(pods[w], "bash", "-c", SWEEP).stdout.strip())
    for r in run:
        p = QUEUE / "failed" / f"{r['name']}.json"
        if p.exists():
            p.rename(QUEUE / "failed_archive" / p.name)
    print("stopped", len(run), "jobs" + (f", requeued at priority {prio}" if prio is not None else ""))


if __name__ == "__main__":
    main()
