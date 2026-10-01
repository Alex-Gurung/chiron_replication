"""Kill and requeue chiron queue jobs whose vLLM engine has hung.

  python3 scripts/hang_watchdog.py [--stale-min 10] [--once]
Hang signature (seen three times with Qwen3.8-27B): the server is READY, its last "Avg generation throughput" line
still shows requests running or waiting, and no new line has been logged for --stale-min minutes. The job neither
fails nor exits, so the queue never notices. For each such job: SIGKILL its process tree on its worker pod
(eaiexp runners/kill_zombies.sh), requeue the same command (read before the kill) under a fresh name (finished work is
skipped on restart) and move the old record from failed/ to failed_archive/ once it lands there. Prints one line per action.
"""
import argparse
import datetime
import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/toolkit/eaiexp")
import jobqueue as q  # noqa: E402

QUEUE = Path("/home/toolkit/eaiexp/state/queue")
RUNS = Path("/home/toolkit/eaiexp/runs")
REPO = Path(__file__).resolve().parent.parent
LINE = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d) .*Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs")


def hung(name, stale):
    log = RUNS / f"{name}.tmp" / "vllm.log"
    if not log.exists() or "READY" not in (RUNS / f"{name}.tmp" / "worker.log").read_text(errors="ignore"):
        return False
    last = None
    for m in LINE.finditer(log.read_text(errors="ignore")[-200000:]):
        last = m
    if not last or int(last.group(3)) + int(last.group(4)) == 0:
        return False
    t = datetime.datetime.strptime(f"{datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None).year}-{last.group(1)}", "%Y-%m-%d %H:%M:%S")
    return (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - t).total_seconds() > stale * 60


def workers():
    out = subprocess.run(["eai", "job", "ls", "--state", "alive", "--fields", "id,name"], capture_output=True, text=True).stdout
    return {n: i for i, n in (l.split()[:2] for l in out.splitlines() if "chiron_w" in l)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale-min", type=float, default=10)
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    while True:
        for f in sorted(QUEUE.glob("running/chiron_*.json")):
            name = f.stem
            if not hung(name, args.stale_min):
                continue
            rec = json.load(open(f))                              # keep the command before the record moves
            pod = workers().get(rec["claimed_by"])
            subprocess.run(["eai", "job", "exec", pod, "--", "bash", "/home/toolkit/eaiexp/runners/kill_zombies.sh", name],
                           stdin=subprocess.DEVNULL, capture_output=True)
            for _ in range(60):
                if (QUEUE / "failed" / f"{name}.json").exists():
                    break
                time.sleep(5)
            short = name.rsplit("_", 1)[0].replace("chiron_", "", 1)
            q.add(q.conn(), f"chiron_{short}_{time.strftime('%m%d%H%M', time.gmtime())}", rec["cmd"], gpus=rec["gpus"], lane="chiron",
                  priority=rec["priority"])
            if (QUEUE / "failed" / f"{name}.json").exists():
                (QUEUE / "failed" / f"{name}.json").rename(QUEUE / "failed_archive" / f"{name}.json")
            print(f"{datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None):%H:%M} hung engine: killed and requeued {name}", flush=True)
        if args.once:
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
