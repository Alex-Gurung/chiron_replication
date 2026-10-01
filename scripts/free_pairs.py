"""Free GPU pairs for starving 2-GPU jobs: while every queued 1-GPU job is held aside, stop chosen running 1-GPU jobs
(two per worker), let the waiting 2-GPU jobs claim the freed pairs, then requeue the stopped jobs and release the held
ones. Finished items are kept, so a stopped eval loses only its in-flight requests.

  python3 scripts/free_pairs.py NPAIRS
Takes whole 2-GPU workers first, then pairs on 8-GPU workers, newest jobs first.
"""
import collections
import json
import subprocess
import sys
import time

from hang_watchdog import QUEUE, workers

sys.path.insert(0, "/home/toolkit/eaiexp")
import jobqueue as q  # noqa: E402

need = int(sys.argv[1])
held = []
for f in QUEUE.glob("queued/chiron_*.json"):
    if json.load(open(f))["gpus"] == 1:
        f.rename(QUEUE / "held" / f.name)
        held.append(f.name)
run = [json.load(open(f)) for f in QUEUE.glob("running/chiron_*.json")]
per = collections.defaultdict(list)
for r in run:
    if r["gpus"] == 1 and "think" in r["name"]:
        per[r["claimed_by"]].append(r)
small = {w for w in per if len([r for r in run if r["claimed_by"] == w]) == len(per[w]) == 2}   # whole 2-GPU workers
order = sorted(per, key=lambda w: (w not in small, -max(r["claimed_at"] for r in per[w])))
victims = []
for w in order:
    if len(victims) >= 2 * need:
        break
    jobs = sorted(per[w], key=lambda r: -r["claimed_at"])
    victims += jobs[:2] if len(jobs) >= 2 else []
pods = workers()
for r in victims:
    subprocess.run(["eai", "job", "exec", pods[r["claimed_by"]], "--", "bash", "/home/toolkit/eaiexp/runners/kill_zombies.sh", r["name"]],
                   stdin=subprocess.DEVNULL, capture_output=True)
for r in victims:
    for _ in range(60):
        p = QUEUE / "failed" / f"{r['name']}.json"
        if p.exists():
            p.rename(QUEUE / "failed_archive" / p.name)
            break
        time.sleep(5)
print("stopped", len(victims), "1-GPU jobs on", len({r["claimed_by"] for r in victims}), "workers", flush=True)
time.sleep(150)                                                   # workers poll the queue; let the 2-GPU jobs claim
stamp = time.strftime("%m%d%H%M", time.gmtime())
for r in victims:
    q.add(q.conn(), f"chiron_{r['name'].rsplit('_', 1)[0].replace('chiron_', '', 1)}_{stamp}", r["cmd"], gpus=1, lane="chiron",
          priority=r["priority"])
for n in held:
    (QUEUE / "held" / n).rename(QUEUE / "queued" / n)
print("requeued the stopped jobs and released", len(held), "held 1-GPU jobs", flush=True)
