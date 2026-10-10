"""Keep queued chiron jobs on the lane of a worker pod that is actually running.

  python3 scripts/lane_balance.py
The two worker pods serve one lane each (chiron_w97: chiron_l70, chiron_w98: chiron) and the cluster preempts them
independently. With one pod running, every queued chiron job moves to its lane; with both running, queued jobs are
split between the lanes (jobs needing 4 GPUs stay together on chiron_l70); with none running nothing changes. Prints one
line when it moves anything.
"""
import glob
import json
import os
import subprocess

Q = "/home/toolkit/eaiexp/state/queue"
LANES = {"chiron_w97": "chiron_l70", "chiron_w98": "chiron"}


def main():
    out = subprocess.run(["eai", "job", "ls", "--me", "--state", "all", "--fields", "name,state"], capture_output=True, text=True).stdout
    up = sorted({lane for line in out.split("\n") for w, lane in LANES.items() if line.startswith(w) and "RUNNING" in line})
    if not up:
        return
    moved = 0
    files = sorted(glob.glob(f"{Q}/queued/chiron_*.json"))
    for i, f in enumerate(files):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):                    # claimed while we looked
            continue
        want = up[0] if len(up) == 1 else ("chiron_l70" if r["gpus"] >= 4 else up[i % 2])
        if r["lane"] != want:
            r["lane"] = want
            json.dump(r, open(f + ".tmp", "w"))
            if os.path.exists(f):
                os.replace(f + ".tmp", f)
                moved += 1
            else:
                os.remove(f + ".tmp")
    if moved:
        print(f"lane balance: {moved} queued jobs -> {', '.join(up)}", flush=True)


if __name__ == "__main__":
    main()
