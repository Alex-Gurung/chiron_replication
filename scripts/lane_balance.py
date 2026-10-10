"""Keep queued chiron jobs on the lane of a worker pod that is actually running.

  python3 scripts/lane_balance.py
The 8-GPU pod chiron_w97 serves lane chiron_l70; every other chiron_w* pod (1-GPU pods, which the cluster preempts far
less often) serves lane chiron. Jobs needing 2 or more GPUs always go to chiron_l70. 1-GPU jobs go to whichever lane
has a running pod, split between the two when both do; with none running nothing changes. Prints one
line when it moves anything.
"""
import glob
import json
import os
import subprocess

Q = "/home/toolkit/eaiexp/state/queue"


def main():
    out = subprocess.run(["eai", "job", "ls", "--me", "--state", "all", "--fields", "name,state"], capture_output=True, text=True).stdout
    up = sorted({("chiron_l70" if line.startswith("chiron_w97") else "chiron") for line in out.split("\n")
                 if line.startswith("chiron_w") and "RUNNING" in line})
    if not up:
        return
    moved = 0
    files = sorted(glob.glob(f"{Q}/queued/chiron_*.json"))
    for i, f in enumerate(files):
        try:
            r = json.load(open(f))
        except (OSError, ValueError):                    # claimed while we looked
            continue
        if r["gpus"] >= 2:                               # only the 8-GPU pod (chiron_l70) can run multi-GPU jobs
            want = "chiron_l70"
        else:
            want = up[0] if len(up) == 1 else up[i % 2]
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
