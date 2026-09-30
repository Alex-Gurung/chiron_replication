"""Hand-written oracles vs everything else, on the passages the agents annotated (manual_oracle.py).

  python3 chiron/analyze_manual.py
For each model and condition: accuracy over (passage, character, block order) on the sampled passages (joint
scoring for thinking-off runs, see analyze.py), with a 95% bootstrap interval over passages, and the paired
difference against v2 on the same passages. The verbatim-quote oracle is also reported over every passage. Writes outputs/analysis_manual.json.
"""
import collections
import glob
import json
import random

from common import DATA, OUT, joint_correct, read_jsonl, write_json

MODELS = ["Qwen3-4B-Instruct-2507", "Qwen3.5-9B-Base", "Qwen3.8-27B_nothink", "Qwen3.8-27B_think"]
CONDS = ["noinfo", "v2", "legacy_full", "oracle_prior", "manual_prior", "oracle_passage", "manual_passage", "oracle_quote",
         "swapname_manual_prior"]


def per_item(model, cond):
    acc, groups = collections.defaultdict(list), collections.defaultdict(dict)
    for f in glob.glob(str(OUT / "eval" / model / "items_*" / f"{cond}.*jsonl")) + glob.glob(str(OUT / "eval" / model / "items_*" / f"{cond}.jsonl")):
        if f.endswith(".errors.jsonl") or not any(f"/items_{s}/" in f for s in ("test", "val", "train")):
            continue
        for r in read_jsonl(f):
            groups[(r["item_id"], tuple(r["order"]))][r["target"]] = (r["logprobs"], r["answer"], r.get("invalid"))
    for (iid, order), d in groups.items():
        if model.endswith("_think"):                          # full mappings already; unreadable counts wrong
            for lp, a, inv in d.values():
                acc[iid].append(int(not inv and max(lp, key=lp.get) == str(a)))
        elif len(d) == len(order):
            acc[iid] += list(joint_correct({t: v[:2] for t, v in d.items()}).values())
    return {i: sum(v) / len(v) for i, v in acc.items()}


def boot(xs, n=2000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return [m[int(0.025 * n)], m[int(0.975 * n)]]


def main():
    sample = [t["item_id"] for k in range(7) for t in json.load(open(DATA / "manual" / f"tasks_{k}.json"))]
    out = {"passages": len(sample), "rows": {}}
    for m in MODELS:
        res = {c: per_item(m, c) for c in CONDS}
        for c in CONDS:
            ids = [i for i in sample if i in res[c]]
            if not ids:
                continue
            xs = [res[c][i] for i in ids]
            row = {"acc": sum(xs) / len(xs), "ci": boot(xs), "passages": len(ids)}
            shared = [i for i in ids if i in res["v2"]]
            if c != "v2" and shared:
                d = [res[c][i] - res["v2"][i] for i in shared]
                row["vs_v2"] = {"mean": sum(d) / len(d), "ci": boot(d)}
            if c == "oracle_quote":
                allx = list(res[c].values())
                row["all_passages"] = {"acc": sum(allx) / len(allx), "passages": len(allx)}
            out["rows"][f"{m}|{c}"] = row
    for k, v in out["rows"].items():
        print(f"{k:45s} {100 * v['acc']:5.1f} [{100 * v['ci'][0]:.1f}, {100 * v['ci'][1]:.1f}] n={v['passages']}"
              + (f"  vs v2 {100 * v['vs_v2']['mean']:+.1f}" if "vs_v2" in v else "")
              + (f"  all passages {100 * v['all_passages']['acc']:.1f} (n={v['all_passages']['passages']})" if "all_passages" in v else ""))
    write_json(OUT / "analysis_manual.json", out)


if __name__ == "__main__":
    main()
