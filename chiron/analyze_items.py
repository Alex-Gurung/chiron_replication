"""Item-level structure of the results: how much of correctness is the passage vs the representation.

  python3 chiron/analyze_items.py [--model Qwen3.8-27B_nothink]
Per (passage, character): mean correctness over block orders for each representation. Reports the share of
variance explained by the passage/character and by the representation, how often every representation is right
or wrong, the best-single vs best-per-passage ("oracle pick") accuracy, and pairwise agreement.
Writes outputs/analysis_items_<model>.json.
"""
import argparse
import collections
import glob
import itertools
import json
import statistics as st

from common import OUT, read_jsonl, write_json

CONDS = ["v2", "charmem", "chiron", "summary", "legacy", "legacy_full", "book_last8000"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3.8-27B_nothink")
    args = ap.parse_args()
    acc = collections.defaultdict(dict)
    for c in CONDS:
        tmp = collections.defaultdict(list)
        for f in glob.glob(str(OUT / "eval" / args.model / "items_*" / f"{c}.*jsonl")) + glob.glob(str(OUT / "eval" / args.model / "items_*" / f"{c}.jsonl")):
            if not any(f"/items_{s}/" in f for s in ("test", "val", "train")):
                continue
            for r in read_jsonl(f):
                lp = r["logprobs"]
                tmp[(r["item_id"], r["target"])].append(int(not r.get("invalid") and max(lp, key=lp.get) == str(r["answer"])))
        for k, v in tmp.items():
            acc[k][c] = sum(v) / len(v)
    keys = [k for k, v in acc.items() if len(v) == len(CONDS)]
    X = [[acc[k][c] for c in CONDS] for k in keys]
    grand = st.mean(v for row in X for v in row)
    ss_tot = sum((v - grand) ** 2 for row in X for v in row)
    ss_item = len(CONDS) * sum((st.mean(row) - grand) ** 2 for row in X)
    ss_cond = len(X) * sum((st.mean(row[j] for row in X) - grand) ** 2 for j in range(len(CONDS)))
    maj = [[v > 0.5 for v in row] for row in X]
    res = {"targets": len(keys), "var_passage": ss_item / ss_tot, "var_representation": ss_cond / ss_tot,
           "all_right": sum(all(r) for r in maj) / len(maj), "all_wrong": sum(not any(r) for r in maj) / len(maj),
           "best_single": max(sum(r[j] for r in maj) / len(maj) for j in range(len(CONDS))),
           "oracle_pick": sum(any(r) for r in maj) / len(maj),
           "agreement": {f"{a}|{b}": sum(r[i] == r[j] for r in maj) / len(maj)
                         for (i, a), (j, b) in itertools.combinations(enumerate(CONDS), 2)}}
    write_json(OUT / f"analysis_items_{args.model}.json", res)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items() if k != "agreement"}))


if __name__ == "__main__":
    main()
