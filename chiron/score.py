"""Score eval outputs: per-character accuracy (paper-style argmax) and one-to-one assignment accuracy.

  python3 chiron/score.py [--model Qwen3-4B-Instruct-2507] [--items items_test]
Per condition: overall and per-book per-character accuracy (independent argmax over the three ids),
assignment accuracy (best one-to-one mapping of the three targets), all-three-correct rate, mean
prompt tokens, and the share of answers whose top token was not a digit. Averages over orders.
Writes outputs/eval/<model>/<items>/scores.json.
"""
import argparse
import collections
import glob
import itertools
import json
import math
import os

from common import OUT, read_jsonl, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B-Instruct-2507")
    ap.add_argument("--items", default="items_test")
    args = ap.parse_args()
    root = OUT / "eval" / args.model / args.items
    table = {}
    for f in sorted(glob.glob(str(root / "*.jsonl"))):
        cond = os.path.basename(f)[:-6]
        rows = read_jsonl(f)
        groups = collections.defaultdict(dict)
        for r in rows:
            groups[(r["item_id"], tuple(r["order"]))][r["target"]] = r
        stats = collections.defaultdict(lambda: collections.Counter())
        for (item, order), g in groups.items():
            book = next(iter(g.values()))["book"]
            for key in ("all", book):
                s = stats[key]
                for r in g.values():
                    lp = r["logprobs"]
                    s["char_n"] += 1
                    s["char_ok"] += int(max(lp, key=lp.get) == str(r["answer"]))
                    s["nondigit"] += int(r["top_token"].strip() not in "012" or not r["top_token"].strip())
                    s["tokens"] += r["prompt_tokens"] or 0
                if len(g) == 3:
                    targets = sorted(g)
                    best = max(itertools.permutations("012"), key=lambda p: sum(
                        g[t]["logprobs"][d] if g[t]["logprobs"][d] > -math.inf else -1e9 for t, d in zip(targets, p)))
                    hits = [d == str(g[t]["answer"]) for t, d in zip(targets, best)]
                    s["assign_n"] += 3
                    s["assign_ok"] += sum(hits)
                    s["all3_n"] += 1
                    s["all3_ok"] += int(all(hits))
        table[cond] = {k: {"char_acc": s["char_ok"] / s["char_n"], "assign_acc": s["assign_ok"] / max(1, s["assign_n"]),
                           "all3": s["all3_ok"] / max(1, s["all3_n"]), "n_char": s["char_n"],
                           "mean_tokens": s["tokens"] / s["char_n"], "nondigit": s["nondigit"] / s["char_n"]}
                       for k, s in stats.items()}
    write_json(root / "scores.json", table)
    books = sorted({k for v in table.values() for k in v if k != "all"})
    print(f"{'condition':24s} {'char':>6s} {'assign':>6s} {'all3':>6s} {'tokens':>7s}  " + "  ".join(f"{b:>6s}" for b in books))
    for cond, v in sorted(table.items(), key=lambda kv: kv[1]["all"]["char_acc"]):
        a = v["all"]
        print(f"{cond:24s} {a['char_acc']:6.3f} {a['assign_acc']:6.3f} {a['all3']:6.3f} {a['mean_tokens']:7.0f}  "
              + "  ".join(f"{v[b]['char_acc']:6.3f}" if b in v else "     -" for b in books)
              + (f"   nondigit={a['nondigit']:.2f}" if a["nondigit"] > 0.01 else ""))


if __name__ == "__main__":
    main()
