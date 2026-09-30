"""Cross-book analysis of eval outputs for one item set across all splits.

  python3 chiron/analyze.py [--set main|two|pron] [--model Qwen3-4B-Instruct-2507] [--min-items 10]
Per condition: pooled per-character accuracy over all items, macro accuracy over books (each book
weighted equally, books with >= min-items items), and paired per-book differences against noinfo and
v2 (mean, books positive / total, 95% bootstrap CI over books). Books, not sections, are the unit.
Writes outputs/analysis_<set>_<model>.json and prints a table.
"""
import argparse
import collections
import glob
import json
import os
import random

from common import OUT, joint_correct, read_jsonl, write_json


def load(model, stems):
    by = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))   # cond -> book -> [ok, n, tokens]
    toks, items = collections.defaultdict(list), collections.defaultdict(set)          # cond -> prompt lengths / item ids
    groups = collections.defaultdict(dict)         # (cond, item, order) -> target -> (logprobs, answer, book)
    by_answer = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))   # cond -> true id -> [ok, n]
    seen = set()                                   # dedupe across run files
    for stem in stems:
        root = OUT / "eval" / model / stem
        for f in glob.glob(str(root / "*.jsonl")):
            if f.endswith(".errors.jsonl"):
                continue
            cond = os.path.basename(f).split(".")[0]
            for r in read_jsonl(f):
                k = (cond, r["item_id"], tuple(r["order"]), r["target"])
                if k in seen:
                    continue
                seen.add(k)
                lp = r["logprobs"]
                cell = by[cond][r["book"]]
                cell[0] += int(not r.get("invalid") and max(lp, key=lp.get) == str(r["answer"]))
                cell[1] += 1
                cell[2] += r.get("prompt_tokens") or 0
                items[cond].add(r["item_id"])
                ba = by_answer[cond][r["answer"]]
                ba[0] += int(not r.get("invalid") and max(lp, key=lp.get) == str(r["answer"]))
                ba[1] += 1
                groups[(cond, r["item_id"], tuple(r["order"]))][r["target"]] = (lp, r["answer"], r["book"])
                if r.get("prompt_tokens"):
                    toks[cond].append((r["item_id"], r["target"], r["prompt_tokens"]))
    joint = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))   # cond -> book -> [ok, n]
    for (cond, _, order), d in groups.items():
        if len(d) != len(order):
            continue
        for t, ok in joint_correct({t: v[:2] for t, v in d.items()}).items():
            cell = joint[cond][d[t][2]]
            cell[0] += ok
            cell[1] += 1
    return by, toks, items, joint, by_answer


def boot(diffs, n=2000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="main", choices=["main", "two", "pron", "window", "short"])
    ap.add_argument("--model", default="Qwen3-4B-Instruct-2507")
    ap.add_argument("--min-items", type=int, default=10)
    args = ap.parse_args()
    suffix = {"main": "", "two": "_two", "pron": "_pron", "window": "_window", "short": "_short"}[args.set]
    by, toks, items, joint, by_answer = load(args.model, [f"items_{s}{suffix}" for s in ("test", "val", "train")])
    n_items = collections.Counter()
    for s in ("test", "val", "train"):
        for it in read_jsonl(OUT.parent / "data" / f"items_{s}{suffix}.jsonl"):
            n_items[it["book"]] += 1
    books = sorted(b for b, n in n_items.items() if n >= args.min_items)
    acc = {c: {b: v[0] / v[1] for b, v in d.items() if v[1]} for c, d in by.items()}
    bases = {}                                       # names-only prompt length per (passage, character); +pre: with the prefix
    for bc in ("noinfo", "noinfo+pre"):
        acc_ = collections.defaultdict(list)
        for i, t, n in toks.get(bc, []):
            acc_[(i, t)].append(n)
        bases[bc] = {k: sum(v) / len(v) for k, v in acc_.items()}
    quant = lambda xs: [sorted(xs)[int(q * (len(xs) - 1))] for q in (0.1, 0.25, 0.5, 0.75, 0.9)] if xs else None
    accj = {c: {b: v[0] / v[1] for b, v in d.items() if v[1]} for c, d in joint.items()}
    rows = {}
    for c, d in by.items():
        ok, n = sum(v[0] for v in d.values()), sum(v[1] for v in d.values())
        common = [b for b in books if b in acc[c]]
        row = {"pooled": ok / n if n else None, "n_trials": n, "books": len(common),
               "mean_tokens": sum(v[2] for v in d.values()) / n if n else None,
               "tokens_q": quant([n for _, _, n in toks[c]]),
               "rep_tokens_mean": None, "rep_tokens_q": None,
               "items": len(items[c]), "items_total": sum(n_items.values()),
               "by_answer": {a: v[0] / v[1] for a, v in sorted(by_answer[c].items())},
               "macro": sum(acc[c][b] for b in common) / len(common) if common else None}
        base = bases["noinfo+pre" if c.endswith("+pre") else "noinfo"]
        rep = [n - base[(i, t)] for i, t, n in toks[c] if (i, t) in base]      # the representation's own tokens (3 blocks)
        if rep and c not in ("noinfo", "noinfo+pre"):
            row["rep_tokens_mean"], row["rep_tokens_q"] = sum(rep) / len(rep), quant(rep)
        cj = [b for b in books if b in accj.get(c, {})]
        row["joint"] = {"macro": sum(accj[c][b] for b in cj) / len(cj) if cj else None, "books": len(cj)}
        for ref in ("noinfo", "v2"):
            shared = [b for b in cj if b in accj.get(ref, {})]
            if c != ref and len(shared) >= 3:
                diffs = [accj[c][b] - accj[ref][b] for b in shared]
                lo, hi = boot(diffs)
                row["joint"][f"vs_{ref}"] = {"mean": sum(diffs) / len(diffs), "pos": sum(x > 0 for x in diffs), "n": len(diffs), "ci": [lo, hi]}
        for ref in ("noinfo", "v2"):
            shared = [b for b in common if b in acc.get(ref, {})]
            if c != ref and len(shared) >= 3:
                diffs = [acc[c][b] - acc[ref][b] for b in shared]
                lo, hi = boot(diffs)
                row[f"vs_{ref}"] = {"mean": sum(diffs) / len(diffs), "pos": sum(x > 0 for x in diffs),
                                    "n": len(diffs), "ci": [lo, hi]}
        rows[c] = row
    write_json(OUT / f"analysis_{args.set}_{args.model}.json", {"books": books, "n_items": n_items, "rows": rows,
                                                                 "per_book": acc, "per_book_joint": accj})
    print(f"set={args.set} model={args.model} books with >={args.min_items} items: {len(books)}")
    print(f"{'condition':22s} {'pooled':>7s} {'macro':>7s} {'joint':>7s}  {'vs noinfo (mean, +books, 95% CI)':>36s}  {'vs v2':>30s}")
    for c, r in sorted(rows.items(), key=lambda kv: kv[1]["macro"] or 0):
        f = lambda k: (f"{r[k]['mean']:+.3f} {r[k]['pos']:2d}/{r[k]['n']:<2d} [{r[k]['ci'][0]:+.3f},{r[k]['ci'][1]:+.3f}]"
                       if k in r else "")
        print(f"{c:22s} {r['pooled']:7.3f} {(r['macro'] or 0):7.3f} {(r['joint']['macro'] or 0):7.3f}  {f('vs_noinfo'):>36s}  {f('vs_v2'):>30s}")


if __name__ == "__main__":
    main()
