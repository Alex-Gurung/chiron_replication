"""Summarise perplexity results: how much each representation lowers the loss on the real next passage.

  python3 chiron/analyze_ppl.py
For each model directory under outputs/ppl and each context (none, story): token-weighted mean NLL per
representation, the change vs names only (nats per token and perplexity ratio), and per-book paired
differences (books with >= 10 passages: mean, books improved, 95% bootstrap CI). Writes outputs/analysis_ppl.json.
"""
import collections
import glob
import json
import math
import os
import random
import re

from common import OUT, read_jsonl, write_json


def boot(xs, n=2000, seed=0):
    rng = random.Random(seed)
    m = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def main():
    result = {}
    for mdir in sorted(glob.glob(str(OUT / "ppl" / "*"))):
        model = os.path.basename(mdir)
        rows = collections.defaultdict(dict)                # (ctx, rep) -> item -> (book, nll, tokens)
        for f in glob.glob(f"{mdir}/*/*.jsonl"):
            if f.endswith(".errors.jsonl"):
                continue
            ctx, rep = os.path.basename(f)[:-6].split("__")
            rep = re.sub(r"\.s\d+of\d+$", "", rep)                 # shard files of one representation
            for r in read_jsonl(f):
                rows[(ctx, rep)][r["item_id"]] = (r["book"], r["nll"], r["tokens"])
        for ctx in sorted({c for c, _ in rows}):
            base = rows.get((ctx, "names"))
            if not base:
                continue
            per_book_n = collections.Counter(b for b, _, _ in base.values())
            books = {b for b, n in per_book_n.items() if n >= 10}
            table = {}
            for (c, rep), d in rows.items():
                if c != ctx:
                    continue
                common = [i for i in d if i in base]
                if not common:
                    continue
                nll = sum(d[i][1] for i in common) / sum(d[i][2] for i in common)
                nll0 = sum(base[i][1] for i in common) / sum(base[i][2] for i in common)
                bybook = collections.defaultdict(lambda: [0.0, 0.0, 0])
                for i in common:
                    b = d[i][0]
                    bybook[b][0] += d[i][1] - base[i][1]
                    bybook[b][1] += d[i][2]
                diffs = [v[0] / v[1] for b, v in bybook.items() if b in books and v[1]]
                lo, hi = boot(diffs) if len(diffs) >= 3 else (float("nan"), float("nan"))
                table[rep] = {"nll": nll, "delta": nll - nll0, "ppl_ratio": math.exp(nll - nll0), "passages": len(common),
                              "books": len(diffs), "books_better": sum(x < 0 for x in diffs),
                              "book_mean_delta": sum(diffs) / len(diffs) if diffs else None, "ci": [lo, hi]}
            result[f"{model}|{ctx}"] = table
            print(f"\n{model}  context={ctx}")
            print(f"{'representation':16s} {'nll/tok':>8s} {'Δ nats':>8s} {'ppl x':>7s}  {'books better, 95% CI (per-book Δ)':>36s}  passages")
            for rep, t in sorted(table.items(), key=lambda kv: kv[1]["nll"]):
                print(f"{rep:16s} {t['nll']:8.4f} {t['delta']:+8.4f} {t['ppl_ratio']:7.4f}  "
                      f"{t['books_better']:>3d}/{t['books']:<3d} [{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}]          {t['passages']}")
    write_json(OUT / "analysis_ppl.json", result)


if __name__ == "__main__":
    main()
