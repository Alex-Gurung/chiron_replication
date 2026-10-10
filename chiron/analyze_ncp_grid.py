"""Summarise ncp_ruler_grid.py: per condition, the gain in the true next section's log-probability over a reference.

  python3 chiron/analyze_ncp_grid.py [--model Qwen3-4B-Instruct-2507]
Merges every tag's files of the model (a section's scores from different waves are joined; the references each wave
repeats are checked to agree). For each condition c and each reference r in ("none|none", "ship|none", "ship|v2") that
shares sections with it: d = score(c) - score(r) per section (nats per token), its mean, B = 100 * (1 - exp(-d))
averaged, the mean of per-book means, and the number of books where c is better. Writes
outputs/analysis_ncp_grid_<model>.json {"rows": {"c||r": {...}}, "tokens": {c: mean prompt tokens}, "sections", "books"}.
"""
import argparse
import collections
import math

from common import OUT, read_jsonl, write_json

REFS = ("none|none", "ship|none", "ship|v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B-Instruct-2507")
    args = ap.parse_args()
    sec, book, toks, worst = collections.defaultdict(dict), {}, collections.defaultdict(list), 0.0
    for f in sorted((OUT / "ncp_grid" / args.model).glob("*.jsonl")):
        for r in read_jsonl(f):
            if "scores" not in r:
                continue
            book[r["example_id"]] = r["book"]
            for c, v in r["scores"].items():
                if c in sec[r["example_id"]]:
                    worst = max(worst, abs(sec[r["example_id"]][c] - v))
                else:
                    sec[r["example_id"]][c] = v
                    toks[c].append(r["prompt_tokens"][c])
    conds = sorted({c for d in sec.values() for c in d})
    print(len(sec), "sections,", len(set(book.values())), "books,", len(conds), "conditions; repeated references differ by at most", f"{worst:.1e}", "nats")
    rows = {}
    for c in conds:
        for ref in REFS:
            if c == ref:
                continue
            ids = [i for i, d in sec.items() if c in d and ref in d]
            if not ids:
                continue
            d = [sec[i][c] - sec[i][ref] for i in ids]
            per = collections.defaultdict(list)
            for i, x in zip(ids, d):
                per[book[i]].append(x)
            bm = [sum(v) / len(v) for v in per.values()]
            rows[f"{c}||{ref}"] = {"delta": sum(d) / len(d), "B": sum(100 * (1 - math.exp(-x)) for x in d) / len(d),
                                   "book_mean": sum(bm) / len(bm), "books_better": sum(x > 0 for x in bm), "books": len(bm), "sections": len(d)}
    for ref in REFS:
        print(f"\nvs {ref}:  {'condition':52s} {'nats/token':>10s} {'B%':>7s}  books better  sections  tokens")
        for c in conds:
            r = rows.get(f"{c}||{ref}")
            if r:
                print(f"           {c:52s} {r['delta']:+10.4f} {r['B']:+7.2f}   {r['books_better']:3d}/{r['books']:<3d}    {r['sections']:5d}  {sum(toks[c]) / len(toks[c]) / 1000:5.1f}k")
    write_json(OUT / f"analysis_ncp_grid_{args.model}.json", {"rows": rows, "tokens": {c: sum(v) / len(v) for c, v in toks.items()},
                                                               "sections": len(sec), "books": len(set(book.values()))})


if __name__ == "__main__":
    main()
