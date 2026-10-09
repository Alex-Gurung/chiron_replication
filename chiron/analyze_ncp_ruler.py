"""Summarise ncp_ruler_sheets.py: what each character sheet adds on the NCP writer ruler.

  python3 chiron/analyze_ncp_ruler.py [--model Qwen3-4B-Instruct-2507]
Per section d = mean logprob with the sheet - mean logprob of the reference (nats per token, as v70_score_writer's
chapter_delta_mean_logprob); B = 100 * (1 - exp(-d)) (the q4v3 / paper unit). Reports, against three references (no
character sheets at all, only the supporting cast, and the shipped v2 sheets): the mean over sections, the mean of the
per-book means with the number of books where the sheet is better, for all books and for the canonical test cohort's
reported sections (exclusions/test_reported_1430.json). Checks the shipped-sheet scores against the stored canonical
controls (rl/ground_truths). Writes outputs/analysis_ncp_ruler_<model>.json.
"""
import argparse
import collections
import json
import math
from pathlib import Path

from common import OUT, read_jsonl, write_json

Q4 = Path("/home/toolkit/ncp_q4_v2_20260908")
KEPT = Path("/home/toolkit/diversity/diversity/ncp_eval/exclusions/test_reported_1430.json")
ORDER = ["none", "cast", "v2", "legacy", "charmem", "summary", "sheet_csum_flat", "sheet_legsum_x_500_flat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen3-4B-Instruct-2507")
    args = ap.parse_args()
    rows = [r for f in (OUT / "ncp_ruler" / args.model).glob("scores.*.jsonl") for r in read_jsonl(f)]
    scored = [r for r in rows if "scores" in r]
    print(len(rows), "sections,", len(rows) - len(scored), "skipped (over the window)")
    diffs = []
    for r in scored:
        p = Q4 / "rl" / "ground_truths" / f"ncp:{r['example_id']}.json"
        if p.exists():
            diffs.append(abs(json.load(open(p))["control_mean_logprob"] - r["scores"]["v2"]))
    if diffs:
        print(f"shipped-sheet score vs the stored canonical control: {len(diffs)} sections, max |diff| {max(diffs):.2e} nats, mean {sum(diffs) / len(diffs):.2e}")
    excluded = {x["section"] for x in json.load(open(KEPT))["excluded"]}
    result = {}
    for name, sel in (("all books", scored), ("canonical test cohort, reported sections", [r for r in scored if r["split"] == "test" and r["example_id"] not in excluded])):
        books = sorted({r["book"] for r in sel})
        print(f"\n== {name}: {len(sel)} sections, {len(books)} books")
        base = sum(r["scores"]["v2"] for r in sel) / len(sel)
        print(f"   shipped v2 sheets: mean logprob {base:.4f} nats/token (perplexity {math.exp(-base):.2f})")
        table = {}
        for ref in ("none", "cast", "v2"):
            print(f"   vs {ref:5s}  {'sheet':26s} {'nats/token':>10s} {'B%':>7s}   per-book mean   books better")
            for c in ORDER:
                if c == ref:
                    continue
                d = [r["scores"][c] - r["scores"][ref] for r in sel]
                per = collections.defaultdict(list)
                for r, x in zip(sel, d):
                    per[r["book"]].append(x)
                bm = [sum(v) / len(v) for v in per.values()]
                row = {"delta": sum(d) / len(d), "B": sum(100 * (1 - math.exp(-x)) for x in d) / len(d),
                       "book_mean": sum(bm) / len(bm), "books_better": sum(x > 0 for x in bm), "books": len(bm), "sections": len(d)}
                table[f"{c}|{ref}"] = row
                print(f"              {c:26s} {row['delta']:+10.4f} {row['B']:+7.2f}   {row['book_mean']:+10.4f}      {row['books_better']}/{row['books']}")
        result[name] = table
    write_json(OUT / f"analysis_ncp_ruler_{args.model}.json", result)


if __name__ == "__main__":
    main()
