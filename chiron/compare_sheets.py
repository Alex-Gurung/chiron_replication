"""Quick Pareto table for the new sheets against the existing representations (reads analysis_main_<model>.json).

  python3 chiron/compare_sheets.py [--model Qwen3.8-27B_nothink] [--pre]
Per condition: joint macro accuracy (books covered), mean whole-prompt tokens, and the per-book paired difference
against charmem (mean, books better / total). Marks the conditions no other listed non-oracle condition dominates.
"""
import argparse
import json

from common import OUT

REF = ["noinfo", "gender", "v2", "charmem", "summary", "legacy", "plot_global_1000", "plot_hier_4000", "chiron_r2000",
       "book_last8000", "chapnotes", "chapnotes_h_long", "legacy_match", "legacy_r6000", "legacy_full", "legacy_gptoss",
       "legacy_gptoss_nofill", "legacy_r1000", "legacy_r2000", "legacy_r3000", "chapnotes_h_long_r1000", "chapnotes_h_long_r2000",
       "index_lgp", "v2+index", "charmem+index", "charmem+last1", "charmem+last2", "v2+last1", "v2+last2", "chapnotes_h_long_s1", "legacy_gptoss_nofill_s1", "ledger_full", "ledger_c2", "ledger_c4", "legacy_gptoss_ent", "legacy_gptoss_ent4", "charmem&book_last500", "charmem&book_last1000", "charmem&book_last2000", "v2&book_last1000", "book_last2000", "book_last1000", "book_last500", "legacy&book_last1000", "sheet_legsum_ent_w800&book_last1000", "sheet_csyn_ldg&book_last1000", "charmem&plot_hier_4000@last500", "charmem&plot_global_500", "summary&book_last1000"]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen3.8-27B_nothink")
ap.add_argument("--pre", action="store_true")
args = ap.parse_args()
d = json.load(open(OUT / f"analysis_main_{args.model}.json"))
rows, pbj = d["rows"], d["per_book_joint"]
suf = "+pre" if args.pre else ""
conds = [c + suf for c in REF + sorted(c for c in rows if c.startswith("sheet_") and "+" not in c) if c + suf in rows]
acc = {c: rows[c]["joint"]["macro"] for c in conds}
tok = {c: rows[c]["mean_tokens"] for c in conds}
front = {c for c in conds if not any(acc[o] >= acc[c] and tok[o] <= tok[c] and (acc[o], tok[o]) != (acc[c], tok[c]) for o in conds)}
base = pbj.get("charmem" + suf, {})
for c in sorted(conds, key=lambda c: tok[c]):
    diffs = [pbj[c][b] - base[b] for b in d["books"] if b in pbj.get(c, {}) and b in base]
    vs = f"{100 * sum(diffs) / len(diffs):+5.1f} ({sum(x > 0 for x in diffs)}/{len(diffs)})" if diffs else ""
    print(f"{'*' if c in front else ' '} {c:28s} {100 * acc[c]:5.1f}  b{rows[c]['joint']['books']:<3d} n{rows[c]['items']:<5d}"
          f"{tok[c]:7.0f} tok   vs charmem {vs}")
