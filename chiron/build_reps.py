"""Assemble every character representation for the items of a split -> data/reps_<split>.jsonl.

One record per (condition, book, boundary, label); every source is built from chapters < boundary:
  v2             character_sheets in ncp_cohorts_v2 (principal sheet only)
  legacy         Llama-3.3-70B CHIRON-style sheet, compressed (~500 words), HF agurung/new_ncp_data_creation
  legacy_full    the same sheet before compression
  summary        gpt-oss rolling character summary S_b (outputs/summary)
  chiron         gpt-oss CHIRON-style statements rated 5, grouped by category, TF-IDF dedup at 0.9
  chiron_<cat>   one category only (for the per-category "Agreed" setting)
  charmem        finished Sep 8 gpt-oss rebuild sheet (ncp_charmem_gptoss120b_reviewed_20260908/sheets)
Run with the repo venv (needs scikit-learn): .venv/bin/python chiron/build_reps.py --split test
"""
import argparse
import collections
import glob
import json
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common import COHORTS, DATA, OUT, read_jsonl

CATEGORIES = ["Physical/Personality", "Dialogue", "Knowledge", "Goals"]
CHARMEM = Path("/home/toolkit/ncp_charmem_gptoss120b_reviewed_20260908/sheets")
LEGACY = DATA / "legacy"


def dedup(statements, threshold=0.9):
    if len(statements) < 2:
        return statements
    sims = cosine_similarity(TfidfVectorizer().fit_transform(statements))
    keep = []
    for i in range(len(statements)):
        if all(sims[i, j] < threshold for j in keep):
            keep.append(i)
    return [statements[i] for i in keep]


def chiron_sheets(keys):
    """keys: set of (book, boundary, label). Statements come from snippets of chapters < boundary, in book order."""
    by = collections.defaultdict(list)
    for f in glob.glob(str(OUT / "chiron" / "shards" / "*.jsonl")):
        if f.endswith(".errors.jsonl"):
            continue
        for r in read_jsonl(f):
            for c in r["claims"]:
                if c["rating"] == 5:
                    by[(r["book"], r["label"])].append((r["chapter_index"], r["idx"], c["category"], c["text"]))
    out = {}
    for book, b, label in keys:
        rows = sorted(x for x in by[(book, label)] if x[0] < b)
        kept = {cat: set(dedup([t for _, _, c, t in rows if c == cat])) for cat in CATEGORIES}
        out[(book, b, label)] = [(c, t) for _, _, c, t in rows if t in kept[c] and not kept[c].discard(t)]
    return out


def budget(rows, k):
    """Most recent statements first until k words, then back in book order."""
    picked, n = [], 0
    for c, t in reversed(rows):
        if n + len(t.split()) > k:
            break
        picked.append((c, t))
        n += len(t.split())
    return picked[::-1]


def render(rows, only=None):
    parts = [f"### {cat}\n" + ("\n".join(f"- {t}" for c, t in rows if c == cat) or "- No information.")
             for cat in CATEGORIES if only in (None, cat)]
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    items = read_jsonl(DATA / f"items_{args.split}.jsonl") + read_jsonl(DATA / f"items_{args.split}_two.jsonl")
    keys = {(it["book"], it["chapter_index"], l) for it in items for l in it["labels"]}
    books = {k[0] for k in keys}
    recs = []

    def add(cond, book, b, label, text):
        if text is not None:
            recs.append({"condition": cond, "book": book, "boundary": b, "label": label, "text": text,
                         "words": len(text.split())})

    seen = set()
    for line in open(COHORTS / f"{args.split}_examples.jsonl"):
        r = json.loads(line)
        for l, t in r["character_sheets"].items():
            k = (r["story_id"], r["chapter_index"], l)
            if k in keys and k not in seen:
                seen.add(k)
                add("v2", *k, t)
    full = pickle.load(open(LEGACY / f"{args.split}_long_story_storycharchap_to_csheet.pkl", "rb"))
    comp = pickle.load(open(LEGACY / f"{args.split}_long_story_character_sheet_summaryllama70B_0.5max.pkl", "rb"))
    for book, b, l in keys:
        add("legacy", book, b, l, comp.get(f"{book}_{l}_{b}"))
        add("legacy_full", book, b, l, full.get(f"{book}_{l}_{b}"))
    summ = {}
    for f in glob.glob(str(OUT / "summary_v2" / "*.jsonl")):       # v1 (outputs/summary) grew past the cap; superseded
        if f.endswith(".errors.jsonl"):
            continue
        for r in read_jsonl(f):
            summ[(r["book"], r["boundary"], r["label"])] = r["summary"]
    for k in keys:
        add("summary", *k, summ.get(k))
    for k, rows in chiron_sheets(keys).items():
        add("chiron", *k, render(rows))
        for cat in CATEGORIES:
            add("chiron_" + cat.split("/")[0].lower(), *k, render(rows, cat))
        for words in (250, 500, 1000, 2000, 4000):
            add(f"chiron_r{words}", *k, render(budget(rows, words)))
    for book, b, l in keys:
        p = CHARMEM / f"{book}__{b:04d}.json"
        if p.exists():
            add("charmem", book, b, l, json.load(open(p))["character_sheets"].get(l))
    with open(DATA / f"reps_{args.split}.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cov = collections.Counter(r["condition"] for r in recs)
    words = collections.defaultdict(list)
    for r in recs:
        words[r["condition"]].append(r["words"])
    for c in sorted(cov):
        w = sorted(words[c])
        print(f"{c:22s} {cov[c]:5d}/{len(keys)} keys  median words {w[len(w) // 2]}")


if __name__ == "__main__":
    main()
