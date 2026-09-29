"""No-LLM baseline: match each masked character to a representation by word overlap.

For each passage, the context of mask i is the words within WIN tokens of each [CHAR i]. Each character's
representation (for book text: the sentences of that text that name the character) is compared to each
context by TF-IDF cosine over content words (fit on the passage plus the representations); the best
one-to-one assignment is the prediction. Reports per-character accuracy per representation, main set.
Run with the repo venv: .venv/bin/python chiron/lexical_baseline.py
"""
import itertools
import json
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, "chiron")
from build_items import alias_regex  # noqa: E402
from common import DATA, REPO, SENT, clean_text, load_chapters, read_jsonl  # noqa: E402

WIN = 30
CONDS = ["v2", "legacy", "chiron_r2000", "charmem", "summary", "chiron", "legacy_full", "book_last8000", "book_last32000"]


def contexts(masked, n):
    toks = masked.split()
    out = {i: [] for i in range(n)}
    for j, t in enumerate(toks):
        m = re.search(r"\[CHAR$", t)
        if t.startswith("[CHAR") and j + 1 < len(toks):
            d = re.match(r"(\d)", toks[j + 1])
            if d and int(d.group(1)) in out:
                out[int(d.group(1))].append(" ".join(toks[max(0, j - WIN):j] + toks[j + 2:j + 2 + WIN]))
    return {i: " ".join(v) for i, v in out.items()}


def main():
    chapters = load_chapters()
    principals = json.load(open(DATA / "principals.json"))
    res = {c: [0, 0] for c in CONDS}
    for split in ("test", "val", "train"):
        reps = {(r["condition"], r["book"], r["boundary"], r["label"]): r["text"] for r in read_jsonl(DATA / f"reps_{split}.jsonl")}
        for it in read_jsonl(DATA / f"items_{split}.jsonl"):
            book, b, labels = it["book"], it["chapter_index"], it["labels"]
            ctx = contexts(it["masked"], len(labels))
            if not all(ctx.values()):
                continue
            al = json.load(open(REPO / "aliases" / f"{book}.json"))["principals"]
            prior = None
            for c in CONDS:
                if c.startswith("book_last"):
                    if prior is None:
                        prior = " ".join(clean_text(ch["chapter_text_normalized"]) for ch in chapters[book] if ch["chapter_index"] < b).split()
                    text = " ".join(prior[-int(c[9:]):])
                    sents = SENT.split(text)
                    rep = {l: " ".join(s for s in sents if alias_regex(al[l]).search(s)) for l in labels}
                else:
                    rep = {l: reps.get((c, book, b, l)) for l in labels}
                if any(not v for v in rep.values()):
                    continue
                strip = lambda s: re.sub(r"\[CHAR \d\]|\[Ch[s]?\.[^\]]*\]|<snippet \d+>", " ", s)
                docs = [strip(ctx[i]) for i in range(len(labels))] + [strip(rep[l]) for l in labels]
                try:
                    X = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(docs)
                except ValueError:
                    continue
                sim = cosine_similarity(X[len(labels):], X[:len(labels)])      # rows: characters, cols: mask ids
                best = max(itertools.permutations(range(len(labels))), key=lambda p: sum(sim[k][p[k]] for k in range(len(labels))))
                for k, l in enumerate(labels):
                    res[c][0] += int(best[k] == it["answer"][l])
                    res[c][1] += 1
    for c in CONDS:
        print(f"{c:16s} lexical matcher accuracy {100 * res[c][0] / res[c][1]:5.1f}%  (n={res[c][1]})")


if __name__ == "__main__":
    main()
