"""Harvest PERSON mentions per book with spaCy, as input to the hand-checked alias table.

Writes data/ner/<book>.json: {name: {"count": n, "first_chapter": c, "chapters": k}} over the
cleaned, NFC chapter text. Run with the repo venv: .venv/bin/python chiron/ner.py --books ...
"""
import argparse
import collections
import json

import spacy

from common import DATA, clean_text, load_chapters, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--model", default="en_core_web_trf")
    args = ap.parse_args()
    nlp = spacy.load(args.model, disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    nlp.max_length = 3_000_000
    chapters = load_chapters()
    for book in args.books:
        stats = collections.defaultdict(lambda: {"count": 0, "first_chapter": None, "chapters": set()})
        for ch in chapters[book]:
            c = ch["chapter_index"]
            text = clean_text(ch["chapter_text_normalized"])
            for doc in nlp.pipe([text[i:i + 20000] for i in range(0, len(text), 20000)], batch_size=4):
                for ent in doc.ents:
                    if ent.label_ != "PERSON":
                        continue
                    s = stats[ent.text.strip()]
                    s["count"] += 1
                    s["chapters"].add(c)
                    s["first_chapter"] = c if s["first_chapter"] is None else min(s["first_chapter"], c)
        out = {k: {"count": v["count"], "first_chapter": v["first_chapter"], "chapters": len(v["chapters"])}
               for k, v in sorted(stats.items(), key=lambda kv: -kv[1]["count"])}
        write_json(DATA / "ner" / f"{book}.json", out)
        print(book, len(out), "names; top:", list(out)[:15], flush=True)


if __name__ == "__main__":
    main()
