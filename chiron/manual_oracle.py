"""Hand-written ("manual", by Claude agents) oracles and a verbatim-quote oracle for a sample of passages.

  python3 chiron/manual_oracle.py sample     3 passages per book (the 21 books with >= 10) -> data/manual/tasks_<k>.json,
                                              data/manual/tasks/<item>/{passage.md,notes/} for the agents, and
                                              data/manual/books/<book>.txt (book text with chapter markers)
  python3 chiron/manual_oracle.py collect    validate data/manual/out_<k>.jsonl (agent output) and the quote oracle
                                              -> data/manual/reps_<split>_manual.jsonl (read by oracle_reps.py export)
Conditions (item-level, same block format as the gpt-oss oracles):
  manual_passage  1-3 sentences per character, from this passage only, enough to tell the three apart (a ceiling)
  manual_prior    1-4 facts per character established before this chapter, chosen for this passage, each with a source
  oracle_quote    up to 2 sentences of the unmasked passage naming the character (every item; pure string matching)
"""
import collections
import json
import random
import re
import sys

from build_items import alias_regex
from common import DATA, REPO, SENT, clean_text, load_chapters, read_jsonl

MAN = DATA / "manual"
BATCHES = 7


def items_all():
    return {s: read_jsonl(DATA / f"items_{s}.jsonl") for s in ("test", "val", "train")}


def sample():
    principals = json.load(open(DATA / "principals.json"))
    its = [(s, it) for s, v in items_all().items() for it in v]
    by_book = collections.defaultdict(list)
    for s, it in its:
        by_book[it["book"]].append((s, it))
    books = sorted(b for b, v in by_book.items() if len(v) >= 10)
    rng = random.Random(0)
    picked = [x for b in books for x in rng.sample(by_book[b], 3)]
    notes = {}
    for s in {s for s, _ in picked}:
        for r in read_jsonl(DATA / f"reps_{s}.jsonl"):
            if r["condition"] in ("v2", "charmem", "legacy_r6000"):
                notes[(r["condition"], r["book"], r["boundary"], r["label"])] = r["text"]
    chapters = load_chapters()
    (MAN / "books").mkdir(parents=True, exist_ok=True)
    for b in books:
        with open(MAN / "books" / f"{b}.txt", "w") as f:
            for ch in chapters[b]:
                f.write(f"\n\n=== CHAPTER {ch['chapter_index']} ===\n\n{clean_text(ch['chapter_text_normalized'])}")
    tasks = []
    for s, it in picked:
        b = it["chapter_index"]
        names = {l: principals[it["book"]]["eval_names"][l][str(b)] for l in it["labels"]}
        tasks.append({"item_id": it["item_id"], "split": s, "book": it["book"], "chapter_index": b,
                      "book_file": str(MAN / "books" / f"{it['book']}.txt"),
                      "names": {names[l]: l for l in it["labels"]},
                      "answer": {names[l]: it["answer"][l] for l in it["labels"]},
                      "passage_masked": it["masked"], "passage_original": it["original"],
                      "prior_notes": {names[l]: {c: notes.get((c, it["book"], b, l)) for c in ("v2", "charmem", "legacy_r6000")}
                                      for l in it["labels"]}})
    for k in range(BATCHES):
        json.dump(tasks[k::BATCHES], open(MAN / f"tasks_{k}.json", "w"), ensure_ascii=False, indent=1)
    for t in tasks:                                    # the agents' view: one directory per passage, notes as separate files
        d = MAN / "tasks" / t["item_id"]
        (d / "notes").mkdir(parents=True, exist_ok=True)
        answer = "\n".join(f"- {n} = [CHAR {i}]" for n, i in sorted(t["answer"].items(), key=lambda kv: kv[1]))
        (d / "passage.md").write_text(
            f"# {t['item_id']}\n\nBook: {t['book']} (full text: {t['book_file']}). This passage is in CHAPTER {t['chapter_index']}; "
            f"prior material is chapters < {t['chapter_index']} only.\n\nPrincipals (names as the model sees them): "
            f"{', '.join(t['answer'])}\n\n## Answer\n\n{answer}\n\n## Masked passage (what the model sees)\n\n{t['passage_masked']}"
            f"\n\n## Original passage\n\n{t['passage_original']}\n")
        for n, cs in t["prior_notes"].items():
            for c, text in cs.items():
                if text:
                    (d / "notes" / f"{n.replace('/', '_')}__{c}.md").write_text(text)
    print(len(tasks), "passages from", len(books), "books in", BATCHES, "task files")


def quotes(it, aliases):
    out = {}
    sents = SENT.split(it["original"])
    for l in it["labels"]:
        mine = [x for x in sents if aliases[l].search(x)]
        alone = [x for x in mine if not any(aliases[o].search(x) for o in it["labels"] if o != l)]
        out[l] = "\n".join(f"- {x.strip()}" for x in (alone + [x for x in mine if x not in alone])[:2])
    return out


def collect():
    recs = collections.defaultdict(list)
    tasks = {t["item_id"]: t for k in range(BATCHES) for t in json.load(open(MAN / f"tasks_{k}.json"))}
    bad = collections.Counter()
    got = set()
    batch = {t["item_id"]: k for k in range(BATCHES) for t in json.load(open(MAN / f"tasks_{k}.json"))}
    latest = {}                                        # the assigned batch's last record wins; other files only fill gaps
    for k in range(BATCHES):
        for r in read_jsonl(MAN / f"out_{k}.jsonl"):
            if r.get("item_id") in tasks and (batch[r["item_id"]] == k or r["item_id"] not in latest):
                latest[r["item_id"]] = r
    for r in latest.values():
        t = tasks[r["item_id"]]
        label = t["names"]
        texts = {"manual_passage": {n: r["passage"][n] for n in label},
                 "manual_prior": {n: "\n".join(f"- {x['fact']}" for x in r["prior"][n]) for n in label}}
        for cond, d in texts.items():
            for n, text in d.items():
                if re.search(r"\[?CHAR\s*\d|\bID\s*\d|\bid\s*\d", text) or not text.strip():
                    bad[cond] += 1
                    continue
                recs[t["split"]].append({"condition": cond, "item_id": r["item_id"], "label": label[n], "text": text})
        got.add(r["item_id"])
    aliases = {}
    for s, v in items_all().items():
        for it in v:
            if it["book"] not in aliases:
                aliases[it["book"]] = {l: alias_regex(a) for l, a in json.load(open(REPO / "aliases" / f"{it['book']}.json"))["principals"].items()}
            for l, text in quotes(it, aliases[it["book"]]).items():
                recs[s].append({"condition": "oracle_quote", "item_id": it["item_id"], "label": l, "text": text})
    for s, v in recs.items():
        with open(MAN / f"reps_{s}_manual.jsonl", "w") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"manual oracles for {len(got)}/{len(tasks)} passages; rejected blocks: {dict(bad)}; missing: {sorted(set(tasks) - got)}")


if __name__ == "__main__":
    {"sample": sample, "collect": collect}[sys.argv[1]]()
