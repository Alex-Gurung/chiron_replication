"""Build masked-character-prediction items from NCP sections.

Needs data/aliases/<book>.json (hand-checked): {"principals": {label: [alias, ...]}, "ambiguous": [str, ...]}.
An item is an NCP section (v2 cohort `next_chapter`, chapter heading removed) in which every principal
is named, and each principal is named in at least MIN_PRIOR chapters before the section's chapter.
Sections containing an ambiguous string (e.g. a surname two principals share) are dropped.
Every alias mention becomes [CHAR i] (possessive kept outside); ids are shuffled per item.
Output: data/items_<split>.jsonl and a per-book count table on stdout.
"""
import argparse
import json
import random
import re

from common import COHORTS, DATA, REPO, clean_text, load_chapters, nfc

MIN_PRIOR = 2
HEADING = re.compile(r"^\s*((CHAPTER|Chapter|PART|Part|PROLOGUE|Prologue|EPILOGUE|Epilogue|INTERLUDE|BOOK)\b[^\n]*|\d+|[IVXLC]+)\s*\n", re.M)


def alias_regex(aliases):
    alts = sorted({nfc(a) for a in aliases}, key=len, reverse=True)
    return re.compile(r"(?<![\w’'])(" + "|".join(re.escape(a) for a in alts) + r")(?![\w])")


def strip_heading(rec):
    text, head = rec["next_chapter"], rec.get("next_chapter_header") or ""
    if head.strip() and text.strip() == head.strip():
        return None                                   # header IS the section (v12u mercy defect)
    if head.strip() and text.startswith(head):
        text = text[len(head):]
    while True:
        m = HEADING.match(text)
        if not m or m.start() != 0 or len(m.group(0)) > 120:
            break
        text = text[m.end():]
    return clean_text(text).strip()


def mask(text, regs, order):
    """order: label -> mask id. Longest alias across ALL principals wins at each position."""
    spans = []
    for label, rx in regs.items():
        for m in rx.finditer(text):
            spans.append((m.start(), m.end(), label))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    out, pos, used = [], 0, []
    for s, e, label in spans:
        if s < pos:
            continue
        out.append(text[pos:s])
        out.append(f"[CHAR {order[label]}]")
        used.append(label)
        pos = e
    out.append(text[pos:])
    return "".join(out), used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    chapters = load_chapters()
    items, table, leftovers = [], {}, []
    for line in open(COHORTS / f"{args.split}_examples.jsonl"):
        rec = json.loads(line)
        book, b = rec["story_id"], rec["chapter_index"]
        if book not in table:
            al = json.load(open(REPO / "aliases" / f"{book}.json"))
            regs = {l: alias_regex(a) for l, a in al["principals"].items()}
            amb = alias_regex(al["ambiguous"]) if al.get("ambiguous") else None
            protect = alias_regex(al["drop_if_present"]) if al.get("drop_if_present") else None
            named_in = {l: {ch["chapter_index"] for ch in chapters[book]
                            if rx.search(clean_text(ch["chapter_text_normalized"]))} for l, rx in regs.items()}
            toks = {w for a in al["principals"].values() for x in a for w in re.findall(r"[^\W\d_][\w’'-]*", nfc(x))
                    if len(w) >= 3 and w[0].isupper() and w not in {"The", "Miss", "Mrs", "Ms.", "Mr.", "Mrs.", "Pan", "Pani", "Panie"}}
            leak = re.compile(r"(?<!\w)(" + "|".join(sorted(map(re.escape, toks), key=len, reverse=True)) + r")(?!\w)", re.I)
            table[book] = {"sections": 0, "no_heading_text": 0, "not_all_named": 0, "ambiguous": 0, "too_early": 0,
                           "leftover": 0, "items": 0}
        t = table[book]
        t["sections"] += 1
        text = strip_heading(rec)
        if not text:
            t["no_heading_text"] += 1
            continue
        if protect and protect.search(text):           # another character's name contains a principal alias
            t["ambiguous"] += 1
            continue
        if not all(rx.search(text) for rx in regs.values()):
            t["not_all_named"] += 1
            continue
        if any(len({c for c in named_in[l] if c < b}) < MIN_PRIOR for l in regs):
            t["too_early"] += 1
            continue
        labels = list(regs)
        ids = list(range(len(labels)))
        item_id = f"{book}__c{b:03d}__k{rec['chunk_index']:02d}"
        random.Random(f"{args.seed}:{item_id}").shuffle(ids)   # per item, so dropping one never moves another's masks
        order = dict(zip(labels, ids))
        masked, used = mask(text, regs, order)
        if amb and amb.search(masked):                # checked after masking: "Liska Radost" is gone, "Dobrawa Radost" is not
            t["ambiguous"] += 1
            continue
        left = [w for w in leak.findall(masked) if not w.islower()]   # name tokens left in any capitalisation ("LISKA")
        if left:
            t["leftover"] += 1
            leftovers.append((book, left[:3]))
            continue
        items.append({"item_id": item_id, "book": book, "chapter_index": b,
                      "chunk_index": rec["chunk_index"], "labels": labels, "answer": order,
                      "mentions": {l: used.count(l) for l in labels}, "masked": masked, "original": text})
        t["items"] += 1
    with open(DATA / f"items_{args.split}.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    for book, t in table.items():
        print(book, t)
    print("total items", len(items))
    for book, left in leftovers[:15]:
        print("  leftover", book, left)


if __name__ == "__main__":
    main()
