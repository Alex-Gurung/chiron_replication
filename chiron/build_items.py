"""Build masked-character-prediction items from NCP sections.

Needs data/aliases/<book>.json (hand-checked): {"principals": {label: [alias, ...]}, "ambiguous": [str, ...]}.
An item is an NCP section (v2 cohort `next_chapter`, chapter heading removed) in which every principal
is named, and each principal is named in at least MIN_PRIOR chapters before the section's chapter.
Sections containing an ambiguous string (e.g. a surname two principals share) are dropped.
Every alias mention becomes [CHAR i] (possessive kept outside); ids are shuffled per item.
Output: data/items_<split>.jsonl and a per-book count table on stdout.
"""
import argparse
import collections
import json
import random
import re

from common import COHORTS, DATA, REPO, SENT, clean_text, load_chapters, nfc

MIN_PRIOR = 2
HEADING = re.compile(r"^\s*((CHAPTER|Chapter|PART|Part|PROLOGUE|Prologue|EPILOGUE|Epilogue|INTERLUDE|BOOK)\b[^\n]*|\d+|[IVXLC]+)\s*\n", re.M)


def alias_regex(aliases):
    alts = sorted({v for a in aliases for v in (nfc(a), nfc(a).upper())}, key=len, reverse=True)   # + ALL-CAPS forms
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


def units(split, mode):
    """Yield (book, chapter, unit_key, chunk_index, text) for one passage type:
    section  one NCP section (the default);
    window   consecutive sections of a chapter grouped until ~900 words (at most 3), non-overlapping;
    short    the tightest run of whole sentences (SHORT_MIN to SHORT_MAX words) naming every principal."""
    if mode == "section":
        for line in open(COHORTS / f"{split}_examples.jsonl"):
            rec = json.loads(line)
            yield rec["story_id"], rec["chapter_index"], f"k{rec['chunk_index']:02d}", rec["chunk_index"], strip_heading(rec)
        return
    by = collections.defaultdict(list)
    for line in open(COHORTS / f"{split}_examples.jsonl"):
        rec = json.loads(line)
        by[(rec["story_id"], rec["chapter_index"])].append((rec["chunk_index"], strip_heading(rec) or ""))
    for (book, b), secs in by.items():
        secs = [t for _, t in sorted(secs)]
        if mode == "window":
            i, k = 0, 0
            while i < len(secs):
                win = []
                while i < len(secs) and len(win) < 3 and (not win or len(" ".join(win).split()) < 900):
                    win.append(secs[i])
                    i += 1
                yield book, b, f"w{k:02d}", None, "\n\n".join(win)
                k += 1
        else:
            yield book, b, "chapter", None, "\n\n".join(secs)       # split into spans by the caller


SHORT_MIN = 20
SHORT_MAX = 150


def short_spans(text, regs):
    """Minimal runs of sentences containing every principal, >= SHORT_MIN words, non-overlapping, left to right."""
    sents = [x for x in SENT.split(text) if x.strip()]
    has = [{l for l, rx in regs.items() if rx.search(x)} for x in sents]
    out, i = [], 0
    while i < len(sents):
        seen, j = set(), i
        while j < len(sents) and seen != set(regs):
            seen |= has[j]
            j += 1
        if seen != set(regs):
            break
        lo = i                                        # shrink from the left while every principal stays named
        while lo < j - 1 and set().union(*has[lo + 1:j]) == set(regs):
            lo += 1
        while len(" ".join(sents[lo:j]).split()) < SHORT_MIN and lo > i:
            lo -= 1
        span = " ".join(sents[lo:j])
        if SHORT_MIN <= len(span.split()) <= SHORT_MAX:
            out.append(span)
        i = j
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--two", action="store_true", help="sections naming exactly two principals (2-way items)")
    ap.add_argument("--mode", default="section", choices=["section", "window", "short"])
    ap.add_argument("--min-mentions", type=int, default=1, help="each candidate principal named at least this often")
    args = ap.parse_args()
    chapters = load_chapters()
    items, table, leftovers = [], {}, []
    stream = units(args.split, args.mode)
    if args.mode == "short":
        def expand(it):
            regs_cache = {}
            for book, b, key, chunk, text in it:
                if book not in regs_cache:
                    regs_cache[book] = {l: alias_regex(a) for l, a in json.load(open(REPO / "aliases" / f"{book}.json"))["principals"].items()}
                for k, span in enumerate(short_spans(text, regs_cache[book])):
                    yield book, b, f"t{k:02d}", None, span
        stream = expand(stream)
    for book, b, key, chunk, text in stream:
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
            table[book] = {"units": 0, "no_heading_text": 0, "not_all_named": 0, "ambiguous": 0, "too_early": 0,
                           "leftover": 0, "items": 0}
        t = table[book]
        t["units"] += 1
        if not text:
            t["no_heading_text"] += 1
            continue
        if protect and protect.search(text):           # another character's name contains a principal alias
            t["ambiguous"] += 1
            continue
        labels = [l for l, rx in regs.items() if len(rx.findall(text)) >= args.min_mentions]
        if len(labels) != (2 if args.two else len(regs)) or (args.two and sum(bool(rx.search(text)) for rx in regs.values()) != 2):
            t["not_all_named"] += 1
            continue
        if any(len({c for c in named_in[l] if c < b}) < MIN_PRIOR for l in labels):
            t["too_early"] += 1
            continue
        ids = list(range(len(labels)))
        item_id = f"{book}__c{b:03d}__{key}" + ("__two" if args.two else "")
        random.Random(f"{args.seed}:{item_id}").shuffle(ids)   # per item, so dropping one never moves another's masks
        order = dict(zip(labels, ids))
        masked, used = mask(text, {l: regs[l] for l in labels}, order)
        if amb and amb.search(masked):                # checked after masking: "Liska Radost" is gone, "Dobrawa Radost" is not
            t["ambiguous"] += 1
            continue
        left = [w for w in leak.findall(masked) if not w.islower()]   # name tokens left in any capitalisation ("LISKA")
        if left:
            t["leftover"] += 1
            leftovers.append((book, left[:3]))
            continue
        items.append({"item_id": item_id, "book": book, "chapter_index": b, "chunk_index": chunk, "labels": labels,
                      "answer": order, "mentions": {l: used.count(l) for l in labels}, "masked": masked, "original": text})
        t["items"] += 1
    suffix = ("_two" if args.two else "") + ("" if args.mode == "section" else f"_{args.mode}")
    with open(DATA / f"items_{args.split}{suffix}.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    for book, t in table.items():
        print(book, t)
    print("total items", len(items))
    for book, left in leftovers[:15]:
        print("  leftover", book, left)


if __name__ == "__main__":
    main()
