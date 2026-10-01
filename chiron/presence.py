"""Which principals each chapter actually involves: a principal is present in a chapter when one of its aliases occurs
in the chapter text or it narrates the chapter in the first person (common.load_narrators).

  python3 chiron/presence.py      prints, per note source, the share of (character, chapter) notes from chapters
                                  without the character (gpt-oss tends to describe whoever is there instead)
present() is used by gen_sheet.py --present to drop those notes.
"""
import json
import re

from build_items import alias_regex
from common import REPO, clean_text, fold, load_chapters, load_narrators


def present():
    """(book, chapter_index) -> set of principal labels present."""
    chapters, narr = load_chapters(), load_narrators()
    out = {}
    for b, chs in chapters.items():
        p = REPO / "aliases" / f"{b}.json"
        if not p.exists():
            continue
        al = json.load(open(p))["principals"]
        rx = {l: alias_regex(a) for l, a in al.items()}
        folded = {l: {fold(x) for x in a} | {fold(x.split()[0]) for x in a} for l, a in al.items()}
        for ch in chs:
            text = clean_text(ch["chapter_text_normalized"])
            n = narr.get((b, ch["chapter_index"]))
            out[(b, ch["chapter_index"])] = {l for l in al if rx[l].search(text) or (n and fold(n) in folded[l])}
    return out


if __name__ == "__main__":
    import collections
    from common import DATA, read_jsonl
    from gen_sheet import load_notes
    pres = present()
    items = [it for s in ("test", "val", "train") for it in read_jsonl(DATA / f"items_{s}.jsonl")]
    keys = {(it["book"], it["chapter_index"], l) for it in items for l in it["labels"]}
    for src in ("chapnotes", "chapnotes_h_long", "legacy_gptoss", "legacy"):
        notes = load_notes(src, keys)
        seen, words = set(), collections.Counter()
        for (b, bd, l), chs in notes.items():
            for c, t in chs:
                if (b, l, c) in seen:
                    continue
                seen.add((b, l, c))
                words["absent" if l not in pres.get((b, c), set()) else "present"] += len(t.split())
        n_abs = sum(1 for b, l, c in seen if l not in pres.get((b, c), set()))
        print(f"{src:18s} {len(seen)} (character, chapter) notes; {100 * n_abs / len(seen):.1f}% from chapters without the "
              f"character, {100 * words['absent'] / sum(words.values()):.1f}% of the words")
