"""Build the shared inputs for generation.

data/snippets.jsonl   one ~300-word, sentence-aligned snippet per line, for EVERY chapter of every
                      book (chapter headers excluded, HTML stripped, NFC).
data/principals.json  book -> {labels, gen_names[label][c], eval_names[label][b]}
                      gen_names: name as known through chapter c (used when reading chapter c);
                      eval_names: name as known before boundary b (chapters < b).
"""
from common import DATA, boundaries, clean_text, display_name, fold, load_chapters, split_snippets, write_json


def main():
    chapters = load_chapters()
    bounds = boundaries()
    principals, n = {}, 0
    DATA.mkdir(parents=True, exist_ok=True)
    with open(DATA / "snippets.jsonl", "w") as f:
        for book, chs in sorted(chapters.items()):
            labels = chs[0]["main_character_labels"]
            seen, gen, ev = "", {l: [] for l in labels}, {l: {} for l in labels}
            for ch in chs:
                c = ch["chapter_index"]
                text = clean_text(ch["chapter_text_normalized"])
                for l in labels:                                   # before chapter c
                    ev[l][c] = display_name(l, seen)
                seen += "\n" + fold(text)
                for l in labels:                                   # through chapter c
                    gen[l].append(display_name(l, seen))
                for i, s in enumerate(split_snippets(text)):
                    f.write(__import__("json").dumps({"snippet_id": f"{book}__c{c:03d}__s{i:03d}", "book": book,
                                                      "chapter_index": c, "idx": i, "split": ch["split"],
                                                      "words": len(s.split()), "text": s}, ensure_ascii=False) + "\n")
                    n += 1
            principals[book] = {"labels": labels, "split": chs[0]["split"], "n_chapters": len(chs),
                                "boundaries": bounds[book], "gen_names": gen,
                                "eval_names": {l: {b: ev[l][b] for b in bounds[book]} for l in labels}}
    write_json(DATA / "principals.json", principals)
    print(f"{n} snippets, {len(principals)} books")
    for book in ("dark", "martyr", "game", "lives"):
        p = principals[book]
        for l in p["labels"]:
            print(book, repr(l), "->", sorted(set(p["gen_names"][l])))


if __name__ == "__main__":
    main()
