"""Style vs content for the ~500-word summaries: gpt-oss rewrites a summary in another style, keeping its facts.

The Llama summary (legacy) is prose that names the character in full sentences; gpt-oss's summary of the same kind of
notes (sheet_legsum_x_500) is telegraphic bullets under bold headings. Rewriting both into both styles (a 2 x 2 of
content source x style, every cell rewritten by gpt-oss once) separates the two.

  python3 chiron/gen_restyle.py --source legacy|<sheet variant> --style prose|bullets --variant NAME --books B...
Output: outputs/sheets/<variant>/<book>.jsonl (build_reps: sheet_<variant>, add the variant to FLAT). Resumable.
"""
import argparse
import pickle
from concurrent.futures import ThreadPoolExecutor

import llm
from build_reps import LEGACY, LEGACY_LABEL
from common import DATA, OUT, append_jsonl, read_jsonl

STYLES = {
    "prose": "Rewrite it as a few paragraphs of plain prose in full sentences that name the character, with no headings and no "
             "bullet points.",
    "bullets": "Rewrite it as compact bullet points under short bold headings, in clipped fragments rather than full sentences.",
}


def sources(src, books):
    if src == "legacy":
        out = {}
        for s in ("test", "val", "train"):
            comp = pickle.load(open(LEGACY / f"{s}_long_story_character_sheet_summaryllama70B_0.5max.pkl", "rb"))
            for it in read_jsonl(DATA / f"items_{s}.jsonl"):
                for l in it["labels"]:
                    t = comp.get(f"{it['book']}_{LEGACY_LABEL.get(l, l)}_{it['chapter_index']}")
                    if t and it["book"] in books:
                        out[(it["book"], it["chapter_index"], l)] = t
        return out
    return {(r["book"], r["boundary"], r["label"]): r["text"] for b in books for r in read_jsonl(OUT / "sheets" / src / f"{b}.jsonl")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--style", required=True, choices=list(STYLES))
    ap.add_argument("--variant", required=True)
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=128)
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    root = OUT / "sheets" / args.variant
    root.mkdir(parents=True, exist_ok=True)
    src = sources(args.source, set(args.books))
    done = {(r["book"], r["boundary"], r["label"]) for b in args.books for r in read_jsonl(root / f"{b}.jsonl")}
    todo = sorted(k for k in src if k not in done)
    print(len(todo), "to do", flush=True)

    def one(k):
        text = src[k]
        msgs = [{"role": "system", "content": "You are an expert story editor."},
                {"role": "user", "content": (
                    f"Below is a character sheet about {k[2]}.\n---\n{text}\n---\n{STYLES[args.style]} Keep every fact, name, quote and "
                    f"snippet reference, add nothing that is not in the sheet, and keep it about the same length "
                    f"({len(text.split())} words). Return only the rewritten sheet.")}]
        try:
            out, meta = llm.ask(msgs, lambda x: x, max_tokens=12000, as_json=False)
        except ValueError as e:
            print("FAILED", *k, str(e)[:200], flush=True)
            return
        append_jsonl(root / f"{k[0]}.jsonl", {"book": k[0], "boundary": k[1], "label": k[2], "text": out, "words": len(out.split()),
                                              "source_words": len(text.split()), **meta})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
