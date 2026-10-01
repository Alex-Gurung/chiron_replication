"""The Llama CHIRON notes redone with gpt-oss-120b: CHIRON's 8 questions answered once per (chapter, principal),
with the whole chapter as the story section, as short prose sentences, unfiltered. Same layout as the NCP archive's
Llama-3.3-70B notes, so the two differ only in the model. (The gpt-oss CHIRON claims in gen_chiron.py instead answer
per 300-word snippet, split answers into single claims and keep only entailed ones.)

  python3 chiron/gen_chapnotes.py --books B... [--workers 64] [--headings]
With --headings each chapter is prefixed with its heading and, when first person, who "I" is (-> outputs/chapnotes_h);
--long asks for thorough answers (-> outputs/chapnotes[_h]_long).
Covers every chapter before the last boundary a book's passages need. Output: outputs/chapnotes/<book>.jsonl, one
record per (chapter, label) with an answer per question ("" when the chapter says nothing). Resumable.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import llm
from common import DATA, OUT, append_jsonl, chapter_context, clean_text, load_chapters, load_narrators, read_jsonl
from gen_chiron import QUESTIONS, ROLE

ROOT = OUT / "chapnotes"                         # --headings: OUT / "chapnotes_h"


THOROUGH = ("Answer each question thoroughly, with every relevant detail from this section: what the character does, says "
            "and thinks, who they are with and how they relate to each other character, where they are, and what changes. Use "
            "as many short, simple sentences as the section supports (often 3 to 8 per question), each naming the character. ")


def messages(chapter, name, long=False):
    qs = "\n".join(f"- {k}: {q}" for k, (_, q) in QUESTIONS.items())
    schema = ", ".join(f'"{k}": "..."' for k in QUESTIONS)
    how = (THOROUGH.replace("the character", name) + "Use only this story section. " if long else
           f"Answer each question about {name} based only on this story section, in a few short, simple sentences that name "
           "the character. ")                              # the original prompt, unchanged
    return [{"role": "system", "content": ROLE},
            {"role": "user", "content": (
                f"Story Section:\n{chapter}\n\nCharacter: {name}\n\n{how}If the section says nothing for a question, answer "
                f"with an empty string.\n\nQuestions:\n{qs}\n\nReturn only JSON: {{{schema}}}")}]


def check(p):
    if set(p) != set(QUESTIONS):
        raise ValueError(f"expected a JSON object with exactly the keys {sorted(QUESTIONS)} and string values")
    p = {k: " ".join(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else v for k, v in p.items()}   # lists of sentences are fine
    if not all(isinstance(v, str) for v in p.values()):
        raise ValueError("every answer must be a string")
    return {k: v.strip() for k, v in p.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--headings", action="store_true", help="prefix each chapter with its heading and who narrates it")
    ap.add_argument("--long", action="store_true", help="ask for thorough answers (-> chapnotes..._long)")
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    global ROOT
    ROOT = OUT / ("chapnotes" + ("_h" if args.headings else "") + ("_long" if args.long else ""))
    narrators = load_narrators() if args.headings else {}
    principals = json.load(open(DATA / "principals.json"))
    chapters = load_chapters()
    last = {}
    for s in ("test", "val", "train"):
        for it in read_jsonl(DATA / f"items_{s}.jsonl"):
            last[it["book"]] = max(last.get(it["book"], 0), it["chapter_index"])
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for book in args.books:
        done = {(r["chapter_index"], r["label"]) for r in read_jsonl(ROOT / f"{book}.jsonl")}
        jobs += [(book, ch, l) for ch in chapters[book] if ch["chapter_index"] < last.get(book, 0)
                 for l in principals[book]["labels"] if (ch["chapter_index"], l) not in done]
    print(len(jobs), "to do", flush=True)

    def one(job):
        book, ch, l = job
        name = principals[book]["gen_names"][l][ch["chapter_index"]]
        try:
            ctx = chapter_context(ch, narrators) if args.headings else ""
            text = clean_text(ch["chapter_text_normalized"])
            ans, meta = llm.ask(messages(f"{ctx}\n\n{text}" if ctx else text, name, args.long), check, max_tokens=16000)
        except ValueError as e:
            print("FAILED", book, ch["chapter_index"], l, str(e)[:200], flush=True)
            return
        append_jsonl(ROOT / f"{book}.jsonl", {"book": book, "chapter_index": ch["chapter_index"], "label": l, "name": name,
                                               "answers": ans, **meta})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
