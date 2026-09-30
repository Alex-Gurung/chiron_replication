"""The Llama CHIRON notes redone with gpt-oss-120b: CHIRON's 8 questions answered once per (chapter, principal),
with the whole chapter as the story section, as short prose sentences, unfiltered. Same layout as the NCP archive's
Llama-3.3-70B notes, so the two differ only in the model. (The gpt-oss CHIRON claims in gen_chiron.py instead answer
per 300-word snippet, split answers into single claims and keep only entailed ones.)

  python3 chiron/gen_chapnotes.py --books B... [--workers 64]
Covers every chapter before the last boundary a book's passages need. Output: outputs/chapnotes/<book>.jsonl, one
record per (chapter, label) with an answer per question ("" when the chapter says nothing). Resumable.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import llm
from common import DATA, OUT, append_jsonl, clean_text, load_chapters, read_jsonl
from gen_chiron import QUESTIONS, ROLE

ROOT = OUT / "chapnotes"


def messages(chapter, name):
    qs = "\n".join(f"- {k}: {q}" for k, (_, q) in QUESTIONS.items())
    schema = ", ".join(f'"{k}": "..."' for k in QUESTIONS)
    return [{"role": "system", "content": ROLE},
            {"role": "user", "content": (
                f"Story Section:\n{chapter}\n\nCharacter: {name}\n\n"
                f"Answer each question about {name} based only on this story section, in a few short, simple sentences "
                f"that name the character. If the section says nothing for a question, answer with an empty string.\n\n"
                f"Questions:\n{qs}\n\nReturn only JSON: {{{schema}}}")}]


def check(p):
    if set(p) != set(QUESTIONS) or not all(isinstance(v, str) for v in p.values()):
        raise ValueError(f"expected a JSON object with exactly the keys {sorted(QUESTIONS)} and string values")
    return {k: v.strip() for k, v in p.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
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
            ans, meta = llm.ask(messages(clean_text(ch["chapter_text_normalized"]), name), check, max_tokens=12000)
        except ValueError as e:
            print("FAILED", book, ch["chapter_index"], l, str(e)[:200], flush=True)
            return
        append_jsonl(ROOT / f"{book}.jsonl", {"book": book, "chapter_index": ch["chapter_index"], "label": l, "name": name,
                                               "answers": ans, **meta})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
