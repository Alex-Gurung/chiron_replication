"""Plot-summary baselines with gpt-oss-120b: one summary of the whole story so far per (book, boundary b), built from
chapters < b, shared by the three characters and shown in place of per-character blocks.

  python3 chiron/gen_plot.py global --books B...   one pass over the text of chapters < b, once per target length
  python3 chiron/gen_plot.py hier --books B...     summarize every chapter (about CH_WORDS words), then combine the
                                                   chapter summaries of chapters < b, once per target length
  python3 chiron/gen_plot.py export                -> data/plot_<split>.jsonl (condition plot_<kind>_<target>)
Targets (TARGETS) are word goals given to the model; a summary outside [0.5, 1.8] x target is sent back with its
length (llm.ask retries); when the source is under 1.5 x target, the source itself is used (flagged). 'global' needs the whole preceding text in context: text beyond MAX_INPUT_WORDS keeps the
most recent part (flagged input_truncated). The text comes first and the instruction last, and a book's boundaries
run in order, so requests share their prefix in vLLM's cache. Output: outputs/plot/<kind>/<book>.jsonl. Resumable.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import llm
from common import DATA, OUT, append_jsonl, clean_text, load_chapters, read_jsonl

TARGETS = (500, 1000, 2000, 4000)
CH_WORDS = 250
MAX_INPUT_WORDS = 85000
ROOT = OUT / "plot"
SYSTEM = {"role": "system", "content": "You are a helpful and expert writing assistant."}
ASK = ("Cover the main events in order, who is involved and where things stand at the end. Name the characters. "
       "Plain prose, no headings or lists.")


def length(target):
    def f(text):
        n = len(text.split())
        if not 0.5 * target <= n <= 1.8 * target:                 # gpt-oss overshoots; the charts use actual lengths
            raise ValueError(f"the summary is {n} words; it must be about {target} words")
        return text
    return f


def boundaries(books):
    keys = {(it["book"], it["chapter_index"]) for s in ("test", "val", "train") for it in read_jsonl(DATA / f"items_{s}.jsonl")}
    return {b: sorted(c for bk, c in keys if bk == b) for b in books}


def run(kind, books, workers):
    chapters = load_chapters()
    want = boundaries(books)
    (ROOT / kind).mkdir(parents=True, exist_ok=True)

    def chapter_summaries(book):
        out = ROOT / "chapter" / f"{book}.jsonl"
        have = {r["chapter_index"]: r["text"] for r in read_jsonl(out)}
        todo = [ch for ch in chapters[book] if ch["chapter_index"] < max(want[book]) and ch["chapter_index"] not in have]

        def one(ch):
            msgs = [SYSTEM, {"role": "user", "content": f"Chapter text:\n{clean_text(ch['chapter_text_normalized'])}\n\n"
                                                         f"Summarize this chapter in about {CH_WORDS} words. {ASK}"}]
            text, meta = llm.ask(msgs, length(CH_WORDS), max_tokens=8000, as_json=False)
            append_jsonl(out, {"book": book, "chapter_index": ch["chapter_index"], "text": text, "words": len(text.split()), **meta})
            return ch["chapter_index"], text
        with ThreadPoolExecutor(16) as ex:
            have.update(dict(ex.map(one, todo)))
        return have

    def book_job(book):
        out = ROOT / kind / f"{book}.jsonl"
        done = {(r["boundary"], r["target"]) for r in read_jsonl(out)}
        summ = chapter_summaries(book) if kind == "hier" else None
        for b in want[book]:                                      # in order: each boundary extends the previous prefix
            if kind == "global":
                words = "\n\n".join(clean_text(ch["chapter_text_normalized"]) for ch in chapters[book] if ch["chapter_index"] < b).split()
                cut = len(words) > MAX_INPUT_WORDS
                body = f"Story so far:\n{' '.join(words[-MAX_INPUT_WORDS:])}"
            else:
                cut = False
                body = "Chapter-by-chapter summaries of the story so far:\n" + "\n\n".join(
                    f"Chapter {c + 1}: {summ[c]}" for c in sorted(summ) if c < b)
            todo = [t for t in TARGETS if (b, t) not in done]

            def one(t):
                if len(body.split()) <= 1.5 * t:                  # nothing to condense: the source is the summary
                    append_jsonl(out, {"book": book, "boundary": b, "target": t, "text": body.split(":\n", 1)[1],
                                       "words": len(body.split()), "input_truncated": cut, "source_shorter_than_target": True})
                    return
                verb = "Summarize the plot of the story so far" if kind == "global" else "Combine these into one plot summary of the story so far"
                msgs = [SYSTEM, {"role": "user", "content": f"{body}\n\n{verb} in about {t} words. {ASK}"}]
                try:
                    text, meta = llm.ask(msgs, length(t), max_tokens=max(12000, 3 * t), as_json=False)
                except ValueError as e:
                    print("FAILED", book, b, t, str(e)[:200], flush=True)
                    return
                append_jsonl(out, {"book": book, "boundary": b, "target": t, "text": text, "words": len(text.split()),
                                   "input_truncated": cut, **meta})
            with ThreadPoolExecutor(len(TARGETS)) as ex:
                list(ex.map(one, todo))
            print(kind, book, b, "done", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(book_job, books))


def export():
    for split in ("test", "val", "train"):
        books = {it["book"] for it in read_jsonl(DATA / f"items_{split}.jsonl")}
        recs = [{"condition": f"plot_{kind}_{r['target']}", "book": r["book"], "boundary": r["boundary"], "text": r["text"],
                 "words": r["words"]}
                for kind in ("global", "hier") for b in sorted(books) for r in read_jsonl(ROOT / kind / f"{b}.jsonl")]
        with open(DATA / f"plot_{split}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(split, len(recs), "plot summaries")


def main():
    if sys.argv[1] == "export":
        return export()
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["global", "hier"])
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    run(args.kind, args.books, args.workers)


if __name__ == "__main__":
    main()
