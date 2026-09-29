"""Character-Summary baseline with gpt-oss-120b: a rolling summary per (book, principal).

S_1 summarises chapter 0; S_{c+1} updates S_c with chapter c. The summary used at boundary b is
S_b, built from chapters < b only. Books exceed gpt-oss's 131k context, hence rolling rather than
whole-book prompts. Each update is asked to stay near LIMIT words by condensing older details; if it
still exceeds CAP, one compression call condenses it to LIMIT (the "update, then compress" rule applies
to every step, so chains are consistent). Chains run in parallel; each chain is sequential. Resumable.
Output: outputs/summary_v2/<tag>.jsonl, one record per (book, label, boundary).
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor

from common import DATA, OUT, append_jsonl, clean_text, load_chapters, read_jsonl
import llm

ASPECTS = ("Include aspects of the character like how they speak, what they look like, their personality, "
           "their goals, etc.")
LIMIT = 700
CAP = 900
EMPTY = "No information yet."
ROOT = OUT / "summary_v2"
SYSTEM = {"role": "system", "content": "You are a helpful and expert writing assistant."}


def cap(n):
    def f(text):
        if len(text.split()) > n:
            raise ValueError(f"summary is {len(text.split())} words; condense it to at most {LIMIT} words")
        return text
    return f


def messages(name, prev, chapter_text):
    if prev is None:
        body = (f"Story so far:\n{chapter_text}\n\nSummarize everything we have learned about {name} in the story "
                f"so far. {ASPECTS} Use only the story text.")
    else:
        body = (f"Summary of everything learned about {name} in the story before this chapter:\n{prev}\n\n"
                f"Next chapter:\n{chapter_text}\n\nWrite an updated summary of everything we have learned about "
                f"{name} in the story so far, including this chapter. {ASPECTS} Use only the previous summary and "
                f"the story text. Stay within {LIMIT} words: condense or drop older, less important details to make "
                "room for new information.")
    if prev is None or prev.strip() == EMPTY:
        body += f" If {name} has not appeared in the story yet, reply exactly: {EMPTY}"
    else:
        body += f" If this chapter adds nothing about {name}, return the previous summary unchanged."
    body += f" Write at most {LIMIT} words of plain prose, with no headings."
    return [SYSTEM, {"role": "user", "content": body}]


def compress_messages(name, text):
    return [SYSTEM, {"role": "user", "content": (
        f"Here is a summary of everything learned about {name} in a story so far:\n{text}\n\n"
        f"Condense it to at most {LIMIT} words of plain prose, with no headings. Keep the most important and most "
        "distinctive information about the character; do not add anything new.")}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--upto", type=int, default=0, help="stop each chain at this boundary (smoke tests)")
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    principals = json.load(open(DATA / "principals.json"))
    chapters = load_chapters()
    out = ROOT / f"{args.tag}.jsonl"
    have = {}                                          # resume from every earlier run's file, not only this tag's
    for f in sorted(ROOT.glob("*.jsonl")) if ROOT.exists() else []:
        if not f.name.endswith(".errors.jsonl"):
            for r in read_jsonl(f):
                have[(r["book"], r["label"], r["boundary"])] = r
    fails = []

    def chain(book, label):
        chs = chapters[book]
        last_boundary = min(max(principals[book]["boundaries"]), args.upto or 10**9)
        prev = None
        for c in range(last_boundary):
            b = c + 1
            if (book, label, b) in have:
                prev = have[(book, label, b)]["summary"]
                continue
            name = principals[book]["gen_names"][label][c]
            text = clean_text(chs[c]["chapter_text_normalized"])
            try:
                summ, meta = llm.ask(messages(name, prev, text), cap(4000), max_tokens=12000, as_json=False)
                compressed = None
                if len(summ.split()) > CAP:
                    compressed = len(summ.split())
                    summ, meta2 = llm.ask(compress_messages(name, summ), cap(CAP), max_tokens=8000, as_json=False)
                    meta = {"update": meta, "compress": meta2}
            except Exception as e:
                fails.append((book, label, b))
                append_jsonl(out.with_suffix(".errors.jsonl"), {"book": book, "label": label, "boundary": b, "error": str(e)[:2000]})
                return
            append_jsonl(out, {"book": book, "label": label, "boundary": b, "name": name, "summary": summ,
                               "words": len(summ.split()), "compressed_from": compressed, "meta": meta})
            prev = summ
        print(f"DONE {book} {label}", flush=True)

    jobs = [(b, l) for b in args.books for l in principals[b]["labels"]]
    with ThreadPoolExecutor(len(jobs)) as pool:
        list(pool.map(lambda j: chain(*j), jobs))
    print(f"{args.tag}: chains={len(jobs)} failed={len(fails)}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
