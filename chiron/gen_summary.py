"""Character-Summary baseline with gpt-oss-120b: a rolling summary per (book, principal).

S_1 summarises chapter 0; S_{c+1} updates S_c with chapter c. The summary used at boundary b is
S_b, built from chapters < b only. Books exceed gpt-oss's 131k context, hence rolling rather than
whole-book prompts. Chains run in parallel; each chain is sequential. Resumable.
Output: outputs/summary/<tag>.jsonl, one record per (book, label, boundary).
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
CAP = 1200                       # prompt asks for 700; accumulated summaries run ~900-1,050
EMPTY = "No information yet."


def check(text):
    if len(text.split()) > CAP:
        raise ValueError(f"summary is {len(text.split())} words; rewrite it to at most {LIMIT} words by compressing older details")
    return text


def messages(name, prev, chapter_text):
    if prev is None:
        body = (f"Story so far:\n{chapter_text}\n\nSummarize everything we have learned about {name} in the story "
                f"so far. {ASPECTS} Use only the story text.")
    else:
        body = (f"Summary of everything learned about {name} in the story before this chapter:\n{prev}\n\n"
                f"Next chapter:\n{chapter_text}\n\nWrite an updated summary of everything we have learned about "
                f"{name} in the story so far, including this chapter. {ASPECTS} Keep earlier information unless "
                "the story has changed it. Use only the previous summary and the story text.")
    if prev is None or prev.strip() == EMPTY:
        body += f" If {name} has not appeared in the story yet, reply exactly: {EMPTY}"
    else:
        body += f" If this chapter adds nothing about {name}, return the previous summary unchanged."
    body += f" Write at most {LIMIT} words of plain prose, with no headings."
    return [{"role": "system", "content": "You are a helpful and expert writing assistant."},
            {"role": "user", "content": body}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--upto", type=int, default=0, help="stop each chain at this boundary (smoke tests)")
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    principals = json.load(open(DATA / "principals.json"))
    chapters = load_chapters()
    out = OUT / "summary" / f"{args.tag}.jsonl"
    have = {(r["book"], r["label"], r["boundary"]): r for r in read_jsonl(out)}
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
                summ, meta = llm.ask(messages(name, prev, text), check, max_tokens=12000, as_json=False)
            except Exception as e:
                fails.append((book, label, b))
                append_jsonl(out.with_suffix(".errors.jsonl"), {"book": book, "label": label, "boundary": b, "error": str(e)[:2000]})
                return
            append_jsonl(out, {"book": book, "label": label, "boundary": b, "name": name,
                               "summary": summ, "words": len(summ.split()), "meta": meta})
            prev = summ
        print(f"DONE {book} {label}", flush=True)

    jobs = [(b, l) for b in args.books for l in principals[b]["labels"]]
    with ThreadPoolExecutor(len(jobs)) as pool:
        list(pool.map(lambda j: chain(*j), jobs))
    print(f"{args.tag}: chains={len(jobs)} failed={len(fails)}", flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
