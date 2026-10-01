"""Consensus sheets: merge independent samples of the same sheet, keeping what at least two of them say.

A resample of charmem's exact synthesis scores 3.7 points below the published sheets (68.2 vs 71.9 on the 27B), so a
single sample carries a lot of generation noise. This keeps the notes the samples agree on.

  python3 chiron/gen_merge.py --variant NAME --from V1 V2 V3 --books B... [--words 900]
Reads outputs/sheets/<Vi>/<book>.jsonl; one gpt-oss call per (book, boundary, label) present in every Vi.
Output: outputs/sheets/<variant>/<book>.jsonl. Resumable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import llm
from common import OUT, append_jsonl, read_jsonl

SYSTEM = {"role": "system", "content": "You are an expert story editor who keeps the character bible for a novel."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--from", dest="src", nargs="+", required=True)
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--words", type=int, default=900)
    ap.add_argument("--workers", type=int, default=128)
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    root = OUT / "sheets" / args.variant
    root.mkdir(parents=True, exist_ok=True)
    drafts = [{(r["book"], r["boundary"], r["label"]): r["text"] for b in args.books for r in read_jsonl(OUT / "sheets" / v / f"{b}.jsonl")}
              for v in args.src]
    done = {(r["book"], r["boundary"], r["label"]) for b in args.books for r in read_jsonl(root / f"{b}.jsonl")}
    todo = sorted(k for k in set.intersection(*(set(d) for d in drafts)) if k not in done)
    print(len(todo), "to do", flush=True)

    def one(k):
        body = "\n\n".join(f"=== Draft {i + 1} ===\n{d[k]}" for i, d in enumerate(drafts))
        msgs = [SYSTEM, {"role": "user", "content": (
            f"{body}\n\nThese are {len(drafts)} drafts of {k[2]}'s character sheet as of chapter {k[1]}, written independently "
            "from the same notes. Write the final sheet in the same format (the same ## section headings; one bullet per note, "
            "ending with its chapter citation). Keep the notes that at least two drafts include, in any wording; merge "
            "duplicates into one richer note; leave out anything only one draft says. Use no more than "
            f"{args.words} words. Return only the sheet.")}]
        try:
            text, meta = llm.ask(msgs, lambda x: x, max_tokens=24000, as_json=False)
        except ValueError as e:
            print("FAILED", *k, str(e)[:200], flush=True)
            return
        append_jsonl(root / f"{k[0]}.jsonl", {"book": k[0], "boundary": k[1], "label": k[2], "text": text, "words": len(text.split()), **meta})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
