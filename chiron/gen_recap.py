"""Where the story stands: a short gpt-oss recap of the last words before the passage's chapter, shared by all three
characters (shown once, like book text: eval condition "<sheet>&plot_recap<W>").

charmem plus the raw last 500-1,000 words of the book beats every sheet at ~5k tokens; this asks whether a recap of
the last 2,000 words can carry the same scene context in fewer tokens.

  python3 chiron/gen_recap.py --books B... [--words 300] [--source-words 2000]
Output: data/recap_<split>.jsonl {condition: plot_recap<W>, book, boundary, text} (read by eval_mcp.load_plots). Resumable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

import llm
from common import DATA, append_jsonl, clean_text, load_chapters, read_jsonl

SYSTEM = {"role": "system", "content": "You are an expert story editor."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--words", type=int, default=300)
    ap.add_argument("--source-words", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=128)
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    chapters = load_chapters()
    cond = f"plot_recap{args.words}"
    jobs = []
    for split in ("test", "val", "train"):
        done = {(r["book"], r["boundary"]) for r in read_jsonl(DATA / f"recap_{split}.jsonl") if r["condition"] == cond}
        keys = sorted({(it["book"], it["chapter_index"]) for it in read_jsonl(DATA / f"items_{split}.jsonl") if it["book"] in args.books})
        jobs += [(split, b, bd) for b, bd in keys if (b, bd) not in done]
    print(len(jobs), "to do", flush=True)

    def one(job):
        split, b, bd = job
        text = " ".join("\n\n".join(clean_text(ch["chapter_text_normalized"]) for ch in chapters[b] if ch["chapter_index"] < bd).split(" ")[-args.source_words:])
        msgs = [SYSTEM, {"role": "user", "content": (
            f"The last {args.source_words:,} words of a novel before chapter {bd + 1}:\n\n{text}\n\nIn about {args.words} words, recap where the "
            "story stands at the end of this excerpt: who is there, where, doing what, what was just said, decided or revealed, "
            "and how each person is feeling. Name people instead of using pronouns, keep events in order, and use only the excerpt.")}]
        try:
            out, _ = llm.ask(msgs, lambda x: x, max_tokens=12000, as_json=False)
        except ValueError as e:
            print("FAILED", b, bd, str(e)[:200], flush=True)
            return
        append_jsonl(DATA / f"recap_{split}.jsonl", {"condition": cond, "book": b, "boundary": bd, "text": out})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
