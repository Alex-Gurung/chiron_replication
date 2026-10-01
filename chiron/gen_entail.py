"""The Llama notes' entailment filter, run with gpt-oss on the gpt-oss Llama-prompt notes (gen_legacy_gptoss.py).

The NCP archive kept a Llama note sentence only when Llama-3.3-70B rated it fully supported (5 of 5) by the whole
chapter. Here: per (book, chapter, principal), every sentence of the 8 answers (filler removed by build_reps.gclean) is
numbered and gpt-oss rates each 1-5 as a statement about that character (in batches of 35: longer lists get miscounted), given the chapter (with its heading and
narrator). Sentences that describe another character, or that the chapter does not state, should score low.

  python3 chiron/gen_entail.py --books B... [--workers 64]
Output: outputs/legacy_gptoss_ent/<book>.jsonl, one record per (chapter, label): {q: [[sentence, rating], ...]}. Resumable.
"""
import argparse
import collections
import json
from concurrent.futures import ThreadPoolExecutor

import llm
from build_reps import gclean
from common import OUT, SENT, append_jsonl, chapter_context, clean_text, load_chapters, load_narrators, read_jsonl
from gen_chiron import QUESTIONS

ROOT = OUT / "legacy_gptoss_ent"
BATCH = 35
SCALE = ("5 = the section states it about {name} explicitly or it follows directly; 4 = strongly implied; 3 = partly "
         "supported; 2 = barely supported; 1 = not supported, contradicted, or really about someone else.")


def messages(chapter, name, sentences):
    listing = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    return [{"role": "system", "content": "You are a careful fact checker for a novel's character notes."},
            {"role": "user", "content": (
                f"Story section:\n{chapter}\n\nStatements about {name}:\n{listing}\n\nFor each numbered statement, rate how "
                f"well the story section supports it as a statement about {name}: {SCALE.format(name=name)} Return only JSON: "
                '{"ratings": [r1, r2, ...]} with one integer per statement, in order.')}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    chapters, narrators = load_chapters(), load_narrators()
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for book in args.books:
        ans = collections.defaultdict(dict)
        for r in read_jsonl(OUT / "legacy_gptoss" / f"{book}.jsonl"):
            ans[(r["chapter_index"], r["label"], r["name"])][r["q"]] = r["answer"]
        done = {(r["chapter_index"], r["label"]) for r in read_jsonl(ROOT / f"{book}.jsonl")}
        jobs += [(book, c, l, n, d) for (c, l, n), d in ans.items() if (c, l) not in done]
    print(len(jobs), "to do", flush=True)

    def one(job):
        book, c, l, name, d = job
        sents = [(q, x) for q in QUESTIONS for x in SENT.split(gclean(d.get(q, ""))) if x.strip()]
        out = {q: [] for q in QUESTIONS}
        if sents:
            ch = chapters[book][c]
            ctx = chapter_context(ch, narrators)
            text = clean_text(ch["chapter_text_normalized"])

            ratings = []
            for i in range(0, len(sents), BATCH):                 # long lists get miscounted: rate in batches
                part = [x for _, x in sents[i:i + BATCH]]

                def check(p, n=len(part)):
                    r = p["ratings"]
                    if len(r) != n or not all(isinstance(x, int) and 1 <= x <= 5 for x in r):
                        raise ValueError(f"need exactly {n} integer ratings from 1 to 5")
                    return r
                try:
                    got, _ = llm.ask(messages(f"{ctx}\n\n{text}" if ctx else text, name, part), check, max_tokens=16000)
                except ValueError as e:
                    print("FAILED", book, c, l, str(e)[:200], flush=True)
                    return
                ratings += got
            for (q, x), r in zip(sents, ratings):
                out[q].append([x, r])
        append_jsonl(ROOT / f"{book}.jsonl", {"book": book, "chapter_index": c, "label": l, "name": name, "rated": out})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
