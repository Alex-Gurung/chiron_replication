"""The CHIRON paper's Character-Summary baseline: the whole story so far -> one summary of the character, then the same
entailment filter as the CHIRON statements.

The paper (Gurung & Lapata 2024) prompted Mistral 7B Instruct v0.2 "with the entire story so-far and asking for a
summary of the given character", and filtered the summary with its entailment pipeline. The prompt is the generation
module's (github.com/Alex-Gurung/CHIRON, chiron_generation_module_utils.py / chiron_utils.py, copied below) with its
"summarize_story" question, decoded greedily. Here gpt-oss-120b writes it; the paper's stories were ~5.5k words, ours
are novels, so when the story so far exceeds MAX_WORDS only its last MAX_WORDS words are given (gpt-oss's 131k
context; ~20% of boundaries). Filter: spaCy sentences, each rated by gen_legacy_exact's archive entailment step against
the same text, 5/5 kept.

  python3 chiron/gen_csum.py --books B... [--workers 32]
Output: outputs/sheets/csum/<book>.jsonl (unfiltered) and outputs/sheets/csum_f/<book>.jsonl (filtered), one record
per (book, boundary, label). Resumable.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import llm
from build_reps import GABOUT, GMETA, GNEG
from common import DATA, GEN, OUT, append_jsonl, clean_text, gen, load_chapters, read_jsonl
from gen_legacy_exact import rate, sentences

MAX_WORDS = 88000
ROLE = ("You are a helpful and expert writing assistant. You will be given a section of a story or screenplay. Please answer "
        "the following questions about the character learned in this story section, and respond in short paragraph form.")
QUESTION = ("Summarize everything we have learned about this character across these snippets. Include aspects of the character "
            "like how they speak, what they look like, their personality, their goals, etc.")


def messages(story, character):
    story = re.sub(r"(\n(\ )?)+", "\n", re.sub(r"\ +", " ", story.strip()))
    return [{"role": "system", "content": ROLE},
            {"role": "user", "content": (
                f"Story Section:\n{story}\n\nPlease answer the following questions about {character} with short, succinct "
                f"sentences based on the given story section.\n\nQuestion: {QUESTION} Respond in paragraph form with short, "
                "simple sentences with no dependent clauses or transition words.\nAnswer:")}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    assert llm.server_up(), "generation server not reachable"
    principals = json.load(open(DATA / "principals.json"))
    chapters = load_chapters()
    raw, filt = OUT / "sheets" / gen("csum"), OUT / "sheets" / (gen("csum") + "_f")
    raw.mkdir(parents=True, exist_ok=True)
    filt.mkdir(parents=True, exist_ok=True)
    keys = sorted({(it["book"], it["chapter_index"], l) for s in ("test", "val", "train") for it in read_jsonl(DATA / f"items_{s}.jsonl")
                   if it["book"] in args.books for l in it["labels"]})
    done = {(r["book"], r["boundary"], r["label"]) for b in args.books for r in read_jsonl(filt / f"{b}.jsonl")}
    todo = [k for k in keys if k not in done]
    print(len(todo), "to do", flush=True)
    calls = ThreadPoolExecutor(4 * args.workers)

    def one(k):
        b, bd, l = k
        story = " ".join("\n\n".join(clean_text(c["chapter_text_normalized"]) for c in chapters[b] if c["chapter_index"] < bd).split(" ")[-MAX_WORDS:])
        name = principals[b]["gen_names"][l][bd - 1]
        try:
            text, meta = llm.ask(messages(story, name), lambda x: x, max_tokens=8000 if GEN == "gptoss" else 1500, temperature=0.0, as_json=False)   # no hidden reasoning: leave the 131k window to the story
        except ValueError as e:
            print("FAILED", *k, str(e)[:200], flush=True)
            return
        src = [x for x in sentences(text) if not (GNEG.search(x) and GMETA.search(x)) and not GABOUT.search(x)]
        ratings = list(calls.map(lambda x: rate(story, name, x), src))
        rec = {"book": b, "boundary": bd, "label": l, "name": name, "story_words": len(story.split()), **meta}
        append_jsonl(raw / f"{b}.jsonl", {**rec, "text": text, "words": len(text.split())})
        kept = " ".join(x for x, v in zip(src, ratings) if v == 5)
        append_jsonl(filt / f"{b}.jsonl", {**rec, "text": kept, "words": len(kept.split()), "ratings": [[x, v] for x, v in zip(src, ratings)]})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
