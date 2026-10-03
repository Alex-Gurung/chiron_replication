"""The Llama notes' extraction step redone with gpt-oss-120b, prompt for prompt (no splitting or entailment filter).

The NCP archive's Llama-3.3-70B notes answer CHIRON's 8 questions once per (chapter, principal), one call per question,
with the whole chapter as the story section and the extraction prompt below (copied from the diversity repo's
ncp_eval/data_creation/legacy_character_replay.py). The archive then split, simplified and entailment-filtered those
answers; this script stops after extraction, to isolate the prompt and the model.

  python3 chiron/gen_legacy_gptoss.py --books B... [--workers 96]
Covers every chapter before the last boundary the passages need. Output: outputs/legacy_gptoss/<book>.jsonl, one record
per (chapter, label, question). Resumable.

Prompt variants (--variant V writes outputs/legacy_gptoss_V/ instead): --questions Q... answers only those questions;
--effort low|medium sets gpt-oss's reasoning effort (medium above; the diversity repo's gpt-oss replay used low); --brief
words the personality and speech questions as a writer's brief (traits and manner, not what happens on the page), since
gpt-oss reads the archive's wording literally and answers with actions and features of single lines.
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import llm
from common import DATA, OUT, append_jsonl, clean_text, load_chapters, read_jsonl
from gen_chiron import QUESTIONS

ROOT = OUT / "legacy_gptoss"
ROLE = ("You are a helpful and expert writing assistant. You will be given a section of a story or screenplay. Please answer "
        "the following questions about the character learned in this story section, and respond in paragraph form.")


def messages(chapter, character, question):
    cleaned = re.sub(r"(\n ?)+", "\n", re.sub(r" +", " ", chapter.strip()))
    return [{"role": "system", "content": ROLE},
            {"role": "user", "content": (
                f"Story Section:\n{cleaned}\n\nPlease answer the following questions about {character} with short, succinct "
                f"sentences based on the given story section.\n\nQuestion: {question} Respond with a comprehensive paragraph "
                "with short, simple sentences with no dependent clauses or transition words. Make sure that each sentence is a "
                "complete thought, and has all of the information necessary to be understood. Only respond with answers that "
                "you think will be useful for our understanding of the character moving forward. Be exhaustive in your "
                "response, giving information from the across the entire story section. Limit your response to at most 10 "
                "sentences of critical information.\nAnswer:")}]


BRIEF = {
    "personality": "What, if any, descriptions of this character's personality are in this snippet? Describe their temperament, "
                   "attitudes and habits, and how they treat the people around them, as general traits rather than a list of "
                   "what they do in this snippet.",
    "dialogue": "What, if anything, have we learned about how this character speaks from this snippet? Describe their manner and "
                "attitude when they talk, how they talk to particular people, and what they tend to talk about, rather than "
                "features of individual lines.",
}


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--variant", default="")
    ap.add_argument("--questions", nargs="+", default=list(QUESTIONS))
    ap.add_argument("--effort", default="medium", choices=["low", "medium"])
    ap.add_argument("--brief", action="store_true")
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    llm.REASONING_EFFORT = args.effort
    ROOT = OUT / ("legacy_gptoss" + (f"_{args.variant}" if args.variant else ""))
    ask = {q: (BRIEF.get(q, QUESTIONS[q][1]) if args.brief else QUESTIONS[q][1]) for q in args.questions}
    principals = json.load(open(DATA / "principals.json"))
    chapters = load_chapters()
    last = {}
    for s in ("test", "val", "train"):
        for it in read_jsonl(DATA / f"items_{s}.jsonl"):
            last[it["book"]] = max(last.get(it["book"], 0), it["chapter_index"])
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for book in args.books:
        done = {(r["chapter_index"], r["label"], r["q"]) for r in read_jsonl(ROOT / f"{book}.jsonl")}
        jobs += [(book, ch, l, q) for ch in chapters[book] if ch["chapter_index"] < last.get(book, 0)
                 for l in principals[book]["labels"] for q in ask if (ch["chapter_index"], l, q) not in done]
    print(len(jobs), "to do", flush=True)

    def one(job):
        book, ch, l, q = job
        name = principals[book]["gen_names"][l][ch["chapter_index"]]
        try:
            text, meta = llm.ask(messages(clean_text(ch["chapter_text_normalized"]), name, ask[q]),
                                 lambda x: x, max_tokens=6000, temperature=0.6, as_json=False)
        except ValueError as e:
            print("FAILED", book, ch["chapter_index"], l, q, str(e)[:200], flush=True)
            return
        append_jsonl(ROOT / f"{book}.jsonl", {"book": book, "chapter_index": ch["chapter_index"], "label": l, "name": name, "q": q,
                                              "answer": text, **meta})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
