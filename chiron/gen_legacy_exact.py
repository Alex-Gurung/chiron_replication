"""The Llama notes' pipeline after extraction, redone with gpt-oss as the NCP archive ran it. Prompts, sentence splitter,
settings and retention rule are copied from the diversity repo's ncp_eval/data_creation/legacy_character_replay.py
(its gpt-oss replay of the archive). Input: the gpt-oss extraction answers (gen_legacy_gptoss.py, the archive's
extraction prompt word for word).

Per (chapter, principal, question) answer:
  1. sentences: spaCy en_core_web_md, each line separately (the archive's splitter), minus the sentences about what the
     chapter does not say ("The text does not mention Fern's hair.", build_reps.gclean's rule). The rubric below rates
     those 1 anyway; dropping them first only saves calls.
  2. simplification: one call per sentence with the archive's few-shot prompt (split compound sentences, resolve unclear
     pronouns), low reasoning, temperature 0.6, top_p 0.9. The first line of the reply (the archive stopped at "\\n") is
     re-split by spaCy into candidates.
  3. entailment: one call per candidate against the whole chapter with the archive's role and rubric, low reasoning,
     temperature 0, 1,024 tokens. The rating is the last digit 1-5 in the reply (none = 1); only 5s are kept.
Departures, both fixes: the archive's replay capped simplification at 256 tokens with stop "\\n", which can end inside
gpt-oss's hidden reasoning and silently drop the sentence; here it gets 1,024 tokens and an empty reply keeps the
sentence as it was. Filler sentences are removed before step 2.

  python3 chiron/gen_legacy_exact.py --books B... [--workers 64]
Output: outputs/legacy_gptoss_x/<book>.jsonl, one record per (chapter, label, question):
  {book, chapter_index, label, name, q, sentences: [{source, statement, rating}]}. Resumable.
"""
import argparse
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.append("/home/toolkit/chiron_replication/.pylib")        # spaCy + en_core_web_md (uv pip install --target)
import spacy  # noqa: E402

import llm  # noqa: E402
from build_reps import GABOUT, GMETA, GNEG  # noqa: E402
from common import OUT, append_jsonl, clean_text, load_chapters, read_jsonl  # noqa: E402

ROOT = OUT / "legacy_gptoss_x"
NLP = spacy.load("en_core_web_md")
NLP_LOCK = threading.Lock()
SIMPLIFICATION_EXAMPLES = (
    ("She's curious about a closed door in Maxim's apartment and feels a strong urge to discover what's behind it.",
     "She's curious about a closed door in Maxim's apartment. She feels a strong urge to discover what's behind the closed "
     "door in Maxim's apartment."),
    ("Kaluros is determined and focused during battles, using his magic and weapons effectively to defeat his enemies.",
     "Kaluros is determined and focused during battles, using his magic and weapons effectively to defeat his enemies."),
    ("Hassan encountered a crab monster and engaged in a card battle to defeat it.",
     "Hassan encountered a crab monster. Hassan engaged in a card battle to defeat the crab monster."),
    ("She uses imperatives to give orders and asks direct questions to gather information.",
     "She uses imperatives to give orders. She asks direct questions to gather information."),
    ("Bob is easily distracted and forgets about the chase when he notices something outside.",
     "Bob is easily distracted. Bob forgets about the chase when he notices something outside."),
    ("Rachel enters the warehouse to join the baby dragon, defying her initial skepticism.",
     "Rachel enters the warehouse to join the baby dragon, defying her initial skepticism."),
    ("He gives commands to his companions and asks for their assistance.",
     "He gives commands to his companions. He asks for his companions' assistance."),
    ("She explores the Zombear's massive body and climbs on it.",
     "She explores the Zombear's massive body. She climbs on the Zombear."),
    ("Jordan opens the locker to find a locket, a newspaper, and a mysterious photograph.",
     "Jordan opens the locker to find a locket, a newspaper, and a mysterious photograph."),
    ("He is quiet and tosses a gold idol between his hands while they wait for rescue.",
     "He is quiet. He tosses a gold idol between his hands while they wait for rescue."),
)
ENTAILMENT_ROLE = (
    "You are a helpful and expert writing assistant. You will be given a section of a story or screenplay from the "
    "perspective of {character}. Please answer the following questions about the given statements and their relationship "
    "with the snippet provided.")
ENTAILMENT_QUESTION = (
    "Rate the accuracy of provided statement about {character} on a scale of 1-5, where 1 is entirely inaccurate or "
    "unsupported and 5 is entirely accurate. If there are no claims made in the statement, mark the consistency of the "
    "statement as 1 as there is no evidence for the statement. Your response should be formatted as 'Answer: <number>'\n"
    "Notes for accuracy:\n"
    "Statements that are do not give us new information about {character}, or are not about {character} should be marked "
    "as 1. Even if the statement is true (e.g. 'X has no goals'), it should be marked as 1 as it does not give us new "
    "information about the character.\n"
    "Statements that are not supported by the story snippet should be marked as 1.\n"
    "Statements that are supported by the story snippet should be marked as 4 or 5, depending on how much evidence there "
    "is for the statement.")


def sentences(text):
    out = []
    with NLP_LOCK:
        doc = NLP(text)
    for s in doc.sents:
        for line in s.text.strip().split("\n"):
            line = line.strip()
            if line and line not in {"Sure!", "Sure, I'd be happy to help!"}:
                out.append(line)
    return out


def simplification_messages(statement):
    examples = "\n".join(f"Sentence: {a}\nSplit Sentences: {b}" for a, b in SIMPLIFICATION_EXAMPLES)
    return [{"role": "user", "content": (
        "Given the provided sentence, please split all independent clauses into independent sentences and resolve any issues "
        "with unclear pronouns or references. Only do this for compound sentences. Every new sentence should make sense on "
        "its own. Write them out in paragraph form, one sentence after another. Non-compound sentences can returned as they "
        f"are.\nExamples:\n{examples}\nWrite out your answer in paragraph form, one sentence after another. Do not include "
        f"any other text.\nSentence: {statement}")}]


def entailment_messages(chapter, character, statement):
    cleaned = re.sub(r"(\n ?)+", "\n", re.sub(r" +", " ", chapter.strip()))
    return [{"role": "system", "content": ENTAILMENT_ROLE.format(character=character)},
            {"role": "user", "content": (
                f"Story Section:    \n{cleaned}\n\nPlease answer the following questions about {character} by comparing the "
                f"provided statement with the story section above:\n\nStatement: {statement.strip()}\n\n"
                f"Question: {ENTAILMENT_QUESTION.format(character=character)}")}]


def reply(msgs, temperature, top_p):
    resp = llm.chat(msgs, 1024, temperature, top_p=top_p, effort="low")
    return (resp["choices"][0]["message"].get("content") or "").strip()


def simplify(statement):
    raw = reply(simplification_messages(statement), 0.6, 0.9)
    raw = raw.replace("Split Sentences:", "").replace("Split Sentence:", "").replace("</s>", "").strip().split("\n")[0]
    return sentences(raw) or [statement]


def rate(chapter, character, statement):
    m = re.findall(r"(?<!\d)([1-5])(?!\d)", reply(entailment_messages(chapter, character, statement), 0.0, 1.0))
    return int(m[-1]) if m else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    ROOT.mkdir(parents=True, exist_ok=True)
    chapters = load_chapters()
    calls = ThreadPoolExecutor(6 * args.workers)
    for b in args.books:
        text = {c["chapter_index"]: clean_text(c["chapter_text_normalized"]) for c in chapters[b]}
        done = {(r["chapter_index"], r["label"], r["q"]) for r in read_jsonl(ROOT / f"{b}.jsonl")}
        todo = [r for r in read_jsonl(OUT / "legacy_gptoss" / f"{b}.jsonl") if (r["chapter_index"], r["label"], r["q"]) not in done]
        print(b, len(todo), "answers to do", flush=True)

        def one(r):
            src = [x for x in sentences(r["answer"]) if not (GNEG.search(x) and GMETA.search(x)) and not GABOUT.search(x)]
            split = list(calls.map(simplify, src))
            pairs = [(s, x) for s, xs in zip(src, split) for x in xs]
            ratings = list(calls.map(lambda p: rate(text[r["chapter_index"]], r["name"], p[1]), pairs))
            append_jsonl(ROOT / f"{b}.jsonl", {**{k: r[k] for k in ("book", "chapter_index", "label", "name", "q")},
                                               "sentences": [{"source": s, "statement": x, "rating": v} for (s, x), v in zip(pairs, ratings)]})
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(one, todo))


if __name__ == "__main__":
    main()
