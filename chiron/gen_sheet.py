"""New gpt-oss-120b character sheets: one call compresses a character's chapter-by-chapter notes into a sheet.

  python3 chiron/gen_sheet.py --variant NAME --source SOURCE --words W [--style STYLE] --books B... [--workers 48]
Sources (notes on chapters before the boundary only, in chapter order):
  chapnotes_h_long  thorough gpt-oss chapter notes with headings and narrators (gen_chapnotes.py --headings --long)
  chapnotes         the short gpt-oss chapter notes (gen_chapnotes.py)
  legacy_gptoss     gpt-oss with the Llama notes' extraction prompt (gen_legacy_gptoss.py)
  legacy            the Llama-3.3-70B notes from the NCP archive, filler removed (a control on the compression step)
One call per (book, boundary, principal). The brief (STYLES) is a writer's character bible and says nothing about
how sheets are evaluated: bible (identity, names used, relationships, appearance, voice, story, now), chrono (a short
profile plus a sentence or two per chapter), chrono_recent (the latest three chapters in a paragraph each), distinct
(chrono plus what sets the character apart from the other principals), dossier (terse bullet lists), chiron (the
CHIRON / v2 sections: personality, voice, history, knowledge, goals, relationships, with chapter citations) or inner
(the character's inner life, voice, relationships and arc). A sheet outside [0.6, 1.5] x W words is sent back, in the same conversation, for a rewrite
at a length scaled by the model's own overshoot (at most 3 rounds). --faithful: W becomes min(W, 0.35 x the notes'
words), short sheets are never sent back to grow, and the brief forbids inventing names, quotes or forms of address
(the fixed-length sheets padded thin notes with invented detail).
Output: outputs/sheets/<variant>/<book>.jsonl, one record per (boundary, label). Resumable.
"""
import argparse
import collections
import json
import pickle
from concurrent.futures import ThreadPoolExecutor

import llm
from build_reps import FILLER, LEGACY, LEGACY_LABEL, gclean, legacy_blocks
from common import DATA, OUT, append_jsonl, read_jsonl
from gen_chiron import QUESTIONS

SYSTEM = {"role": "system", "content": "You are an expert story editor who keeps the character bible for a novel."}
STYLES = {
    "bible": [
        ("Identity", "who {name} is: age, roles, occupation, status, family, where {name} lives."),
        ("How others refer to {name}", "titles, nicknames, descriptions and forms of address used besides the name, and who "
                                       "uses each."),
        ("Relationships", "one line per important person (the other main characters and recurring minor characters, by "
                          "name): who they are to {name}, how the two interact, and where things stand now."),
        ("Appearance and manner", "distinctive physical traits, clothing, possessions, habits and gestures."),
        ("Voice", "how {name} speaks: vocabulary, tone, recurring phrases."),
        ("Story so far", "the key events involving {name}, in order, each in one sentence that names the people and places "
                         "involved."),
        ("Current situation", "where {name} is, who {name} is with, and what {name} wants or fears at the end of the "
                              "latest chapter."),
    ],
    "chrono": [                                                   # the history kept chapter by chapter
        ("Profile", "identity, roles, titles and nicknames (and who uses them), appearance, possessions, habits and voice, "
                    "in a few dense sentences."),
        ("Relationships", "one line per important person, by name: who they are to {name} and where things stand now."),
        ("Chapter by chapter", "for every chapter in which {name} appears, one or two sentences on what {name} does there, "
                               "naming the people, places and objects involved. Mark each with its chapter number."),
        ("Current situation", "where {name} is, who {name} is with, and what {name} wants at the end of the latest "
                              "chapter."),
    ],
    "chrono_recent": [                                            # chrono, with the latest chapters in more detail
        ("Profile", "identity, roles, titles and nicknames (and who uses them), appearance, possessions, habits and voice, "
                    "in a few dense sentences."),
        ("Relationships", "one line per important person, by name: who they are to {name} and where things stand now."),
        ("Chapter by chapter", "for every chapter in which {name} appears, one or two sentences on what {name} does there, "
                               "naming the people, places and objects involved; for the three latest chapters, a short "
                               "paragraph each. Mark each with its chapter number."),
        ("Current situation", "where {name} is, who {name} is with, what {name} knows and wants, and what is unresolved "
                              "at the end of the latest chapter."),
    ],
    "distinct": [                                                 # chrono, plus what sets the character apart
        ("Profile", "identity, roles, titles and nicknames (and who uses them), appearance, possessions, habits and voice, "
                    "in a few dense sentences."),
        ("Distinguishing marks", "what sets {name} apart from {others}: the roles, people, places, habits, objects and "
                                 "turns of phrase that belong to {name} and not to them."),
        ("Relationships", "one line per important person, by name: who they are to {name} and where things stand now."),
        ("Chapter by chapter", "for every chapter in which {name} appears, one or two sentences on what {name} does there, "
                               "naming the people, places and objects involved. Mark each with its chapter number."),
        ("Current situation", "where {name} is, who {name} is with, and what {name} wants at the end of the latest "
                              "chapter."),
    ],
    "chiron": [                                                   # the CHIRON / v2 sheet sections
        ("Physicality And Personality", "appearance, manner, temperament and how {name} tends to act and react."),
        ("Dialogue And Voice", "how {name} speaks and to whom, with short characteristic quotes from the notes."),
        ("History And Circumstances", "background, situation and the key events {name} has been part of."),
        ("Knowledge And Beliefs", "what {name} knows, has learned, believes and suspects."),
        ("Goals And Motivations", "what {name} wants, fears and is driven by, and how that has changed."),
        ("Relationships", "the people {name} deals with, by name, and how {name} feels about each."),
    ],
    "inner": [                                                    # the character from the inside
        ("Who {name} is", "identity, role, situation and appearance in a few sentences."),
        ("Inner life", "temperament, feelings, fears, desires, beliefs and secrets: how {name} thinks and reacts, and how "
                       "this has changed over the story."),
        ("Voice", "how {name} speaks and thinks in words, with short characteristic quotes from the notes."),
        ("Relationships", "the people {name} deals with, by name: what each is to {name} and how {name} feels about them "
                          "now."),
        ("Arc so far", "the turning points for {name}, in order, with what {name} felt and decided at each."),
        ("Now", "where {name} stands at the end of the latest chapter: situation, company, mood, what {name} wants next."),
    ],
    "dossier": [                                                  # compact lists instead of prose
        ("Names and titles", "every name, title, nickname and form of address used for {name}, and who uses it."),
        ("Facts", "identity, occupation, family, home, age, appearance, possessions and habits, as terse bullet points."),
        ("People", "a bullet per person {name} has dealt with, by name: their relation to {name}, and the latest state "
                   "between them."),
        ("Places", "bullets for the places {name} lives, works or has been, and what happened there."),
        ("Voice", "a few bullets on how {name} speaks, with short characteristic quotes."),
        ("Timeline", "a bullet per chapter in which {name} appears: chapter number, then what {name} does, naming the "
                     "people and places involved."),
        ("Now", "bullets on where {name} is, with whom, and what {name} wants at the end of the latest chapter."),
    ],
}


FAITHFUL = (" Never invent anything: no names, quotes, nicknames, forms of address or details the notes do not state. If the "
            "notes give nothing for a section, write: None noted. Shorter is fine when the notes are thin.")


RULES = {"bible": "Write short, self-contained sentences that name people instead of using pronouns. Prefer concrete, distinctive "
                  "details over general personality traits.",
         "dossier": "Write terse bullet points that name people instead of using pronouns. Prefer concrete, distinctive details "
                    "over general personality traits.",
         "chiron": "Write bullet points, each a self-contained sentence that names people instead of using pronouns, ending with "
                   "the chapters it comes from, like [Ch. 4] or [Chs. 2, 7]. Be specific to this character.",
         "inner": "Write short, self-contained sentences that name people instead of using pronouns. Be specific to this "
                  "character: what sets {name}'s thoughts, feelings and words apart from anyone else's."}


def brief(name, others, words, style, upto, faithful=False):
    secs = "\n".join(f"- {h.format(name=name)}: {d.format(name=name, others=' and '.join(others))}" for h, d in STYLES[style])
    return (f"Write a character sheet for {name} as of the end of chapter {upto}, for a writer continuing the novel who must "
            f"keep {name} consistent. The other main characters are {', '.join(others)}. About {words} words, with these "
            f"sections, each under a markdown heading (## Section name):\n{secs}\n" + RULES.get(style, RULES["bible"]).format(name=name)
            + " Use only the notes." + (FAITHFUL if faithful else ""))


def write(msgs, target, low=0.6):
    text, meta = llm.ask(msgs, lambda x: x, max_tokens=24000, as_json=False)
    ask = target
    for rnd in range(3):
        n = len(text.split())
        if low * target <= n <= 1.5 * target:
            return text, {**meta, "resize_rounds": rnd}
        ask = max(50, int(ask * target / n))
        text, _ = llm.ask(msgs + [{"role": "assistant", "content": text},
                                  {"role": "user", "content": f"That sheet is {n} words. Rewrite it at about {ask} words, with the "
                                                              "same sections and rules."}],
                          lambda x: x, max_tokens=24000, as_json=False)
    n = len(text.split())
    if not low * target <= n <= 1.5 * target:
        raise ValueError(f"still {n} words after resizing")
    return text, {**meta, "resize_rounds": 3}


def load_notes(source, keys):
    """(book, boundary, label) -> [(chapter, note text)] for chapters < boundary."""
    per = collections.defaultdict(dict)                           # (book, label) -> chapter -> text
    if source == "legacy_gptoss":
        tmp = collections.defaultdict(dict)
        for f in (OUT / "legacy_gptoss").glob("*.jsonl"):
            for r in read_jsonl(f):
                tmp[(r["book"], r["label"], r["chapter_index"])][r["q"]] = r["answer"].strip()
        for (b, l, c), d in tmp.items():
            ans = [gclean(d.get(q, "")) for q in QUESTIONS]
            per[(b, l)][c] = "\n".join(f"- {a}" for a in ans if a)
    elif source in ("chapnotes", "chapnotes_h_long"):
        for f in (OUT / source).glob("*.jsonl"):
            for r in read_jsonl(f):
                per[(r["book"], r["label"])][r["chapter_index"]] = "\n".join(f"- {r['answers'][q]}" for q in QUESTIONS if r["answers"].get(q))
    else:                                                         # Llama notes: question-major blocks regrouped by chapter
        out = {}
        for s in ("test", "val", "train"):
            full = pickle.load(open(LEGACY / f"{s}_long_story_storycharchap_to_csheet.pkl", "rb"))
            for b, bd, l in keys:
                t = full.get(f"{b}_{LEGACY_LABEL.get(l, l)}_{bd}")
                if t and (b, bd, l) not in out:
                    chs = collections.defaultdict(list)
                    for _, blocks in legacy_blocks(t):
                        for c, x in blocks:
                            x = " ".join(FILLER.sub("", x).split())
                            if x:
                                chs[c].append(f"- {x}")
                    out[(b, bd, l)] = [(c, "\n".join(chs[c])) for c in sorted(chs) if c < bd]
        return out
    return {(b, bd, l): [(c, t) for c, t in sorted(per[(b, l)].items()) if c < bd and t] for b, bd, l in keys
            if len([c for c in per[(b, l)] if c < bd]) == bd}       # every earlier chapter has notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--source", required=True, choices=["chapnotes_h_long", "chapnotes", "legacy_gptoss", "legacy"])
    ap.add_argument("--words", type=int, default=1000)
    ap.add_argument("--style", default="bible")
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--faithful", action="store_true", help="target at most 0.35 x the notes' length, never expand, no invention")
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    principals = json.load(open(DATA / "principals.json"))
    items = [it for s in ("test", "val", "train") for it in read_jsonl(DATA / f"items_{s}.jsonl") if it["book"] in args.books]
    keys = {(it["book"], it["chapter_index"], l) for it in items for l in it["labels"]}
    root = OUT / "sheets" / args.variant
    root.mkdir(parents=True, exist_ok=True)
    done = {(r["book"], r["boundary"], r["label"]) for b in args.books for r in read_jsonl(root / f"{b}.jsonl")}
    notes = load_notes(args.source, keys)
    todo = sorted((k for k in keys if k not in done and k in notes), key=lambda k: -sum(len(t) for _, t in notes[k]))
    todo = todo[:args.limit] if args.limit else todo
    print(len(todo), "sheets to write;", len(keys - set(notes)), "keys without complete notes", flush=True)

    def one(k):
        b, bd, l = k
        name = principals[b]["gen_names"][l][bd - 1]
        others = [principals[b]["gen_names"][o][bd - 1] for o in principals[b]["labels"] if o != l]
        body = "\n\n".join(f"Chapter {c + 1}:\n{t}" for c, t in notes[k])
        words = min(args.words, max(100, round(0.35 * len(body.split()), -1))) if args.faithful else args.words
        msgs = [SYSTEM, {"role": "user", "content": f"Chapter-by-chapter notes about {name} (chapters 1 to {bd}):\n\n{body}\n\n"
                                                    + brief(name, others, words, args.style, bd, args.faithful)}]
        try:
            text, meta = write(msgs, words, 0 if args.faithful else 0.6)
        except Exception as e:
            print("FAILED", b, bd, l, str(e)[:200], flush=True)
            return
        append_jsonl(root / f"{b}.jsonl", {"book": b, "boundary": bd, "label": l, "name": name, "text": text,
                                           "words": len(text.split()), "target": words, "notes_words": len(body.split()), **meta})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
