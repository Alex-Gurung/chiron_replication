"""The Llama notes' final step, run with gpt-oss: summarize a character's full chapter notes into a condensed sheet with
the archive's own prompt (SUMMARY_ROLE / SUMMARY_QUERY / SUMMARY_INSTRUCTION, copied from the diversity repo's
ncp_eval/data_creation/legacy_character_replay.py). The archive's Llama summary of the Llama notes is the "legacy"
condition (~500 words, 70.4 on the 27B at 2.7k prompt tokens).

  python3 chiron/gen_legsum.py --source legacy_gptoss_ent|legacy_gptoss_nofill|legacy_gptoss_x|legacy_gptoss_xs|exact --variant NAME --books B... [--words W]
The input is the source's Llama-layout notes (build_reps: questions grouped by section, one <snippet c> per chapter).
--words W appends "Use about W words." to the instruction (none by default: the archive gave no length, only a token cap)
and, since gpt-oss writes ~2x that, sends longer summaries back to be rewritten shorter (gen_sheet.write, <= 1.5 x W).
--source exact reads the current generator's notes (CHIRON_GEN: outputs/legacy_<gen>_x); a generator without hidden
reasoning is decoded as the archive did (at most min(0.8 x input, 2048) tokens, no length guidance).
Output: outputs/sheets/<variant>/<book>.jsonl (picked up by build_reps as sheet_<variant>). Resumable.
"""
import argparse
import collections
import json
from concurrent.futures import ThreadPoolExecutor

import llm
from build_reps import CHAPNOTE_LAYOUT, archive_layout, exact_notes, exact_notes_xs, gclean
from common import DATA, GEN, OUT, append_jsonl, read_jsonl
from gen_chiron import QUESTIONS
from gen_sheet import write

SUMMARY_ROLE = "You are an expert author and writing assistant."
SUMMARY_INSTRUCTION = (
    "Write a condensed version of the character sheet provided. Please summarize the sections within the character sheet "
    "provided above to create a more condensed character sheet, make sure to include vital information related to key events, "
    "backgrounds, settings, characters, their objectives, and motivations. Do not include any statements that do not add to "
    "our understanding of the character (e.g. \"there are no physical descriptions of X\"). You must briefly introduce "
    "characters, places, and other major elements if they are being mentioned for the first time in the summary. The story "
    "may feature non-linear narratives, flashbacks, switches between alternate worlds or viewpoints, etc. Therefore, you should "
    "organize the summary so it presents a consistent and chronological narrative. Please mark and describe how the character "
    "changes over time, using the snippet ids as reference.")
SUMMARY_QUERY = (
    "Below is a character sheet up to a point in a story (i.e. the character sheet does not represent the final state of the "
    "character). Please summarize the sections within the character sheet provided above to create a more condensed character "
    "sheet, make sure to include vital information related to key events, backgrounds, settings, characters, their objectives, "
    "and motivations. Do not include any statements that do not add to our understanding of the character (e.g. \"there are no "
    "physical descriptions of X\"). You must briefly introduce characters, places, and other major elements if they are being "
    "mentioned for the first time in the summary. The story may feature non-linear narratives, flashbacks, switches between "
    "alternate worlds or viewpoints, etc. Therefore, you should organize the summary so it presents a consistent and "
    "chronological narrative. Please mark and describe how the character changes over time, using the snippet ids as "
    "reference.\n---\n{character_sheet}\n---\n{instruction}")


def layouts(source, keys):
    """(book, boundary, label) -> the Llama-layout notes text, as build_reps renders the source."""
    if source in ("legacy_gptoss_x", "legacy_gptoss_xs", "exact"):   # the archive's own layout, every chapter listed
        have = exact_notes(f"legacy_{GEN}_x") if source == "exact" else exact_notes() if source == "legacy_gptoss_x" else exact_notes_xs()
        return {(b, bd, l): archive_layout(have[(b, l)], bd) for b, bd, l in keys if all(c in have.get((b, l), {}) for c in range(bd))}
    ans = collections.defaultdict(dict)                           # (book, label) -> chapter -> q -> text
    if source == "legacy_gptoss_ent":
        for f in (OUT / "legacy_gptoss_ent").glob("*.jsonl"):
            for r in read_jsonl(f):
                ans[(r["book"], r["label"])][r["chapter_index"]] = {q: " ".join(x for x, s in v if s >= 5) for q, v in r["rated"].items()}
    else:
        for f in (OUT / "legacy_gptoss").glob("*.jsonl"):
            for r in read_jsonl(f):
                ans[(r["book"], r["label"])].setdefault(r["chapter_index"], {})[r["q"]] = gclean(r["answer"])
    out = {}
    for b, bd, l in keys:
        have = ans.get((b, l), {})
        chs = [c for c in sorted(have) if c < bd]
        if len(chs) != bd:
            continue
        parts = []
        for head, qs in CHAPNOTE_LAYOUT:
            parts.append(f"## {head}")
            for q in qs:
                body = "\n".join(f"<snippet {c}>\n{have[c].get(q, '')}" for c in chs if have[c].get(q, "").strip())
                if body:
                    parts.append(f"Question: {QUESTIONS[q][1]}\n\n{body}")
        out[(b, bd, l)] = "\n\n".join(parts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["legacy_gptoss_ent", "legacy_gptoss_nofill", "legacy_gptoss_x", "legacy_gptoss_xs", "exact"])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--words", type=int, default=0)
    ap.add_argument("--workers", type=int, default=128)
    args = ap.parse_args()
    assert llm.server_up(), "generation server not reachable"
    items = [it for s in ("test", "val", "train") for it in read_jsonl(DATA / f"items_{s}.jsonl") if it["book"] in args.books]
    keys = {(it["book"], it["chapter_index"], l) for it in items for l in it["labels"]}
    root = OUT / "sheets" / args.variant
    root.mkdir(parents=True, exist_ok=True)
    done = {(r["book"], r["boundary"], r["label"]) for b in args.books for r in read_jsonl(root / f"{b}.jsonl")}
    sheets = layouts(args.source, keys)
    todo = sorted((k for k in sheets if k not in done), key=lambda k: -len(sheets[k]))
    print(len(todo), "to do;", len(keys - set(sheets)), "keys without complete notes", flush=True)
    instruction = SUMMARY_INSTRUCTION + (f" Use about {args.words} words." if args.words else "")

    def one(k):
        msgs = [{"role": "system", "content": SUMMARY_ROLE},
                {"role": "user", "content": SUMMARY_QUERY.format(character_sheet=sheets[k], instruction=instruction)}]
        try:
            if GEN != "gptoss":                                   # the archive's decoding: min(0.8 x input tokens, 2048) tokens
                c = llm.chat(msgs, min(int(0.8 * 1.35 * len(sheets[k].split())), 2048), 0.6, top_p=0.9)["choices"][0]
                text, meta = (c["message"].get("content") or "").strip(), {"finish_reason": c.get("finish_reason")}
                if not text:
                    raise ValueError("empty summary")
            elif args.words:                                      # gpt-oss ignores the length: rewrite until <= 1.5 x W
                text, meta = write(msgs, args.words, 0)
            else:
                text, meta = llm.ask(msgs, lambda x: x, max_tokens=24000, temperature=0.6, as_json=False)
        except ValueError as e:
            print("FAILED", *k, str(e)[:200], flush=True)
            return
        append_jsonl(root / f"{k[0]}.jsonl", {"book": k[0], "boundary": k[1], "label": k[2], "text": text, "words": len(text.split()),
                                              "notes_words": len(sheets[k].split()), **meta})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
