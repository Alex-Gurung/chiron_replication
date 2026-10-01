"""Run charmem's exact principal-sheet synthesis (charmem_synth.py, copied verbatim) on a choice of notes.

  python3 chiron/gen_charsynth.py --source ledger|legacy_gptoss_ent --variant NAME --books B... [--workers 96]
ledger            charmem's own source-reviewed ledgers: a replication check against the charmem condition (71.9)
legacy_gptoss_ent gpt-oss Llama-prompt notes kept by gpt-oss's entailment ratings (5 of 5), one entry per sentence;
                  questions map to charmem's sections (physical/personality -> physicality_and_personality, dialogue ->
                  dialogue_and_voice, facts -> history_and_circumstances, learned -> knowledge_and_beliefs, goals and
                  motivation -> goals_and_motivations); a presence line per chapter (absent when nothing survived)
Same prompt, JSON schema, validator (with charmem's repair of unknown or leaked source IDs applied before every check) (<= 1,000 words, every note cites supplied source IDs), temperature 0.1 and
rendering ("## Section", "- note [Ch. N]") as charmem (entries beyond ~90k tokens lose their oldest lines); --words changes the requested length (cap 1.1x). Output: outputs/sheets/<variant>/<book>.jsonl. Resumable.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import charmem_synth
import llm
from charmem_synth import MAX_TOKENS, make_repair, principal_messages, render_principal, source_entries, validate_principal
from common import DATA, OUT, append_jsonl, load_narrators, read_jsonl
from gen_sheet import LEDGERS

MAX_ENTRY_CHARS = 300000                                          # ~90k tokens, leaving room for 24k of output
SECTION_OF = {"physical": "physicality_and_personality", "personality": "physicality_and_personality",
              "dialogue": "dialogue_and_voice", "facts": "history_and_circumstances", "learned": "knowledge_and_beliefs",
              "goals_gained": "goals_and_motivations", "goals_completed": "goals_and_motivations",
              "motivation_change": "goals_and_motivations"}


def ent_entries(rated, label, bd, narrators, book):
    """charmem-format entries from entailment-rated sentences of chapters < bd."""
    lines, ids = [], {}
    for c in range(bd):
        r = rated.get(c, {})
        keep = [(SECTION_OF[q], x) for q, v in r.items() for x, s in v if s >= 5]
        n = narrators.get((book, c))
        sid = f"ch{c:03d}:presence:000"
        ids[sid] = c
        lines.append(json.dumps({"id": sid, "ch": str(c + 1), "presence": "present" if keep else "absent",
                                 "narration": "first_person" if n else "", "pov": n or "", "setting": ""}, ensure_ascii=False))
        count = {}
        for sec, x in keep:
            count[sec] = count.get(sec, 0) + 1
            sid = f"ch{c:03d}:{sec}:{count[sec]:03d}"
            ids[sid] = c
            lines.append(json.dumps({"id": sid, "ch": str(c + 1), "sec": sec, "claim": x, "when": ""}, ensure_ascii=False))
    return "\n".join(lines), ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["ledger", "legacy_gptoss_ent"])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--words", type=int, default=900, help="requested words (charmem: 900, capped at 1,000 = 1.1x)")
    args = ap.parse_args()
    charmem_synth.REQUESTED_PRINCIPAL_WORDS, charmem_synth.MAX_PRINCIPAL_WORDS = args.words, round(1.1 * args.words + 10, -1)
    assert llm.server_up(), "gpt-oss server not reachable"
    items = [it for s in ("test", "val", "train") for it in read_jsonl(DATA / f"items_{s}.jsonl") if it["book"] in args.books]
    keys = {(it["book"], it["chapter_index"], l) for it in items for l in it["labels"]}
    root = OUT / "sheets" / args.variant
    root.mkdir(parents=True, exist_ok=True)
    done = {(r["book"], r["boundary"], r["label"]) for b in args.books for r in read_jsonl(root / f"{b}.jsonl")}
    if args.source == "ledger":
        ledgers = {}
        for f in LEDGERS.glob("*.json"):
            d = json.load(open(f))
            if d["book_id"] in args.books:
                ledgers[(d["book_id"], d["chapter_index"])] = d
    else:
        rated = {}
        for b in args.books:
            for r in read_jsonl(OUT / "legacy_gptoss_ent" / f"{b}.jsonl"):
                rated.setdefault((b, r["label"]), {})[r["chapter_index"]] = r["rated"]
        narrators = load_narrators()
    todo = sorted(k for k in keys if k not in done)
    print(len(todo), "to do", flush=True)

    def one(k):
        b, bd, l = k
        if args.source == "ledger":
            lgs = [ledgers[(b, c)] for c in range(bd)]
            entries, ids = source_entries(lgs, character=l)
            labels = {lg["chapter_index"]: lg["chapter_label"] for lg in lgs}
            nxt = ledgers[(b, bd)]["chapter_label"] if (b, bd) in ledgers else str(bd + 1)
        else:
            entries, ids = ent_entries(rated.get((b, l), {}), l, bd, narrators, b)
            labels, nxt = {c: str(c + 1) for c in range(bd)}, str(bd + 1)

        lines = entries.split("\n")
        while sum(len(x) + 1 for x in lines) > MAX_ENTRY_CHARS:     # over gpt-oss's 131k context: oldest entries out
            lines = lines[1:]
        entries = "\n".join(lines)

        def check(p):                                             # charmem's last-attempt repair, applied every time
            if isinstance(p, dict):
                p["character"] = l
                make_repair(ids)(p, "")
            validate_principal(p, l, ids)
            return p
        try:
            payload, meta = llm.ask(principal_messages(l, entries, labels[bd - 1], nxt, b), check, max_tokens=MAX_TOKENS)
        except Exception as e:
            print("FAILED", b, bd, l, str(e)[:200], flush=True)
            return
        text = render_principal(payload, ids, labels)
        append_jsonl(root / f"{b}.jsonl", {"book": b, "boundary": bd, "label": l, "text": text, "words": len(text.split()), **meta})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, todo))


if __name__ == "__main__":
    main()
