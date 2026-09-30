"""Do the reasoning traces use the character representations? (Qwen3.8-27B, thinking on, saved traces)

  python3 chiron/analyze_traces.py
For each saved trace: the share of the reasoning's word 4-grams that also occur in the representations shown,
and the share that occur in the (masked) passage; whether the trace talks about the notes; and accuracy by
how much it drew on the notes. Prints a few short excerpts. Writes outputs/analysis_traces.json.
"""
import collections
import glob
import json
import os
import re
import statistics as st

from common import DATA, OUT, read_jsonl, write_json

NOTES_WORDS = re.compile(r"\b(sheet|notes?|character information|profile|according to|information says|described as)\b", re.I)


def grams(text, n=4):
    w = re.findall(r"[a-z']+", text.lower())
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    items = {it["item_id"]: it for s in ("test", "val") for it in read_jsonl(DATA / f"items_{s}.jsonl")}
    reps = {}
    for s in ("test", "val"):
        for r in read_jsonl(DATA / f"reps_{s}.jsonl"):
            reps[(r["condition"], r["book"], r["boundary"], r["label"])] = r["text"]
    out, examples = {}, {}
    for f in glob.glob(str(OUT / "eval" / "Qwen3.8-27B_traces_think" / "items_*" / "*.jsonl")):
        cond = os.path.basename(f).split(".")[0]
        base = cond.replace("swapname_", "")
        recs = read_jsonl(f)
        by_gen = collections.defaultdict(list)
        for r in recs:
            by_gen[(r["item_id"], tuple(r["order"]))].append(r)
        stats = collections.defaultdict(list)
        for (iid, order), rs in by_gen.items():
            trace = next((r.get("reasoning") for r in rs if r.get("reasoning")), None)
            if not trace:
                continue
            it = items[iid]
            g = grams(trace)
            if not g:
                continue
            notes = set()
            if base != "noinfo":
                for l in it["labels"]:
                    notes |= grams(reps.get((base, it["book"], it["chapter_index"], l), ""))
            passage = grams(it["masked"])
            correct = sum(not r["invalid"] and max(r["logprobs"], key=r["logprobs"].get) == str(r["answer"]) for r in rs) / len(rs)
            stats["notes_share"].append(len(g & notes) / len(g))
            stats["passage_share"].append(len(g & passage) / len(g))
            stats["mentions_notes"].append(int(bool(NOTES_WORDS.search(trace))))
            stats["correct"].append(correct)
            stats["words"].append(len(trace.split()))
            if cond not in examples and 400 < len(trace) and correct == 1:
                examples[cond] = trace[:1200]
        if not stats["correct"]:
            continue
        hi = [c for c, s in zip(stats["correct"], stats["notes_share"]) if s >= st.median(stats["notes_share"])]
        lo = [c for c, s in zip(stats["correct"], stats["notes_share"]) if s < st.median(stats["notes_share"])]
        out[cond] = {"traces": len(stats["correct"]), "accuracy": st.mean(stats["correct"]),
                     "median_words": st.median(stats["words"]),
                     "median_share_4grams_from_notes": st.median(stats["notes_share"]),
                     "median_share_4grams_from_passage": st.median(stats["passage_share"]),
                     "share_traces_that_talk_about_the_notes": st.mean(stats["mentions_notes"]),
                     "accuracy_when_notes_share_above_median": st.mean(hi) if hi else None,
                     "accuracy_when_notes_share_below_median": st.mean(lo) if lo else None}
    for c, v in sorted(out.items()):
        print(c, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in v.items()}))
    for c, e in examples.items():
        print(f"\n--- example ({c}) ---\n{e}")
    write_json(OUT / "analysis_traces.json", {"stats": out, "examples": examples})


if __name__ == "__main__":
    main()
