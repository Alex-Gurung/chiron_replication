"""Per-passage oracle representations with gpt-oss-120b -> data/reps_item_<split>.jsonl.

oracle_passage  2-4 clues per character taken from the unmasked passage itself: a ceiling that tests whether
                a model can use exactly the right information when it is given.
oracle_prior    up to 5 facts per character copied verbatim from what was known before this chapter (the
                most recent legacy sheet and the charmem sheet), chosen because they identify the character in
                this passage. The writer sees the unmasked passage; facts that are not verbatim copies are dropped,
                so nothing from the passage leaks in. This is "a sheet with exactly the facts needed".
One gpt-oss call per passage; resumable; shardable.
"""
import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from common import DATA, append_jsonl, read_jsonl
import llm


def norm(s):
    return " ".join(re.sub(r"[“”]", '"', re.sub(r"[‘’]", "'", s)).split()).lower()


def messages(passage, names, materials):
    mats = "\n\n".join(f"### Earlier notes on {names[l]}\n{t}" for l, t in materials.items())
    keys = ", ".join(f'"{names[l]}"' for l in materials)
    return [{"role": "system", "content": "You build test materials for a character-identification study. Return only JSON."},
            {"role": "user", "content": (
                f"# Passage from a novel\n\n{passage}\n\n# Notes written before this chapter\n\n{mats}\n\n# Task\n\n"
                f"In a test, the names of {keys} are hidden in this passage and a reader must say which hidden name is which "
                "character. Prepare two things for each character.\n"
                "1. passage_clues: 2 to 4 short statements, based only on the passage, that would let the reader recognise "
                "the character in it (what they do, say, or have happen to them here). Refer to the character as \"this "
                "character\"; do not name them.\n"
                "2. prior_facts: up to 5 facts from that character's earlier notes that would best help the reader recognise "
                "them in this passage. Copy each fact exactly, word for word, from the notes (one sentence or one bullet). "
                "Use only that character's own notes. Give an empty list if nothing in the notes helps.\n\n"
                f'Return JSON: {{"passage_clues": {{{keys.replace(",", ": [...],")}: [...]}}, '
                f'"prior_facts": {{{keys.replace(",", ": [...],")}: [...]}}}}')}]


def check(names):
    want = set(names.values())

    def f(p):
        if not isinstance(p, dict) or set(p) != {"passage_clues", "prior_facts"}:
            raise ValueError('expected {"passage_clues": {...}, "prior_facts": {...}}')
        for k in ("passage_clues", "prior_facts"):
            if not isinstance(p[k], dict) or set(p[k]) != want:
                raise ValueError(f"{k} must have exactly the keys {sorted(want)}")
            for v in p[k].values():
                if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                    raise ValueError(f"{k} values must be lists of strings")
        return p
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    principals = json.load(open(DATA / "principals.json"))
    reps = {(r["condition"], r["book"], r["boundary"], r["label"]): r["text"] for r in read_jsonl(DATA / f"reps_{args.split}.jsonl")
            if r["condition"] in ("legacy_r6000", "charmem")}
    items = read_jsonl(DATA / f"items_{args.split}.jsonl")[args.shard::args.nshards]
    out = DATA / "oracle" / f"{args.split}_{args.shard:02d}of{args.nshards:02d}.jsonl"
    done = {r["item_id"] for r in read_jsonl(out)}
    lock = threading.Lock()

    def one(it):
        if it["item_id"] in done:
            return
        book, b, labels = it["book"], it["chapter_index"], it["labels"]
        names = {l: principals[book]["eval_names"][l][str(b)] for l in labels}
        materials = {l: "\n\n".join(t for t in (reps.get(("legacy_r6000", book, b, l)), reps.get(("charmem", book, b, l))) if t)
                     for l in labels}
        try:
            p, meta = llm.ask(messages(it["original"], names, materials), check(names), max_tokens=12000)
        except Exception as e:
            print("FAILED", it["item_id"], str(e)[:200], flush=True)
            return
        by_name = {v: k for k, v in names.items()}
        rec = {"item_id": it["item_id"], "passage_clues": {}, "prior_facts": {}, "dropped_nonverbatim": 0}
        for l in labels:
            rec["passage_clues"][l] = p["passage_clues"][names[l]]
            src = norm(materials[l])
            kept = [x for x in p["prior_facts"][names[l]] if x.strip() and norm(x) in src]
            rec["dropped_nonverbatim"] += len(p["prior_facts"][names[l]]) - len(kept)
            rec["prior_facts"][l] = kept
        with lock:
            append_jsonl(out, rec)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, items))
    print("finished", out, flush=True)


def export():
    """Collect oracle records into reps_item_<split>.jsonl (conditions oracle_passage, oracle_prior)."""
    for split in ("test", "val", "train"):
        recs = [r for f in sorted((DATA / "oracle").glob(f"{split}_*.jsonl")) for r in read_jsonl(f)]
        with open(DATA / f"reps_item_{split}.jsonl", "w") as f:
            for r in recs:
                for l in r["passage_clues"]:
                    for cond, key in (("oracle_passage", "passage_clues"), ("oracle_prior", "prior_facts")):
                        text = "\n".join(f"- {x}" for x in r[key][l]) or "- No relevant earlier information."
                        f.write(json.dumps({"condition": cond, "item_id": r["item_id"], "label": l, "text": text}, ensure_ascii=False) + "\n")
        print(split, len(recs), "passages")


if __name__ == "__main__":
    export() if sys.argv[1:] == ["export"] else main()
