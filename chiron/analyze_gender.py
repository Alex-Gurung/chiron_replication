"""How much of the accuracy is gender? Accuracy split by whether a character's gender is unique among the three principals.

  python3 chiron/analyze_gender.py
Each principal's gender comes from common.genders (pronouns in its v2 and charmem sheets; checked by hand).
For each passage and character: "unique" if no other principal shares its gender, "shared" if exactly one other
does, "all same" if all three do. Pooled accuracy per group for each model, set and representation.
Writes outputs/analysis_gender.json.
"""
import collections
import glob

from common import DATA, OUT, genders, read_jsonl, write_json

RUNS = [("Qwen3-4B-Instruct-2507", ""), ("Qwen3.5-9B-Base", ""), ("Qwen3.8-27B_nothink", ""), ("Qwen3.8-27B_think", ""),
        ("Qwen3.8-27B_nothink", "_short"), ("Qwen3.8-27B_think", "_short"), ("Qwen3.8-27B_nothink", "_window")]
CONDS = ["noinfo", "gender", "v2", "charmem", "summary", "legacy_full", "book_last8000", "swapname_v2"]


def main():
    g = genders(r for s in ("test", "val", "train") for r in read_jsonl(DATA / f"reps_{s}.jsonl"))
    print("genders:", collections.Counter(g.values()))
    out = {}
    for model, suf in RUNS:
        items = {it["item_id"]: it for s in ("test", "val", "train") for it in read_jsonl(DATA / f"items_{s}{suf}.jsonl")}
        group = {}
        for iid, it in items.items():
            gs = [g.get((it["book"], l), "?") for l in it["labels"]]
            for l, x in zip(it["labels"], gs):
                if x == "?" or "?" in gs:
                    continue
                k = gs.count(x)
                group[(iid, l)] = {1: "unique", 2: "shared", 3: "all same"}[k]
        for c in CONDS:
            tmp = collections.defaultdict(list)
            for f in glob.glob(str(OUT / "eval" / model / f"items_*{suf}" / f"{c}.*jsonl")) + glob.glob(str(OUT / "eval" / model / f"items_*{suf}" / f"{c}.jsonl")):
                if f.endswith(".errors.jsonl") or not any(f"/items_{s}{suf}/" in f for s in ("test", "val", "train")):
                    continue
                for r in read_jsonl(f):
                    lp = r["logprobs"]
                    tmp[(r["item_id"], r["target"])].append(int(not r.get("invalid") and max(lp, key=lp.get) == str(r["answer"])))
            by = collections.defaultdict(list)
            for k, v in tmp.items():
                if k in group:
                    by[group[k]].append(sum(v) / len(v))
            if by:
                out[f"{model}{suf}|{c}"] = {k: {"acc": sum(v) / len(v), "n": len(v)} for k, v in by.items()}
    for k, v in out.items():
        print(f"{k:45s}", "  ".join(f"{grp}: {100 * v[grp]['acc']:.1f} ({v[grp]['n']})" for grp in ("unique", "shared", "all same") if grp in v))
    write_json(OUT / "analysis_gender.json", {"genders": {f"{b}|{l}": x for (b, l), x in g.items()}, "acc": out})


if __name__ == "__main__":
    main()
