"""Masked-character prediction with a local vLLM server (Qwen3-4B-Instruct by default). Stdlib only.

For every item x condition x character order x target character, asks which [CHAR i] the target is
and records the next-token probabilities of "0", "1", "2" (temperature 0, one token). Character
order = order of the representation blocks in the prompt; all 6 orders by default.
Conditions come from data/reps_<split>.jsonl (condition, book, boundary, label, text) plus:
  noinfo         names only
  book           the novel's chapters before the section's chapter (no per-character blocks)
  book_last<k>   the last k words of that text
  <cond>@<k>     representation truncated to its first k words
  swap:<cond>    each character gets the next principal's representation (cyclic)
Output: outputs/eval/<model>/<split>/<condition>.jsonl, resumable.
"""
import argparse
import itertools
import json
import math
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib import request as urlrequest

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, OUT, append_jsonl, clean_text, load_chapters, read_jsonl  # noqa: E402

API = os.environ.get("CHIRON_API_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("CHIRON_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
INTRO = ("Below is information about three characters from a novel, followed by a passage from a later part of the "
         "same novel. In the passage, each of these characters' names has been replaced with an ID: [CHAR 0], "
         "[CHAR 1] or [CHAR 2]. Each ID stands for exactly one of the three characters.")


def words(text, k, last=False):
    w = text.split()
    return " ".join(w[-k:] if last else w[:k]) if len(w) > k else text


def prompt(item, names, blocks, book_text, target):
    parts = [INTRO]
    if book_text is not None:
        parts.append("# The novel so far\n\n" + book_text)
    if blocks is None:
        parts.append("# Characters\n\n" + "\n".join(f"- {names[l]}" for l in item["order"]))
    else:
        parts.append("# Character information\n\n" + "\n\n".join(f"## {names[l]}\n\n{blocks[l].strip()}" for l in item["order"]))
    parts.append("# Passage\n\n" + item["masked"])
    parts.append(f"# Question\n\nWhich ID in the passage is {names[target]}? Answer with only the digit 0, 1 or 2.")
    return "\n\n".join(parts)


def score(text):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": 1,
                       "temperature": 0.0, "logprobs": True, "top_logprobs": 20}).encode()
    req = urlrequest.Request(API + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    for k in range(5):
        try:
            with urlrequest.urlopen(req, timeout=3600) as r:
                resp = json.loads(r.read().decode())
            break
        except Exception as e:
            if k == 4:
                raise
    top = resp["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    lp = {d: -math.inf for d in "012"}
    for t in top:
        tok = t["token"].strip()
        if tok in lp:
            lp[tok] = max(lp[tok], t["logprob"])
    return lp, top[0]["token"], resp.get("usage", {}).get("prompt_tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--orders", type=int, default=6)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    items = read_jsonl(DATA / f"items_{args.split}.jsonl")
    principals = json.load(open(DATA / "principals.json"))
    reps = {}
    for r in read_jsonl(DATA / f"reps_{args.split}.jsonl"):
        reps[(r["condition"], r["book"], r["boundary"], r["label"])] = r["text"]
    chapters = load_chapters() if any(c.startswith("book") for c in args.conditions) else None
    model_tag = MODEL.split("/")[-1] + args.tag
    for cond in args.conditions:
        out = OUT / "eval" / model_tag / args.split / f"{cond.replace(':', '_')}.jsonl"
        done = {(r["item_id"], tuple(r["order"]), r["target"]) for r in read_jsonl(out)}
        jobs = []
        for it in items:
            book, b, labels = it["book"], it["chapter_index"], it["labels"]
            names = {l: principals[book]["eval_names"][l][str(b)] for l in labels}
            book_text, blocks = None, None
            if cond.startswith("book"):
                full = "\n\n".join(clean_text(ch["chapter_text_normalized"]) for ch in chapters[book] if ch["chapter_index"] < b)
                book_text = words(full, int(cond[9:]), last=True) if cond.startswith("book_last") else full
            elif cond != "noinfo":
                base, k = (cond.split("@") + [None])[:2]
                swap = base.startswith("swap:")
                base = base.replace("swap:", "")
                src = {l: labels[(i + 1) % len(labels)] if swap else l for i, l in enumerate(labels)}
                blocks = {l: reps.get((base, book, b, src[l]), "No information.") for l in labels}
                if k:
                    blocks = {l: words(t, int(k)) for l, t in blocks.items()}
            orders = list(itertools.permutations(labels))[:args.orders] if blocks is not None else [tuple(labels)]
            for order in orders:
                for target in labels:
                    if (it["item_id"], order, target) not in done:
                        jobs.append((it, names, blocks, book_text, order, target))
        jobs.sort(key=lambda j: (j[0]["book"], j[0]["chapter_index"], j[0]["item_id"], j[4]))   # shared prefixes adjacent
        print(f"{cond}: {len(done)} done, {len(jobs)} to score", flush=True)
        lock = threading.Lock()

        def one(j):
            it, names, blocks, book_text, order, target = j
            text = prompt({**it, "order": order}, names, blocks, book_text, target)
            lp, top, ntok = score(text)
            rec = {"item_id": it["item_id"], "book": it["book"], "boundary": it["chapter_index"], "order": list(order),
                   "target": target, "answer": it["answer"][target], "logprobs": lp, "top_token": top,
                   "prompt_tokens": ntok}
            with lock:
                append_jsonl(out, rec)

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(one, jobs))
        print(f"{cond}: finished", flush=True)


if __name__ == "__main__":
    main()
