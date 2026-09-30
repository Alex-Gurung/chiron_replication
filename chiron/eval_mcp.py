"""Masked-character prediction with a local vLLM server (Qwen3-4B-Instruct by default). Stdlib only.

For every item x condition x character order x target character, asks which [CHAR i] the target is
and records the next-token probabilities of "0", "1", "2" (temperature 0, one token). Character
order = order of the representation blocks in the prompt; all 6 orders by default.
Conditions come from data/reps_<split>.jsonl (condition, book, boundary, label, text) plus:
  noinfo         names only
  book           the novel's chapters before the section's chapter (no per-character blocks)
  book_last<k>   the last k words of that text; more variants (last k chapters, the chapter so far) in book_context
  plot_<kind>_<n>[@last<k>]  a gpt-oss plot summary of the story so far (gen_plot.py), optionally its last k words
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
from urllib import error as urlerror, request as urlrequest

sys.path.insert(0, os.path.dirname(__file__))
from common import COHORTS, DATA, OUT, REPO, append_jsonl, clean_text, load_chapters, read_jsonl  # noqa: E402
from build_items import alias_regex  # noqa: E402


def rename(text, a, b, names_a, names_b, aliases):
    """Swap the two characters' names inside a representation: every alias of a -> b's name and vice versa."""
    ra, rb = alias_regex(aliases[a]), alias_regex(aliases[b])
    spans = sorted([(m.start(), m.end(), names_b) for m in ra.finditer(text)] +
                   [(m.start(), m.end(), names_a) for m in rb.finditer(text)], key=lambda x: (x[0], -x[1]))
    out, pos = [], 0
    for s, e, rep in spans:
        if s >= pos:
            out += [text[pos:s], rep]
            pos = e
    return "".join(out + [text[pos:]])

API = os.environ.get("CHIRON_API_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("CHIRON_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
INTRO = {3: ("Below is information about three characters from a novel, followed by a passage from a later part of the "
             "same novel. In the passage, each of these characters' names has been replaced with an ID: [CHAR 0], "
             "[CHAR 1] or [CHAR 2]. Each ID stands for exactly one of the three characters."),
         2: ("Below is information about two characters from a novel, followed by a passage from a later part of the "
             "same novel. In the passage, each of these characters' names has been replaced with an ID: [CHAR 0] or "
             "[CHAR 1]. Each ID stands for exactly one of the two characters.")}
DIGITS = {3: "0, 1 or 2", 2: "0 or 1"}


def words(text, k, last=False):
    w = text.split()
    return " ".join(w[-k:] if last else w[:k]) if len(w) > k else text


BOOK = re.compile(r"book(?:_(last|noprev)(\d+)|_prevonly|_ch(\d+)(p?)|_prefix)?$")


def load_prefixes(split):
    """(book, chapter, chunk) -> the passage's own chapter up to the passage (ncp_cohorts_v2 chapter_prefix)."""
    return {(r["story_id"], r["chapter_index"], r["chunk_index"]): clean_text(r.get("chapter_prefix") or "")
            for r in map(json.loads, open(COHORTS / f"{split}_examples.jsonl"))}


def load_plots(split):
    """(condition, book, boundary) -> plot summary of chapters < boundary (gen_plot.py export)."""
    return {(r["condition"], r["book"], r["boundary"]): r["text"] for r in read_jsonl(DATA / f"plot_{split}.jsonl")}


def book_context(cond, it, chapters, prefixes, plots=None):
    """Book text for the book conditions (shown instead of per-character blocks). b = the passage's chapter.
    book: chapters < b; book_last<k>: its last k words; book_noprev<k>: last k words of chapters < b-1;
    book_prevonly: chapter b-1; book_ch<k>: chapters b-k .. b-1; book_prefix: chapter b up to the passage;
    book_ch<k>p: chapters b-k .. b-1 plus chapter b up to the passage."""
    b = it["chapter_index"]
    if cond.startswith("plot_"):                           # plot_<kind>_<target>[@last<k>]: a plot summary, optionally its last k words
        base, k = (cond.split("@") + [None])[:2]
        t = plots.get((base, it["book"], b))
        return None if t is None else ("Summary of the novel so far", words(t, int(k[4:]), last=True) if k else t)
    m = BOOK.match(cond)
    text = lambda lo, hi: "\n\n".join(clean_text(ch["chapter_text_normalized"]) for ch in chapters[it["book"]] if lo <= ch["chapter_index"] < hi)
    prefix = lambda: prefixes[(it["book"], b, it["chunk_index"])]
    if cond == "book_prefix":
        return prefix()
    if cond == "book_prevonly":
        return text(b - 1, b)
    if m.group(3):
        return "\n\n".join(x for x in (text(b - int(m.group(3)), b), prefix() if m.group(4) else "") if x)
    full = text(0, b - 1 if m.group(1) == "noprev" else b)
    return words(full, int(m.group(2)), last=True) if m.group(1) else full


def prompt(item, names, blocks, book_text, target):
    n = len(item["order"])
    parts = [INTRO[n]]
    if book_text is not None:
        head, text = book_text if isinstance(book_text, tuple) else ("The novel so far", book_text)
        parts.append(f"# {head}\n\n{text}")
    if blocks is None:
        parts.append("# Characters\n\n" + "\n".join(f"- {names[l]}" for l in item["order"]))
    else:
        parts.append("# Character information\n\n" + "\n\n".join(f"## {names[l]}\n\n{blocks[l].strip()}" for l in item["order"]))
    parts.append("# Passage\n\n" + item["masked"])
    parts.append(f"# Question\n\nWhich ID in the passage is {names[target]}? Answer with only the digit {DIGITS[n]}.")
    return "\n\n".join(parts)


PREFIX = os.environ.get("CHIRON_ANSWER_PREFIX", "")   # e.g. "[CHAR " forces the next token to be the id digit
THINKING = os.environ.get("CHIRON_THINKING")          # "0" turns a hybrid model's thinking off via its chat template


BASE = os.environ.get("CHIRON_BASE") == "1"            # base model: raw completion ending in "Answer: [CHAR "


def score(text, n=3):
    if BASE:
        body = json.dumps({"model": MODEL, "prompt": text + "\n\nAnswer: [CHAR ", "max_tokens": 1, "temperature": 0.0,
                           "logprobs": 20}).encode()
        req = urlrequest.Request(API + "/completions", data=body, headers={"Content-Type": "application/json"})
        with urlrequest.urlopen(req, timeout=3600) as r:
            resp = json.loads(r.read().decode())
        top = resp["choices"][0]["logprobs"]["top_logprobs"][0]
        lp = {str(d): -math.inf for d in range(n)}
        for tok, v in top.items():
            if tok.strip() in lp:
                lp[tok.strip()] = max(lp[tok.strip()], v)
        return lp, max(top, key=top.get), resp.get("usage", {}).get("prompt_tokens")
    msgs = [{"role": "user", "content": text}]
    extra = {}
    if PREFIX:
        msgs.append({"role": "assistant", "content": PREFIX})
        extra = {"continue_final_message": True, "add_generation_prompt": False}
    if THINKING is not None:
        extra["chat_template_kwargs"] = {"enable_thinking": THINKING == "1"}
    body = json.dumps({"model": MODEL, "messages": msgs, "max_tokens": 1, "temperature": 0.0,
                       "logprobs": True, "top_logprobs": 20, **extra}).encode()
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
    lp = {str(d): -math.inf for d in range(n)}
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
    ap.add_argument("--rotations", action="store_true", help="the n cyclic rotations of the blocks instead of permutations")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--tag", default="")
    ap.add_argument("--items", default="", help="items file stem, default items_<split> (e.g. items_test_pron)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()
    stem = args.items or f"items_{args.split}"
    items = read_jsonl(DATA / f"{stem}.jsonl")[args.shard::args.nshards]
    suffix = f".s{args.shard}of{args.nshards}" if args.nshards > 1 else ""
    principals = json.load(open(DATA / "principals.json"))
    reps = {}
    for r in read_jsonl(DATA / f"reps_{args.split}.jsonl"):
        reps[(r["condition"], r["book"], r["boundary"], r["label"])] = r["text"]
    item_reps = {(r["condition"], r["item_id"], r["label"]): r["text"] for r in read_jsonl(DATA / f"reps_item_{args.split}.jsonl")}
    chapters = load_chapters() if any(c.startswith("book") for c in args.conditions) else None
    plots = load_plots(args.split) if any(c.startswith("plot_") for c in args.conditions) else None
    prefixes = load_prefixes(args.split) if any(c == "book_prefix" or re.match(r"book_ch\d+p$", c) for c in args.conditions) else None
    model_tag = MODEL.split("/")[-1] + args.tag + ("_prefix" if PREFIX else "") + {None: "", "0": "_nothink", "1": "_think"}[THINKING]
    aliases = {}
    for cond in args.conditions:
        base = OUT / "eval" / model_tag / stem / f"{cond.replace(':', '_')}{suffix}"
        out = base.with_name(base.name + f".r{os.environ.get('JOB_NAME', os.getpid())}.jsonl")   # one file per run: no shared appends
        name = cond.replace(':', '_')                                     # any earlier run of this condition, whatever its sharding
        done = {(r["item_id"], tuple(r["order"]), r["target"])
                for f in [base.parent / f"{name}.jsonl", *base.parent.glob(f"{name}.*jsonl")] if not f.name.endswith(".errors.jsonl")
                for r in read_jsonl(f)}
        jobs, missing = [], 0
        for it in items:
            book, b, labels = it["book"], it["chapter_index"], it["labels"]
            names = {l: principals[book]["eval_names"][l][str(b)] for l in labels}
            book_text, blocks = None, None
            if cond.startswith(("book", "plot_")):
                book_text = book_context(cond, it, chapters, prefixes, plots)
                if book_text is None:
                    missing += 1
                    continue
            elif cond != "noinfo":
                base, k = (cond.split("@") + [None])[:2]
                swapname = base.startswith("swapname:")
                swap = base.startswith("swap:") or swapname
                base = base.split(":")[-1]
                src = {l: labels[(i + 1) % len(labels)] if swap else l for i, l in enumerate(labels)}
                blocks = {l: item_reps.get((base, it["item_id"], src[l]), reps.get((base, book, b, src[l]))) for l in labels}
                if any(t is None for t in blocks.values()):     # never score a missing rep as "no information"
                    missing += 1
                    continue
                if swapname:
                    al = aliases.setdefault(book, json.load(open(REPO / "aliases" / f"{book}.json"))["principals"])
                    blocks = {l: rename(t, src[l], l, names[src[l]], names[l], al) for l, t in blocks.items()}
                if k:
                    blocks = {l: words(t, int(k)) for l, t in blocks.items()}
            if blocks is None:
                orders = [tuple(labels)]
            elif args.rotations:
                orders = [tuple(labels[i:] + labels[:i]) for i in range(len(labels))]
            else:
                orders = list(itertools.permutations(labels))[:args.orders]
            for order in orders:
                for target in labels:
                    if (it["item_id"], order, target) not in done:
                        jobs.append((it, names, blocks, book_text, order, target))
        jobs.sort(key=lambda j: (j[0]["book"], j[0]["chapter_index"], j[0]["item_id"], j[4]))   # shared prefixes adjacent
        print(f"{cond}: {len(done)} done, {len(jobs)} to score, {missing} items skipped for missing reps", flush=True)
        if missing == len(items):
            print(f"{cond}: ERROR no representations found; skipping condition", flush=True)
            continue
        lock = threading.Lock()

        def one(j):
            it, names, blocks, book_text, order, target = j
            text = prompt({**it, "order": order}, names, blocks, book_text, target)
            try:
                lp, top, ntok = score(text, len(order))
            except urlerror.HTTPError as e:              # 400 = prompt longer than the served context
                with lock:
                    append_jsonl(out.with_suffix(".errors.jsonl"), {"item_id": it["item_id"], "order": list(order),
                                                                    "target": target, "error": f"HTTP {e.code}"})
                return
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
