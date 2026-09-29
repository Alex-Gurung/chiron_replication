"""Memorization probe (RAM "unslop" check): does the model continue the book verbatim?

For N sections per book, the prompt is the ~500 words of book text right before the section; the model
continues for ~300 words at temperature 0 via the raw completions endpoint. A continuation is flagged
if it reproduces >5% of the true continuation's 13-grams or any verbatim run of >=20 words.
Output: outputs/memo/<model>/<book>.jsonl and a per-book summary line on stdout.
"""
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from urllib import request as urlrequest

sys.path.insert(0, os.path.dirname(__file__))
from common import OUT, clean_text, load_chapters, write_json  # noqa: E402

API = os.environ.get("CHIRON_API_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("CHIRON_MODEL", "Qwen/Qwen3-4B-Instruct-2507")


def toks(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())


def ngram_overlap(gen, ref, n=13):
    g, r = toks(gen), toks(ref)
    ref_grams = {tuple(r[i:i + n]) for i in range(len(r) - n + 1)}
    gen_grams = {tuple(g[i:i + n]) for i in range(len(g) - n + 1)}
    return len(ref_grams & gen_grams) / max(1, len(ref_grams))


def longest_run(gen, ref):
    g, r = gen.lower().split(), ref.lower().split()
    best, prev = 0, [0] * (len(r) + 1)
    for i in range(1, len(g) + 1):
        cur = [0] * (len(r) + 1)
        for j in range(1, len(r) + 1):
            if g[i - 1] == r[j - 1]:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def complete(prompt, max_tokens=400):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}).encode()
    req = urlrequest.Request(API + "/completions", data=body, headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode())["choices"][0]["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    chapters = load_chapters()
    rng = random.Random(0)
    tag = MODEL.split("/")[-1]
    jobs = []
    for book in args.books:
        words = []
        for ch in chapters[book]:
            words += clean_text(ch["chapter_text_normalized"]).split()
        starts = sorted(rng.sample(range(500, len(words) - 300), args.n))
        jobs += [(book, " ".join(words[s - 500:s]), " ".join(words[s:s + 300])) for s in starts]

    def one(j):
        book, prefix, ref = j
        gen = complete(prefix)
        ov, run = ngram_overlap(gen, ref), longest_run(gen, ref)
        return {"book": book, "overlap13": ov, "longest_run": run, "flag": ov > 0.05 or run >= 20,
                "prefix_tail": prefix[-200:], "generated": gen, "reference": ref}

    with ThreadPoolExecutor(64) as pool:
        recs = list(pool.map(one, jobs))
    summary = {}
    for book in args.books:
        rs = [r for r in recs if r["book"] == book]
        summary[book] = {"n": len(rs), "flagged": sum(r["flag"] for r in rs),
                         "max_overlap13": max(r["overlap13"] for r in rs), "max_run": max(r["longest_run"] for r in rs)}
        print(book, summary[book], flush=True)
        (OUT / "memo" / tag).mkdir(parents=True, exist_ok=True)
        with open(OUT / "memo" / tag / f"{book}.jsonl", "w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_json(OUT / "memo" / tag / "summary.json", summary)


if __name__ == "__main__":
    main()
