"""How much more likely does character information make the real next passage? (per-token NLL under a base LM)

Prompt (raw text, no chat template):
  [Character notes for the novel's main characters: one block per principal]      (unless rep == "names")
  [The story so far: the STORY_WORDS words right before the passage]                (context "story" only)
  The next passage of the novel:\n\n<the real passage, names unmasked>
Scores the passage tokens with the completions endpoint (echo + logprobs). Main-set passages.
Output: outputs/ppl/<model>/<split>/<context>__<rep>.jsonl with {item_id, book, nll, tokens}; resumable.
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib import request as urlrequest

sys.path.insert(0, os.path.dirname(__file__))
from common import COHORTS, DATA, OUT, append_jsonl, clean_text, load_chapters, read_jsonl  # noqa: E402

API = os.environ.get("CHIRON_API_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("CHIRON_MODEL", "Qwen/Qwen3.5-9B-Base")
STORY_WORDS = 4000
REPS = ["names", "v2", "charmem", "chiron_r2000", "chiron", "summary", "legacy", "legacy_full"]


CHAT = os.environ.get("CHIRON_CHAT") == "1"            # instruct models: writing prompt as the user turn, passage as the reply
THINKING = os.environ.get("CHIRON_THINKING")


def post(path, payload):
    req = urlrequest.Request(API.rsplit("/v1", 1)[0] + path if path == "/tokenize" else API + path,
                             data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=3600) as r:
        return json.loads(r.read().decode())


def score_chat(prefix, target):
    """Token ids of the chat prompt (with generation prompt) + the passage; score the passage tokens."""
    msg = {"model": MODEL, "messages": [{"role": "user", "content": prefix.rstrip()}], "add_generation_prompt": True}
    if THINKING is not None:
        msg["chat_template_kwargs"] = {"enable_thinking": THINKING == "1"}
    head = post("/tokenize", msg)["tokens"]
    tail = post("/tokenize", {"model": MODEL, "prompt": target, "add_special_tokens": False})["tokens"]
    lp = post("/completions", {"model": MODEL, "prompt": head + tail, "max_tokens": 1, "temperature": 0.0,
                               "echo": True, "logprobs": 1})["choices"][0]["logprobs"]["token_logprobs"]
    vals = [v for v in lp[len(head):len(head) + len(tail)] if v is not None]
    return -sum(vals), len(vals)


def score(prefix, target):
    if CHAT:
        return score_chat(prefix.replace("\n\nThe next passage of the novel:\n\n", "\n\nWrite the next passage of the novel."), target)
    body = json.dumps({"model": MODEL, "prompt": prefix + target, "max_tokens": 1, "temperature": 0.0,
                       "echo": True, "logprobs": 1}).encode()
    req = urlrequest.Request(API + "/completions", data=body, headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=3600) as r:
        lp = json.loads(r.read().decode())["choices"][0]["logprobs"]
    start = len(prefix)
    vals = [v for off, v in zip(lp["text_offset"], lp["token_logprobs"]) if v is not None and start <= off < start + len(target)]
    return -sum(vals), len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--context", choices=["none", "story"], required=True)
    ap.add_argument("--reps", nargs="+", default=REPS)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    items = read_jsonl(DATA / f"items_{args.split}.jsonl")
    principals = json.load(open(DATA / "principals.json"))
    reps = {(r["condition"], r["book"], r["boundary"], r["label"]): r["text"] for r in read_jsonl(DATA / f"reps_{args.split}.jsonl")}
    story = {}
    if args.context == "story":
        chapters = load_chapters()
        prefix = {}
        want = {(it["book"], it["chapter_index"], it["chunk_index"]) for it in items}
        for line in open(COHORTS / f"{args.split}_examples.jsonl"):
            r = json.loads(line)
            k = (r["story_id"], r["chapter_index"], r["chunk_index"])
            if k in want:
                prefix[k] = clean_text(r.get("chapter_prefix") or "")
        for it in items:
            before = " ".join(clean_text(ch["chapter_text_normalized"]) for ch in chapters[it["book"]] if ch["chapter_index"] < it["chapter_index"])
            story[it["item_id"]] = " ".join((before + " " + prefix[(it["book"], it["chapter_index"], it["chunk_index"])]).split()[-STORY_WORDS:])
    model_tag = MODEL.split("/")[-1] + ("_chat" if CHAT else "")
    for rep in args.reps:
        out = OUT / "ppl" / model_tag / args.split / f"{args.context}__{rep}.jsonl"
        done = {r["item_id"] for r in read_jsonl(out)}
        lock = threading.Lock()

        def one(it):
            if it["item_id"] in done:
                return
            book, b = it["book"], it["chapter_index"]
            names = {l: principals[book]["eval_names"][l][str(b)] for l in it["labels"]}
            parts = []
            if rep != "names":
                blocks = {l: reps.get((rep, book, b, l)) for l in it["labels"]}
                if any(v is None for v in blocks.values()):
                    return
                parts.append("Character notes for the novel's main characters:\n\n" +
                             "\n\n".join(f"## {names[l]}\n{blocks[l].strip()}" for l in it["labels"]))
            else:
                parts.append("The novel's main characters: " + ", ".join(names.values()) + ".")
            if args.context == "story":
                parts.append("The story so far (most recent part):\n\n" + story[it["item_id"]])
            prefix = "\n\n".join(parts) + "\n\nThe next passage of the novel:\n\n"
            nll, n = score(prefix, it["original"])
            with lock:
                append_jsonl(out, {"item_id": it["item_id"], "book": book, "nll": nll, "tokens": n})

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(one, items))
        print(f"{args.context}__{rep}: finished", flush=True)


if __name__ == "__main__":
    main()
