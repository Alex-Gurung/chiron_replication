"""Masked-character prediction with reasoning: the model thinks, then gives the full name -> id mapping.

One generation per passage and block rotation (the paper's framing: match each character to its id).
Thinking is on (CHIRON_THINKING=1); sampling follows Qwen's thinking-mode settings. The final answer must
end with a JSON object {"<name>": <id>, ...}; one retry with the parse error, then the passage is marked
invalid and every character in it counts as wrong. Records use eval_mcp's schema (one per character, with
a one-hot "logprobs" over the ids) so score.py / analyze.py read them unchanged.
Output: outputs/eval/<model>_think/<items>/<condition>[.s<k>of<n>].r<job>.jsonl, resumable.
"""
import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib import request as urlrequest

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, OUT, append_jsonl, read_jsonl  # noqa: E402
import eval_mcp as E  # noqa: E402

API = os.environ.get("CHIRON_API_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("CHIRON_MODEL", "Qwen/Qwen3.8-27B")
DIGITS = {3: "0, 1 or 2", 2: "0 or 1"}


def question(names, order):
    example = ", ".join(f'"{names[l]}": ?' for l in order)          # placeholders: a concrete example would prime a copy
    return (f"# Question\n\nFor each of the {len(order)} characters, decide which ID ({DIGITS[len(order)]}) stands for them "
            f"in the passage. Each ID is used exactly once. End your reply with a JSON object mapping each character's "
            f"name to its ID, in the form {{{example}}}.")


def ask(text, names, order):
    msgs = [{"role": "user", "content": text}]
    for attempt in range(2):
        body = json.dumps({"model": MODEL, "messages": msgs, "max_tokens": 32768, "temperature": 0.6, "top_p": 0.95,
                           "top_k": 20, "chat_template_kwargs": {"enable_thinking": True}}).encode()
        req = urlrequest.Request(API + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
        with urlrequest.urlopen(req, timeout=7200) as r:
            resp = json.loads(r.read().decode())
        msg = resp["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        usage = resp.get("usage", {})
        try:
            found = re.findall(r"\{[^{}]*\}", content)
            if not found:
                raise ValueError("no JSON object at the end of the reply")
            raw = json.loads(found[-1])
            want = {names[l]: l for l in order}
            got = {want[k]: v for k, v in raw.items() if k in want}
            if set(got) != set(order) or sorted(got.values()) != list(range(len(order))):
                raise ValueError(f"need each of {list(want)} mapped to a distinct id in {DIGITS[len(order)]}")
            return got, usage, attempt + 1, len(msg.get("reasoning_content") or msg.get("reasoning") or "")
        except (ValueError, json.JSONDecodeError, TypeError) as e:
            msgs = [*msgs, {"role": "assistant", "content": content[-4000:]},
                    {"role": "user", "content": f"Your answer could not be read: {e}. Reply with only the JSON object."}]
    return None, usage, 2, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--items", default="")
    ap.add_argument("--conditions", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    stem = args.items or f"items_{args.split}"
    items = read_jsonl(DATA / f"{stem}.jsonl")[args.shard::args.nshards]
    if args.limit:
        items = items[:args.limit]
    suffix = f".s{args.shard}of{args.nshards}" if args.nshards > 1 else ""
    principals = json.load(open(DATA / "principals.json"))
    reps = {(r["condition"], r["book"], r["boundary"], r["label"]): r["text"] for r in read_jsonl(DATA / f"reps_{args.split}.jsonl")}
    chapters = E.load_chapters() if any(c.startswith("book") for c in args.conditions) else None
    aliases = {}
    model_tag = MODEL.split("/")[-1] + args.tag + "_think"
    for cond in args.conditions:
        base = OUT / "eval" / model_tag / stem / f"{cond.replace(':', '_')}{suffix}"
        out = base.with_name(base.name + f".r{os.environ.get('JOB_NAME', os.getpid())}.jsonl")
        done = {(r["item_id"], tuple(r["order"])) for f in base.parent.glob(base.name + ".r*.jsonl") for r in read_jsonl(f)}
        jobs, missing = [], 0
        for it in items:
            book, b, labels = it["book"], it["chapter_index"], it["labels"]
            names = {l: principals[book]["eval_names"][l][str(b)] for l in labels}
            book_text, blocks = None, None
            if cond.startswith("book"):
                full = "\n\n".join(E.clean_text(ch["chapter_text_normalized"]) for ch in chapters[book] if ch["chapter_index"] < b)
                book_text = E.words(full, int(cond[9:]), last=True) if cond.startswith("book_last") else full
            elif cond != "noinfo":
                basec, k = (cond.split("@") + [None])[:2]
                swapname = basec.startswith("swapname:")
                swap = basec.startswith("swap:") or swapname
                basec = basec.split(":")[-1]
                src = {l: labels[(i + 1) % len(labels)] if swap else l for i, l in enumerate(labels)}
                blocks = {l: reps.get((basec, book, b, src[l])) for l in labels}
                if any(t is None for t in blocks.values()):
                    missing += 1
                    continue
                if swapname:
                    al = aliases.setdefault(book, json.load(open(E.REPO / "aliases" / f"{book}.json"))["principals"])
                    blocks = {l: E.rename(t, src[l], l, names[src[l]], names[l], al) for l, t in blocks.items()}
                if k:
                    blocks = {l: E.words(t, int(k)) for l, t in blocks.items()}
            orders = [tuple(labels)] if blocks is None else [tuple(labels[i:] + labels[:i]) for i in range(len(labels))]
            for order in orders:
                if (it["item_id"], order) not in done:
                    jobs.append((it, names, blocks, book_text, order))
        print(f"{cond}: {len(done)} done, {len(jobs)} to generate, {missing} items skipped for missing reps", flush=True)
        lock = threading.Lock()

        def one(j):
            it, names, blocks, book_text, order = j
            text = E.prompt({**it, "order": order}, names, blocks, book_text, order[0])
            text = text[:text.rindex("# Question")] + question(names, order)
            got, usage, attempts, rlen = ask(text, names, order)
            with lock:
                for t in order:
                    pred = None if got is None else got[t]
                    lp = {str(d): (0.0 if pred == d else -1e9) for d in range(len(order))}
                    append_jsonl(out, {"item_id": it["item_id"], "book": it["book"], "boundary": it["chapter_index"],
                                       "order": list(order), "target": t, "answer": it["answer"][t], "logprobs": lp,
                                       "top_token": "" if pred is None else str(pred), "invalid": got is None,
                                       "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                                       "attempts": attempts, "reasoning_chars": rlen})

        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(one, jobs))
        print(f"{cond}: finished", flush=True)


if __name__ == "__main__":
    main()
