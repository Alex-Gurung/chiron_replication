"""CHIRON-style statements with gpt-oss-120b, one shard of snippets.

For every (snippet, principal): one call answers CHIRON's 8 questions as lists of single-claim
sentences (generation + simplification) and says whether the character is present; one call then
rates every claim on the paper's 1-5 entailment scale against the snippet. Sheets keep rating 5.
Output: outputs/chiron/shards/<shard>.jsonl, one record per (snippet, principal); resumable.
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from common import DATA, OUT, append_jsonl, read_jsonl
import llm

QUESTIONS = {
    "dialogue": ("Dialogue", "What, if anything, have we learned about how this character speaks from this snippet?"),
    "physical": ("Physical/Personality", "What, if any, physical descriptions of this character are in this snippet?"),
    "personality": ("Physical/Personality", "What, if any, descriptions of this character's personality are in this snippet?"),
    "facts": ("Knowledge", "What, if any, factual information is given about this character in this snippet?"),
    "learned": ("Knowledge", "What, if any, information has this character learned in this snippet?"),
    "goals_gained": ("Goals", "What, if any, goals does this character gain in this snippet that they wish to accomplish in the future?"),
    "goals_completed": ("Goals", "What, if any, goals does this character complete in this snippet?"),
    "motivation_change": ("Goals", "How, if at all, does this character's internal motivations change in this snippet?"),
}
ROLE = ("You are a helpful and expert writing assistant. You will be given a section of a story or screenplay. "
        "Please answer the following questions about the character learned in this story section.")


def gen_messages(snippet, name):
    qs = "\n".join(f"- {k}: {q}" for k, (_, q) in QUESTIONS.items())
    schema = ", ".join(f'"{k}": ["..."]' for k in QUESTIONS)
    return [{"role": "system", "content": ROLE},
            {"role": "user", "content": (
                f"Story Section:\n{snippet}\n\nCharacter: {name}\n\n"
                f"First decide whether {name} appears in, or is directly referred to in, this story section. "
                f"Then answer each question about {name} based only on this story section. Answer with short, "
                "simple sentences with no dependent clauses or transition words; each sentence must state a single "
                "claim and name the character rather than using a pronoun. If the section gives no information for "
                "a question, give an empty list. If the character is not present, every list must be empty.\n\n"
                f"Questions:\n{qs}\n\n"
                f'Return only JSON: {{"present": true or false, "answers": {{{schema}}}}}')}]


def check_gen(p):
    if set(p) != {"present", "answers"} or type(p["present"]) is not bool or set(p["answers"]) != set(QUESTIONS):
        raise ValueError('expected {"present": bool, "answers": {<the 8 question keys>}}')
    for k, v in p["answers"].items():
        if not isinstance(v, list) or not all(isinstance(s, str) for s in v):
            raise ValueError(f"answers.{k} must be a list of strings")
        p["answers"][k] = [s.strip() for s in v if s.strip()]
    if not p["present"] and any(p["answers"].values()):
        raise ValueError("present is false but some answers are non-empty")
    return p


def rate_messages(snippet, name, claims):
    listing = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    return [{"role": "system", "content": "You verify statements about story characters against a story section."},
            {"role": "user", "content": (
                f"Story Section:\n{snippet}\n\nCharacter: {name}\n\nStatements:\n{listing}\n\n"
                "Rate each statement by whether it is entailed by the story section alone, using this scale:\n"
                "1 = entirely unsupported by the snippet\n2 = largely contradicted by the snippet\n"
                "3 = ambiguous in its relationship with the snippet, including statements too vague to verify, "
                "statements that refer to unspecified people or objects, and statements that make no claim about "
                "the character\n4 = likely true based on the snippet, but with a minor unclear or unsupported part\n"
                "5 = entirely supported by the snippet\n\n"
                'Return only JSON: {"ratings": [{"id": 0, "rating": 5}, ...]} with exactly one entry per statement.')}]


def check_rate(n):
    def f(p):
        rs = p.get("ratings") if isinstance(p, dict) else None
        if not isinstance(rs, list):
            raise ValueError('expected {"ratings": [...]}')
        got = {}
        for r in rs:
            if not isinstance(r, dict) or type(r.get("id")) is not int or r.get("rating") not in (1, 2, 3, 4, 5):
                raise ValueError("each rating needs an integer id and a rating 1-5")
            got[r["id"]] = r["rating"]
        if set(got) != set(range(n)):
            raise ValueError(f"need exactly one rating for each id 0..{n - 1}")
        return [got[i] for i in range(n)]
    return f


def process(snip, label, name):
    gen, m1 = llm.ask(gen_messages(snip["text"], name), check_gen, max_tokens=8000)
    claims = [(k, c) for k in QUESTIONS for c in gen["answers"][k]]
    ratings, m2 = [], {}
    if claims:
        ratings, m2 = llm.ask(rate_messages(snip["text"], name, [c for _, c in claims]), check_rate(len(claims)),
                              max_tokens=8000)
    return {"item_id": f"{snip['snippet_id']}::{label}", "snippet_id": snip["snippet_id"], "book": snip["book"],
            "chapter_index": snip["chapter_index"], "idx": snip["idx"], "label": label, "name": name,
            "present": gen["present"],
            "claims": [{"q": k, "category": QUESTIONS[k][0], "text": c, "rating": r}
                       for (k, c), r in zip(claims, ratings)],
            "meta": {"gen": m1, "rate": m2}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    principals = json.load(open(DATA / "principals.json"))
    snips = [s for s in read_jsonl(DATA / "snippets.jsonl") if s["book"] in args.books]
    snips = snips[args.shard::args.nshards]
    tag = f"{'-'.join(args.books) if len(args.books) <= 4 else f'{len(args.books)}books'}_{args.shard:03d}of{args.nshards:03d}"
    out = OUT / "chiron" / "shards" / f"{tag}.jsonl"
    done = {r["item_id"] for r in read_jsonl(out)}
    items = [(s, l) for s in snips for l in principals[s["book"]]["labels"] if f"{s['snippet_id']}::{l}" not in done]
    if args.limit:
        items = items[:args.limit]
    print(f"{tag}: {len(done)} done, {len(items)} to do", flush=True)
    lock, counts = threading.Lock(), {"ok": 0, "fail": 0}

    def one(item):
        s, l = item
        name = principals[s["book"]]["gen_names"][l][s["chapter_index"]]
        try:
            rec = process(s, l, name)
            with lock:
                append_jsonl(out, rec)
                counts["ok"] += 1
        except Exception as e:
            with lock:
                append_jsonl(out.with_suffix(".errors.jsonl"), {"item_id": f"{s['snippet_id']}::{l}", "error": str(e)[:2000]})
                counts["fail"] += 1
        n = counts["ok"] + counts["fail"]
        if n % 50 == 0:
            print(f"{tag}: {n}/{len(items)} ok={counts['ok']} fail={counts['fail']}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(one, items))
    print(f"{tag}: finished ok={counts['ok']} fail={counts['fail']}", flush=True)
    sys.exit(1 if counts["fail"] else 0)


if __name__ == "__main__":
    main()
