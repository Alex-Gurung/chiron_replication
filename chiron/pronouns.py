"""Pronoun variant of the items: third-person pronouns that refer to a principal become its mask id.

gpt-oss-120b reads the ORIGINAL section with every he/she/him/her/his/hers/himself/herself numbered,
and says for each one which principal it refers to (or "other") and whether it is possessive.
Principal pronouns become [CHAR i] (possessive: [CHAR i]'s); others are left as written. Treat results
with care: a missed principal pronoun still leaks gender.
Output: data/items_<split>_pron.jsonl (same items, `masked` replaced, `pronouns_masked` counts).
"""
import argparse
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from build_items import alias_regex
from common import DATA, REPO, read_jsonl
import llm

PRON = re.compile(r"\b(he|she|him|her|his|hers|himself|herself)\b", re.I)


def numbered(text):
    spans = list(PRON.finditer(text))
    out, pos = [], 0
    for i, m in enumerate(spans):
        out += [text[pos:m.end()], f"⟨{i}⟩"]
        pos = m.end()
    out.append(text[pos:])
    return "".join(out), spans


def messages(text, labels):
    return [{"role": "system", "content": "You resolve pronouns in fiction."},
            {"role": "user", "content": (
                f"Characters: {json.dumps(labels, ensure_ascii=False)}\n\nPassage (each third-person pronoun is "
                f"followed by a number in angle brackets):\n{text}\n\nFor every numbered pronoun, give the character "
                "from the list it refers to, or \"other\" if it refers to anyone else, and whether it is possessive "
                "(his, hers, or her used as a determiner such as 'her hand').\n"
                'Return only JSON: {"pronouns": [{"id": 0, "ref": "<character from the list or other>", "possessive": true}, ...]} '
                "with exactly one entry per number.")}]


def check(n, labels):
    def f(p):
        rows = p.get("pronouns") if isinstance(p, dict) else None
        if not isinstance(rows, list):
            raise ValueError('expected {"pronouns": [...]}')
        got = {}
        for r in rows:
            if type(r.get("id")) is not int or r.get("ref") not in [*labels, "other"] or type(r.get("possessive")) is not bool:
                raise ValueError("each entry needs an integer id, a ref from the list or 'other', and a boolean possessive")
            got[r["id"]] = r
        if set(got) != set(range(n)):
            raise ValueError(f"need exactly one entry for each id 0..{n - 1}")
        return [got[i] for i in range(n)]
    return f


def remask(item, aliases, resolved, spans):
    regs = {l: alias_regex(a) for l, a in aliases.items()}
    edits = [(m.start(), m.end(), f"[CHAR {item['answer'][l]}]") for l, rx in regs.items() for m in rx.finditer(item["original"])]
    for m, r in zip(spans, resolved):
        if r["ref"] != "other":
            edits.append((m.start(), m.end(), f"[CHAR {item['answer'][r['ref']]}]" + ("'s" if r["possessive"] else "")))
    edits.sort(key=lambda e: (e[0], -(e[1] - e[0])))
    out, pos = [], 0
    for s, e, rep in edits:
        if s < pos:
            continue
        out += [item["original"][pos:s], rep]
        pos = e
    out.append(item["original"][pos:])
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    items = read_jsonl(DATA / f"items_{args.split}.jsonl")

    def one(it):
        aliases = json.load(open(REPO / "aliases" / f"{it['book']}.json"))["principals"]
        text, spans = numbered(it["original"])
        resolved = []
        if spans:
            resolved, _ = llm.ask(messages(text, it["labels"]), check(len(spans), it["labels"]), max_tokens=12000)
        masked = remask(it, aliases, resolved, spans)
        return {**it, "masked": masked, "pronouns_masked": sum(r["ref"] != "other" for r in resolved),
                "pronouns_total": len(spans)}

    with ThreadPoolExecutor(64) as pool:
        out = list(pool.map(one, items))
    with open(DATA / f"items_{args.split}_pron.jsonl", "w") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(out)} items; principal pronouns masked {sum(i['pronouns_masked'] for i in out)} of "
          f"{sum(i['pronouns_total'] for i in out)} pronouns", flush=True)


if __name__ == "__main__":
    sys.exit(main())
