"""Assemble every character representation for the items of a split -> data/reps_<split>.jsonl.

One record per (condition, book, boundary, label); every source is built from chapters < boundary:
  v2             character_sheets in ncp_cohorts_v2 (principal sheet only)
  legacy         Llama-3.3-70B CHIRON-style sheet, compressed (~500 words), HF agurung/new_ncp_data_creation
  legacy_full    the same sheet before compression
  summary        gpt-oss rolling character summary S_b (outputs/summary)
  chiron         gpt-oss CHIRON-style statements rated 5, grouped by category, TF-IDF dedup at 0.9
  chiron_<cat>   one category only (for the per-category "Agreed" setting)
  charmem        finished Sep 8 gpt-oss rebuild sheet (ncp_charmem_gptoss120b_reviewed_20260908/sheets)
  chapnotes      gpt-oss chapter notes (gen_chapnotes.py): the Llama notes' layout and questions, redone with gpt-oss
  gender         "Gender: female." / "Gender: male." only (common.genders, from the v2 and charmem sheets)
Run with the repo venv (needs scikit-learn): .venv/bin/python chiron/build_reps.py --split test
"""
import argparse
import collections
import glob
import json
import os
import pickle
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from build_items import alias_regex
from common import COHORTS, DATA, OUT, REPO, SENT, genders, read_jsonl
from gen_chiron import QUESTIONS

CATEGORIES = ["Physical/Personality", "Dialogue", "Knowledge", "Goals"]
CHARMEM = Path("/home/toolkit/ncp_charmem_gptoss120b_reviewed_20260908/sheets")
LEGACY = DATA / "legacy"
CHAPNOTE_LAYOUT = [("Personality/Physical Attributes", ["physical", "personality"]), ("Knowledge", ["facts", "learned"]),
                   ("Plot and Motivation", ["goals_gained", "goals_completed", "motivation_change"]), ("Character Dialogue", ["dialogue"])]
LEGACY_LABEL = {"Rohan": "Rohanc", "Michael Bradshaw": "Michael BradshaDavinaw"}   # the archive keeps the pre-repair names


def dedup(statements, threshold=0.9):
    if len(statements) < 2:
        return statements
    sims = cosine_similarity(TfidfVectorizer().fit_transform(statements))
    keep = []
    for i in range(len(statements)):
        if all(sims[i, j] < threshold for j in keep):
            keep.append(i)
    return [statements[i] for i in keep]


def chiron_sheets(keys):
    """keys: set of (book, boundary, label). Statements come from snippets of chapters < boundary, in book order."""
    by = collections.defaultdict(list)
    for f in glob.glob(str(OUT / "chiron" / "shards" / "*.jsonl")):
        if f.endswith(".errors.jsonl"):
            continue
        for r in read_jsonl(f):
            for c in r["claims"]:
                if c["rating"] == 5:
                    by[(r["book"], r["label"])].append((r["chapter_index"], r["idx"], c["category"], c["text"]))
    out = {}
    for book, b, label in keys:
        rows = sorted(x for x in by[(book, label)] if x[0] < b)
        kept = {cat: set(dedup([t for _, _, c, t in rows if c == cat])) for cat in CATEGORIES}
        out[(book, b, label)] = [(c, t, ch) for ch, _, c, t in rows if t in kept[c] and not kept[c].discard(t)]
    return out


def budget(rows, k):
    """Most recent statements first until k words, then back in book order."""
    picked, n = [], 0
    for c, t, ch in reversed(rows):
        if n + len(t.split()) > k:
            break
        picked.append((c, t, ch))
        n += len(t.split())
    return picked[::-1]


FILLER = re.compile(r"[^.!?]*(?:not mentioned|no (?:physical )?description|not (?:explicitly )?described|no information)[^.!?]*[.!?]?", re.I)


def legacy_blocks(text):
    """Split a full legacy sheet into (header lines, [(chapter, block text)]) per question, keeping order."""
    out, cur = [], None
    for line in text.split("\n"):
        m = re.match(r"<snippet (\d+)>", line)
        if line.startswith(("## ", "Question:")):
            cur = [line, []]
            out.append(cur)
        elif m and cur is not None:
            cur[1].append([int(m.group(1)), ""])
        elif cur is not None and cur[1]:
            cur[1][-1][1] += line + "\n"
    return out


def legacy_render(text, keep, clean=False, keep_sentence=None):
    parts = []
    for head, blocks in legacy_blocks(text):
        if keep_sentence:
            blocks = [(c, " ".join(x for x in SENT.split(" ".join(t.split())) if keep_sentence(x))) for c, t in blocks]
        body = [f"<snippet {c}>\n{FILLER.sub('', t).strip() if clean else t.strip()}" for c, t in blocks if keep(c)]
        body = [x for x in body if x.split("\n", 1)[-1].strip()]
        if body or head.startswith("## "):
            parts.append(head + ("\n\n" + "\n".join(body) if body else ""))
    return "\n\n".join(parts)


def legacy_recent(text, budget):
    chapters = sorted({c for _, blocks in legacy_blocks(text) for c, _ in blocks}, reverse=True)
    words, keep = {c: 0 for c in chapters}, set()
    for _, blocks in legacy_blocks(text):
        for c, t in blocks:
            words[c] += len(t.split())
    total = 0
    for c in chapters:
        if total + words[c] > budget and keep:
            break
        keep.add(c)
        total += words[c]
    return legacy_render(text, lambda c: c in keep)


def render(rows, only=None):
    parts = [f"### {cat}\n" + ("\n".join(f"- {t}" for c, t, _ in rows if c == cat) or "- No information.")
             for cat in CATEGORIES if only in (None, cat)]
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    items = [it for f in sorted(DATA.glob(f"items_{args.split}*.jsonl")) for it in read_jsonl(f)]   # every item set of the split
    keys = {(it["book"], it["chapter_index"], l) for it in items for l in it["labels"]}
    books = {k[0] for k in keys}
    recs = []

    def add(cond, book, b, label, text):
        if text is not None:
            recs.append({"condition": cond, "book": book, "boundary": b, "label": label, "text": text,
                         "words": len(text.split())})

    aliases = {}

    def names_other(book, label, text):
        if book not in aliases:
            aliases[book] = {l: alias_regex(a) for l, a in json.load(open(REPO / "aliases" / f"{book}.json"))["principals"].items()}
        return any(rx.search(text) for l, rx in aliases[book].items() if l != label)

    seen = set()
    for line in open(COHORTS / f"{args.split}_examples.jsonl"):
        r = json.loads(line)
        for l, t in r["character_sheets"].items():
            k = (r["story_id"], r["chapter_index"], l)
            if k in keys and k not in seen:
                seen.add(k)
                add("v2", *k, t)
    full = pickle.load(open(LEGACY / f"{args.split}_long_story_storycharchap_to_csheet.pkl", "rb"))
    comp = pickle.load(open(LEGACY / f"{args.split}_long_story_character_sheet_summaryllama70B_0.5max.pkl", "rb"))
    for book, b, l in keys:
        k = f"{book}_{LEGACY_LABEL.get(l, l)}_{b}"
        add("legacy", book, b, l, comp.get(k))
        add("legacy_full", book, b, l, full.get(k))
        if full.get(k):                                # ablations: previous chapter removed / alone, recency budget, no filler
            add("legacy_noprev", book, b, l, legacy_render(full[k], lambda c: c != b - 1))
            add("legacy_onlyprev", book, b, l, legacy_render(full[k], lambda c: c == b - 1))
            add("legacy_r6000", book, b, l, legacy_recent(full[k], 6000))
            add("legacy_nofill", book, b, l, legacy_render(full[k], lambda c: True, clean=True))
            for name, want in (("legacy_inter", True), ("legacy_nointer", False)):
                add(name, book, b, l, legacy_render(full[k], lambda c: True, clean=True,
                                                    keep_sentence=lambda x, want=want: names_other(book, l, x) == want))
    summ = {}
    for f in glob.glob(str(OUT / "summary_v2" / "*.jsonl")):       # v1 (outputs/summary) grew past the cap; superseded
        if f.endswith(".errors.jsonl"):
            continue
        for r in read_jsonl(f):
            summ[(r["book"], r["boundary"], r["label"])] = r["summary"]
    for k in keys:
        add("summary", *k, summ.get(k))
    for k, rows in chiron_sheets(keys).items():
        add("chiron", *k, render(rows))
        add("chiron_inter", *k, render([r for r in rows if names_other(k[0], k[2], r[1])]))       # statements naming another principal
        add("chiron_nointer", *k, render([r for r in rows if not names_other(k[0], k[2], r[1])]))
        prev = k[1] - 1
        add("chiron_noprev", *k, render([r for r in rows if r[2] != prev]))
        add("chiron_onlyprev", *k, render([r for r in rows if r[2] == prev]))
        for cat in CATEGORIES:
            add("chiron_" + cat.split("/")[0].lower(), *k, render(rows, cat))
        for words in (250, 500, 1000, 2000, 4000):
            add(f"chiron_r{words}", *k, render(budget(rows, words)))
    notes = collections.defaultdict(dict)                     # gpt-oss chapter notes, laid out like the Llama notes
    for f in glob.glob(str(OUT / "chapnotes" / "*.jsonl")):
        for r in read_jsonl(f):
            notes[(r["book"], r["label"])][r["chapter_index"]] = r["answers"]
    for book, b, l in keys:
        chs = sorted(c for c in notes.get((book, l), {}) if c < b)
        if chs and len(chs) == b:                                 # every earlier chapter present
            parts = []
            for head, qs in CHAPNOTE_LAYOUT:
                parts.append(f"## {head}")
                for q in qs:
                    body = "\n".join(f"<snippet {c}>\n{notes[(book, l)][c][q]}" for c in chs if notes[(book, l)][c][q])
                    if body:
                        parts.append(f"Question: {QUESTIONS[q][1]}\n\n{body}")
            add("chapnotes", book, b, l, "\n\n".join(parts))
    for book, b, l in keys:
        p = CHARMEM / f"{book}__{b:04d}.json"
        if p.exists():
            add("charmem", book, b, l, json.load(open(p))["character_sheets"].get(l))
    for (book, label), x in genders(recs).items():               # gender alone: what every sheet gives away
        for k in keys:
            if k[0] == book and k[2] == label and x != "?":
                add("gender", *k, f"Gender: {'female' if x == 'F' else 'male'}.")
    sections = {"physical": "Physicality And Personality", "dialogue": "Dialogue And Voice", "history": "History And Circumstances",
                "knowledge": "Knowledge And Beliefs", "goals": "Goals And Motivations", "relationships": "Relationships"}
    # one sheet section at a time, and everything but relationships
    sheet = {(r["condition"], r["book"], r["boundary"], r["label"]): r["text"] for r in recs if r["condition"] in ("v2", "charmem")}
    for (src, *k), t in sheet.items():
        parts = {h.strip(): body for h, body in re.findall(r"^## (.+?)\n(.*?)(?=^## |\Z)", t, re.M | re.S)}
        for short, head in sections.items():
            if head in parts:
                add(f"{src}_sec_{short}", *k, f"## {head}\n{parts[head].strip()}")
        add(f"{src}_norel", *k, "\n\n".join(f"## {h}\n{b.strip()}" for h, b in parts.items() if h != "Relationships"))
    combos = {"combo_short": ["v2", "charmem", "summary"], "combo_legacy_v2": ["legacy_nofill", "v2"],
              "combo_all": ["v2", "charmem", "summary", "legacy_nofill"]}       # complementary sources, concatenated
    have = {(r["condition"], r["book"], r["boundary"], r["label"]): r["text"] for r in recs}
    titles = {"v2": "Sheet", "charmem": "Second sheet", "summary": "Summary", "legacy_nofill": "Chapter-by-chapter notes"}
    for name, parts in combos.items():
        for k in keys:
            texts = [have.get((c, *k)) for c in parts]
            if all(texts):
                add(name, *k, "\n\n".join(f"#### {titles[c]}\n{t}" for c, t in zip(parts, texts)))
    tmp = DATA / f"reps_{args.split}.jsonl.tmp"               # atomic swap: running jobs never read a half-written file
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, DATA / f"reps_{args.split}.jsonl")
    cov = collections.Counter(r["condition"] for r in recs)
    words = collections.defaultdict(list)
    for r in recs:
        words[r["condition"]].append(r["words"])
    for c in sorted(cov):
        w = sorted(words[c])
        print(f"{c:22s} {cov[c]:5d}/{len(keys)} keys  median words {w[len(w) // 2]}")


if __name__ == "__main__":
    main()
