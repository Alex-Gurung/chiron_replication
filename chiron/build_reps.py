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
  *_h            summary_h, chapnotes_h, chiron_h(_r2000): the same, generated with chapter headings and narrator hints
  chapnotes_h_long  gpt-oss chapter notes asked for thorough answers (gen_chapnotes.py --headings --long)
  legacy_gptoss  gpt-oss with the Llama notes' own extraction prompt (gen_legacy_gptoss.py), no splitting or filtering
  legacy_gptoss_nofill  the same without sentences saying the chapter does not mention the character (gclean)
  legacy_match   Llama notes without filler, each chapter's answer cut to gpt-oss's answer length for that chapter and question
 sheet_<variant> gpt-oss character sheets compressed from chapter notes (gen_sheet.py, outputs/sheets/<variant>)
  gender         "Gender: female." / "Gender: male." only (common.genders, from the v2 and charmem sheets)
The sheet_* conditions go to data/reps_sheets_<split>.jsonl instead (--sheets-only rebuilds just that file).
Run with the repo venv (needs scikit-learn): .venv/bin/python chiron/build_reps.py --split test [--sheets-only]
"""
import argparse
import collections
import glob
import json
import os
import pickle
import re
from pathlib import Path

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
    from sklearn.feature_extraction.text import TfidfVectorizer       # imported here so gen_sheet can use this module
    from sklearn.metrics.pairwise import cosine_similarity
    if len(statements) < 2:
        return statements
    sims = cosine_similarity(TfidfVectorizer().fit_transform(statements))
    keep = []
    for i in range(len(statements)):
        if all(sims[i, j] < threshold for j in keep):
            keep.append(i)
    return [statements[i] for i in keep]


def chiron_sheets(keys, root="chiron"):
    """keys: set of (book, boundary, label). Statements come from snippets of chapters < boundary, in book order.
    root "chiron_h": the run that saw each snippet's chapter heading and narrator."""
    by = collections.defaultdict(list)
    for f in glob.glob(str(OUT / root / "shards" / "*.jsonl")):
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


GNEG = re.compile(r"\b(?:no|not|never|none|nothing|neither|nor)\b|n't\b", re.I)
GMETA = re.compile(r"\b(?:text|snippet|excerpt|section|passage|story|mention\w*|describ\w*|information|indication|quot\w*|details?|"
                   r"depict\w*|reveal\w*|provid\w*|specif\w*|stated|appears?|shown|given|attributed|listed|spoken|assigned|explained|"
                   r"refer\w*|record\w*|clues?|attached|indicated)\b", re.I)
GABOUT = re.compile(r"^(?:(?:the|this) (?:text|snippet|excerpt|section|story section|passage)\b|therefore\b|no\b|none\b|nothing\b|"
                    r"there (?:is|are) no\b|we (?:have|do) not\b|we have no\b)", re.I)


def gclean(text):
    """gpt-oss notes without sentences about what the chapter does not say ("The snippet never mentions X.")."""
    return " ".join(x for x in SENT.split(" ".join(text.split())) if not (GNEG.search(x) and GMETA.search(x)) and not GABOUT.search(x))


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


SHEET_ABLATE = {"refer": "how others refer", "voice": "voice", "story": "story so far", "rel": "relationships"}


def add_sheets(add, keys):
    for d in sorted((OUT / "sheets").glob("*")):                # new gpt-oss sheets, one condition per variant
        sh = {(r["book"], r["boundary"], r["label"]): re.sub(r"^(#+) ", r"#\1 ", r["text"], flags=re.M)   # nest under the name
              .replace("\u2011", "-").replace("\u202f", " ") for f in d.glob("*.jsonl") for r in read_jsonl(f)}
        for k in keys:
            add(f"sheet_{d.name}", *k, sh.get(k))
        if d.name in ("bible_cn", "bible_leg"):                 # one section dropped at a time
            for k in keys:
                if k not in sh:
                    continue
                secs = re.split(r"(?m)^(?=#+ )", sh[k])
                for short, head in SHEET_ABLATE.items():
                    kept = [x for x in secs if not re.match(r"#+\s*" + head, x.strip(), re.I)]
                    add(f"sheet_{d.name}_no{short}", *k, "".join(kept))


NAMESEQ = re.compile(r"(?<=[a-z,;:’'\"] )((?:(?:Dr|Mr|Mrs|Ms|St)\. )?[A-Z][\w’'-]+(?: (?:of |de |van |von |the )?[A-Z][\w’'-]+)*)")


def name_index(notes, keys, n=40):
    """(book, boundary, label) -> the non-principal names in that character's notes, ranked by count x the share of
    their mentions that fall in this character's notes rather than the other principals' (most distinctive first)."""
    out = {}
    counts = {}
    for k, chs in notes.items():
        c = collections.Counter()
        for _, t in chs:
            c.update(re.sub(r"[’']s$", "", x) for x in NAMESEQ.findall(t))
        counts[k] = c
    for (b, bd, l) in keys:
        if (b, bd, l) not in counts:
            continue
        al = json.load(open(REPO / "aliases" / f"{b}.json"))["principals"]
        principal = {w for v in al.values() for a in v for w in a.split()}
        others = [counts.get((b, bd, o), collections.Counter()) for o in al if o != l]
        own = counts[(b, bd, l)]
        score = {e: m * m / (m + sum(o[e] for o in others) / max(1, len(others))) for e, m in own.items()
                 if m >= 2 and not set(e.replace(".", "").split()) & principal and len(e) > 2}
        out[(b, bd, l)] = [e for e, _ in sorted(score.items(), key=lambda x: -x[1])[:n]]
    return out


def add_index(add, keys):
    """Name indexes from the gpt-oss notes (legacy_gptoss), alone and appended to v2 / charmem / the new sheets."""
    from gen_sheet import load_notes
    idx = name_index(load_notes("legacy_gptoss", keys), keys)
    have = {}
    for k, names in idx.items():
        if names:
            add("index_lgp", *k, "Names in this character's story so far, most distinctive first: " + ", ".join(names) + ".")
    return idx


def save(recs, split, keys):
    tmp = DATA / f"reps_{split}.jsonl.tmp"                    # atomic swap: running jobs never read a half-written file
    with open(tmp, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, DATA / f"reps_{split}.jsonl")
    cov = collections.Counter(r["condition"] for r in recs)
    words = collections.defaultdict(list)
    for r in recs:
        words[r["condition"]].append(r["words"])
    for c in sorted(cov):
        w = sorted(words[c])
        print(f"{c:22s} {cov[c]:5d}/{len(keys)} keys  median words {w[len(w) // 2]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--sheets-only", action="store_true", help="only refresh the sheet_* conditions in the existing file")
    args = ap.parse_args()
    items = [it for f in sorted(DATA.glob(f"items_{args.split}*.jsonl")) for it in read_jsonl(f)]   # every item set of the split
    keys = {(it["book"], it["chapter_index"], l) for it in items for l in it["labels"]}
    books = {k[0] for k in keys}
    recs = []

    def add(cond, book, b, label, text):
        if text is not None:
            recs.append({"condition": cond, "book": book, "boundary": b, "label": label, "text": text,
                         "words": len(text.split())})
    add_sheets(add, keys)                                         # sheets live in their own small file (rebuilt often)
    idx = add_index(add, keys)
    base = {(r["condition"], r["book"], r["boundary"], r["label"]): r["text"] for r in read_jsonl(DATA / f"reps_{args.split}.jsonl")
            if r["condition"] in ("v2", "charmem")} if args.sheets_only else {}
    base.update({(r["condition"], r["book"], r["boundary"], r["label"]): r["text"] for r in recs if r["condition"].startswith("sheet_")})
    for (c, *k), t in base.items():
        if idx.get(tuple(k)) and (c in ("v2", "charmem") or c in ("sheet_chiron_lgp", "sheet_chiron_cnl")):
            add(f"{c}+index", *k, t.rstrip() + "\n\n### Names in the story\n" + ", ".join(idx[tuple(k)]) + ".")
    save(recs, f"sheets_{args.split}", keys)
    if args.sheets_only:
        return
    recs.clear()

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
            for n in (1000, 2000, 3000, 6000):                    # the most recent n words
                add(f"legacy_r{n}", book, b, l, legacy_recent(full[k], n))
            add("legacy_nofill", book, b, l, legacy_render(full[k], lambda c: True, clean=True))
            for name, want in (("legacy_inter", True), ("legacy_nointer", False)):
                add(name, book, b, l, legacy_render(full[k], lambda c: True, clean=True,
                                                    keep_sentence=lambda x, want=want: names_other(book, l, x) == want))
    for cond, root in (("summary", "summary_v2"), ("summary_h", "summary_h")):   # v1 (outputs/summary) grew past the cap
        summ = {}
        for f in glob.glob(str(OUT / root / "*.jsonl")):
            if f.endswith(".errors.jsonl"):
                continue
            for r in read_jsonl(f):
                summ[(r["book"], r["boundary"], r["label"])] = r["summary"]
        for k in keys:
            add(cond, *k, summ.get(k))
    if glob.glob(str(OUT / "chiron_h" / "shards" / "*.jsonl")):
        for k, rows in chiron_sheets(keys, "chiron_h").items():
            add("chiron_h", *k, render(rows))
            add("chiron_h_r2000", *k, render(budget(rows, 2000)))
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
    lg = collections.defaultdict(lambda: collections.defaultdict(dict))   # gpt-oss with the Llama notes' extraction prompt
    for f in glob.glob(str(OUT / "legacy_gptoss" / "*.jsonl")):
        for r in read_jsonl(f):
            lg[(r["book"], r["label"])][r["chapter_index"]][r["q"]] = r["answer"]
    for book, b, l in keys:
        chs = [c for c in sorted(lg.get((book, l), {})) if c < b]
        if chs and len(chs) == b and all(len(lg[(book, l)][c]) == len(QUESTIONS) for c in chs):
            parts = []
            for head, qs in CHAPNOTE_LAYOUT:
                parts.append(f"## {head}")
                for q in qs:
                    parts.append(f"Question: {QUESTIONS[q][1]}\n\n" + "\n".join(f"<snippet {c}>\n{lg[(book, l)][c][q].strip()}" for c in chs))
            add("legacy_gptoss", book, b, l, "\n\n".join(parts))
            parts = []                                                # the same without "the section never mentions X" filler
            for head, qs in CHAPNOTE_LAYOUT:
                parts.append(f"## {head}")
                for q in qs:
                    body = [(c, gclean(lg[(book, l)][c][q])) for c in chs]
                    body = "\n".join(f"<snippet {c}>\n{t}" for c, t in body if t)
                    if body:
                        parts.append(f"Question: {QUESTIONS[q][1]}\n\n{body}")
            add("legacy_gptoss_nofill", book, b, l, "\n\n".join(parts))
    for cond, root in (("chapnotes", "chapnotes"), ("chapnotes_h", "chapnotes_h"), ("chapnotes_h_long", "chapnotes_h_long")):
        notes = collections.defaultdict(dict)                     # gpt-oss chapter notes, Llama layout
        for f in glob.glob(str(OUT / root / "*.jsonl")):
            for r in read_jsonl(f):
                notes[(r["book"], r["label"])][r["chapter_index"]] = r["answers"]
        for book, b, l in keys:
            chs = sorted(c for c in notes.get((book, l), {}) if c < b)
            if chs and len(chs) == b:                             # every earlier chapter present
                parts = []
                for head, qs in CHAPNOTE_LAYOUT:
                    parts.append(f"## {head}")
                    for q in qs:
                        body = "\n".join(f"<snippet {c}>\n{notes[(book, l)][c][q]}" for c in chs if notes[(book, l)][c][q])
                        if body:
                            parts.append(f"Question: {QUESTIONS[q][1]}\n\n{body}")
                add(cond, book, b, l, "\n\n".join(parts))
                if cond == "chapnotes_h_long":
                    for n in (1000, 2000):
                        add(f"{cond}_r{n}", book, b, l, legacy_recent("\n\n".join(parts), n))
        if cond == "chapnotes":                                   # Llama notes cut to gpt-oss's length, answer by answer
            qkey = {q: k for k, (_, q) in QUESTIONS.items()}
            for book, b, l in keys:
                k = f"{book}_{LEGACY_LABEL.get(l, l)}_{b}"
                if not full.get(k) or len(notes.get((book, l), {})) < b:
                    continue
                parts = []
                for head, blocks in legacy_blocks(full[k]):
                    q = qkey.get(head[len("Question: "):].strip()) if head.startswith("Question:") else None
                    if not q:
                        parts.append(head)
                        continue
                    body = []
                    for c, t in blocks:
                        n = len((notes[(book, l)].get(c) or {}).get(q, "").split())
                        w = FILLER.sub("", t).split()[:n]
                        if w:
                            body.append(f"<snippet {c}>\n{' '.join(w)}")
                    if body:
                        parts.append(head + "\n\n" + "\n".join(body))
                add("legacy_match", book, b, l, "\n\n".join(parts))
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
    save(recs, args.split, keys)


if __name__ == "__main__":
    main()
