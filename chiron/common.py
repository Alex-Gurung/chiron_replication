"""Shared paths, book loading, snippet splitting, display names and jsonl helpers.

Stdlib only, so it runs under any pod python. Book text comes from the chapter file
rebuilt from ncp_cohorts_v2 (ncp_charmem_v3/chapters.jsonl); principals are the
book-level main_character_labels.
"""
import collections
import html
import json
import os
import re
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "outputs"
CHAPTERS = Path("/home/toolkit/ncp_charmem_v3/chapters.jsonl")
COHORTS = Path("/home/toolkit/ncp_cohorts_v2")
TEST_BOOKS = ("dark", "god", "mercy", "witch")
SNIPPET_WORDS = 300

NAME_SKIP = {"The", "A", "An", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.", "Dr", "Dr.", "Miss", "Sir", "Lady",
             "Lord", "Aunt", "Uncle", "Father", "Mother", "Sister", "Brother"}


def nfc(text):
    return unicodedata.normalize("NFC", text)


def fold(text):
    """NFC + accent-stripped, for name matching (book text is NFD, labels NFC)."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def clean_text(text):
    """Strip the HTML some books carry and unescape entities (same rules as charmem common.clean_text)."""
    t = re.sub(r"<!--.*?-->|<!--<hr/>", "", text, flags=re.S)
    t = re.sub(r"<img[^<>]*>", "", t)
    t = re.sub(r"<hr\s*/?>", "\n", t)
    t = re.sub(r"</?[A-Za-z][^<>\n]{0,200}/?>", "", t)
    return nfc(html.unescape(t))


def load_chapters():
    by_book = collections.defaultdict(list)
    for line in open(CHAPTERS):
        r = json.loads(line)
        by_book[r["book_id"]].append(r)
    for v in by_book.values():
        v.sort(key=lambda r: r["chapter_index"])
    return dict(by_book)


def boundaries():
    """book -> sorted target chapter indices of the v2 cohort (2..N-1)."""
    out = collections.defaultdict(set)
    for split in ("train", "val", "test"):
        for line in open(COHORTS / f"{split}_examples.jsonl"):
            r = json.loads(line)
            out[r["story_id"]].add(r["chapter_index"])
    return {k: sorted(v) for k, v in out.items()}


SENT = re.compile(r"(?<=[.!?…])[\"”’)\]]*\s+(?=[\"“‘(\[]?[A-Z0-9])")


def split_snippets(text, target=SNIPPET_WORDS):
    """Greedy sentence packing into ~target-word snippets; a short tail joins the previous one."""
    sents = [s for s in SENT.split(" ".join(text.split())) if s.strip()]
    chunks, cur, n = [], [], 0
    for s in sents:
        cur.append(s)
        n += len(s.split())
        if n >= target:
            chunks.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        if chunks and n < target // 3:
            chunks[-1] += " " + " ".join(cur)
        else:
            chunks.append(" ".join(cur))
    return chunks


def name_seen(name, text_folded):
    toks = [t.strip(".,'’") for t in re.findall(r"[^\s\"“”]+", fold(name))]
    toks = [t for t in toks if t and t[0].isupper() and t not in NAME_SKIP]
    return all(re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", text_folded) for t in toks) if toks else True


def display_name(label, seen_folded):
    """Label with any variant (slash part or parenthetical) not yet named in the seen text removed."""
    kept = []
    for part in [p.strip() for p in label.split("/") if p.strip()]:
        base = re.sub(r"\s*\([^)]*\)", "", part).strip()
        parens = [p.strip() for p in re.findall(r"\(([^)]*)\)", part)]
        keep_parens = [p for p in parens if name_seen(p, seen_folded)]
        if name_seen(base, seen_folded):
            kept.append(base + "".join(f" ({p})" for p in keep_parens))
        elif keep_parens:
            kept.append(keep_parens[0])
    if kept:
        return " / ".join(kept)
    first = label.split("/")[0]
    return re.sub(r"\s*\([^)]*\)", "", first).strip()


def read_jsonl(path):
    out = []
    if not Path(path).exists():
        return out
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass                                    # torn final line after preemption
    return out


def append_jsonl(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


PRONOUN_M = re.compile(r"\b(he|him|his|himself)\b", re.I)
PRONOUN_F = re.compile(r"\b(she|her|hers|herself)\b", re.I)
GENDER_FIX = {("first_lie", "Mr. Smith"): "M", ("ugly", "Kitty Goldman"): "F"}    # too few pronouns in their sheets


def genders(reps):
    """Book-level gender per principal: over 65% of the pronouns in its v2 and charmem sheets one way.

    (Rolling summaries are not used: in two long books they drift onto another principal.)"""
    counts = collections.defaultdict(lambda: [0, 0])
    for r in reps:
        if r["condition"] in ("v2", "charmem"):
            c = counts[(r["book"], r["label"])]
            c[0] += len(PRONOUN_M.findall(r["text"]))
            c[1] += len(PRONOUN_F.findall(r["text"]))
    g = {k: "M" if m > 0.65 * (m + f) else "F" if f > 0.65 * (m + f) else "?" for k, (m, f) in counts.items()}
    return {**g, **{k: v for k, v in GENDER_FIX.items() if k in g}}


def joint_correct(d):
    """d: target -> (logprobs over id digits, answer id) for one passage and block order. Returns target -> 1/0 under
    the one-to-one name-to-id mapping with the highest summed log-probability (each id is used once)."""
    import itertools
    ts = list(d)
    best = max(itertools.permutations(range(len(ts))), key=lambda p: sum(d[t][0].get(str(i), -1e9) for t, i in zip(ts, p)))
    return {t: int(i == d[t][1]) for t, i in zip(ts, best)}


NARRATOR = OUT / "narrator"          # gen_narrator.py: who says "I", per (book, chapter)


def load_narrators():
    """(book, chapter_index) -> the first-person narrator's name, or None. Chapters whose narrator the model could not
    name inherit the book's narrator when one name covers at least 80% of the named first-person chapters."""
    got, first = {}, {}
    for f in NARRATOR.glob("*.jsonl"):
        for r in read_jsonl(f):
            got[(r["book"], r["chapter_index"])] = r["narrator"] or None
            first[(r["book"], r["chapter_index"])] = r["first_person"]
    per_book = collections.defaultdict(collections.Counter)
    for (b, c), n in got.items():
        if n:
            per_book[b][n] += 1
    for (b, c), n in list(got.items()):
        if first[(b, c)] and not n and per_book[b]:
            name, k = per_book[b].most_common(1)[0]
            if k >= 0.8 * sum(per_book[b].values()):
                got[(b, c)] = name
    return got


def chapter_context(ch, narrators):
    """What every gpt-oss generator reading chapter ch is told besides its text: the chapter heading, and who "I" is."""
    parts = []
    head = " ".join((ch.get("chapter_header") or "").split())
    if head:
        parts.append(f"Chapter heading: {head}")
    n = narrators.get((ch["book_id"], ch["chapter_index"]))
    if n:
        parts.append(f'This chapter is narrated in the first person by {n}: "I" and "me" refer to {n}.')
    return "\n".join(parts)
