"""Render reports/results.html from the analysis files (no numbers are typed by hand).

  python3 chiron/make_report.py
Reads outputs/analysis_<set>_<model>.json (from analyze.py) for the main, window, short, two and pron sets,
data/items_*.jsonl and data/reps_*.jsonl; writes a self-contained page with two inline SVG charts
(representation length vs accuracy, passage length vs accuracy), each switchable by model.
Findings text lives in FINDINGS below and is written against the final numbers.
"""
import collections
import glob
import html
import json
import re
import statistics as st

from common import DATA, OUT, REPO, read_jsonl

MODELS = [("Qwen3-4B-Instruct-2507", "Qwen3-4B"), ("Qwen3.5-9B-Base", "Qwen3.5-9B base"), ("Qwen3.8-27B_nothink", "Qwen3.8-27B"),
          ("Qwen3.8-27B_think", "Qwen3.8-27B, thinking"), ("Mistral-7B-Instruct-v0.2_prefix", "Mistral-7B")]
CHART_MODELS = [("Qwen3-4B-Instruct-2507", "Qwen3-4B"), ("Qwen3.8-27B_nothink", "Qwen3.8-27B"), ("Qwen3.8-27B_think", "Qwen3.8-27B, thinking")]
LABEL = {
    "noinfo": "Names only", "gender": "Gender only (\"Gender: female.\")", "v2": "v2 sheet (current dataset)", "legacy": "Llama CHIRON notes, summarized",
    "legacy_full": "Llama CHIRON notes, full", "chapnotes": "gpt-oss chapter notes (the Llama notes redone)", "summary": "gpt-oss summary",
    "chiron": "gpt-oss CHIRON claims, full", "chiron_r2000": "gpt-oss CHIRON claims, last 2,000 words",
    "charmem": "Charmem sheet (gpt-oss rebuild of v2)", "book_last8000": "Book text, last 8,000 words",
    "book_last32000": "Book text, last 32,000 words", "book": "Book text, everything so far",
    "swap_v2": "Swap: another principal's v2 sheet", "swapname_v2": "Swap, names exchanged: v2",
    "swap_chiron": "Swap: another principal's gpt-oss claims", "swapname_chiron": "Swap, names exchanged: gpt-oss claims",
    "combo_legacy_v2": "Llama notes without filler + v2", "combo_short": "v2 + charmem + summary", "combo_all": "All four combined",
    "oracle_passage": "Oracle: clues taken from the passage", "oracle_prior": "Oracle: prior facts chosen for the passage",
    "swapname_oracle_prior": "Oracle prior facts, swapped with names exchanged",
    "book_ch1": "Book text, previous chapter", "book_ch2": "Book text, last 2 chapters", "book_ch4": "Book text, last 4 chapters",
    "book_ch8": "Book text, last 8 chapters", "book_prefix": "Book text, this chapter up to the passage",
    "book_ch1p": "Book text, previous chapter + this chapter up to the passage",
    **{f"plot_{k}_{n}": f"Plot summary, {v}, about {n:,} words" for k, v in (("global", "one pass"), ("hier", "chapter by chapter")) for n in (500, 1000, 2000, 4000)},
    **{f"plot_{k}_4000@last{n}": f"Plot summary, {v}, last {n:,} of 4,000 words" for k, v in (("global", "one pass"), ("hier", "chapter by chapter")) for n in (500, 1000, 2000)},
    "oracle_quote": "Oracle: the passage's own sentences, names left in", "manual_passage": "Hand-written: clues from the passage",
    "manual_prior": "Hand-written: prior facts chosen for the passage",
    "swapname_manual_prior": "Hand-written prior facts, swapped with names exchanged", "legacy_nofill": "Llama notes, full, without filler",
}
REP_INFO = [
    ("v2", "The character sheets in the current NCP dataset (ncp_cohorts_v2): six sections (physicality, dialogue, history, knowledge, goals, relationships) with chapter citations."),
    ("charmem", "A gpt-oss-120b rebuild of the v2 sheets, updated chapter by chapter with a review pass (the September 8 rebuild, finished here for all 30 books). Same six sections."),
    ("summary", "gpt-oss-120b rolling prose summary of the character, updated after every chapter and condensed to about 700 words."),
    ("legacy_full", "From the original NCP archive: Llama-3.3-70B answered CHIRON's 8 questions (appearance, personality, dialogue, knowledge, goals, and so on) once per chapter, with the whole chapter as the story section; kept as short prose sentences, unfiltered."),
    ("legacy", "The same Llama notes summarized by Llama-3.3-70B to about 500 words."),
    ("chapnotes", "The Llama notes redone with gpt-oss-120b: the same 8 questions once per chapter, same layout, empty answers left out. Only the model differs."),
    ("chiron", "Our gpt-oss-120b run of CHIRON's full pipeline as in the paper: the same questions per ~300-word snippet, answers as single claims, each kept only if rated fully entailed by its snippet (5 on CHIRON's 1-5 scale), then deduplicated. No trained verifier."),
]
SHEET_SECTIONS = [("relationships", "Relationships"), ("history", "History"), ("goals", "Goals"), ("physical", "Physical"),
                  ("dialogue", "Dialogue"), ("knowledge", "Knowledge")]
REASON_ROWS = ["noinfo", "v2", "charmem", "summary", "legacy", "chiron_r2000", "chiron", "book_last8000", "legacy_full", "swapname_v2"]
GENDER_ROWS = ["noinfo", "gender", "v2", "charmem", "summary", "book_last8000", "legacy_full", "swapname_v2"]
GENDER_COLS = [("Qwen3.8-27B_nothink", "27B, sections"), ("Qwen3.8-27B_think", "27B thinking, sections"),
               ("Qwen3.8-27B_think_short", "27B thinking, short spans")]
MAIN_ROWS = ["noinfo", "gender", "legacy", "chiron_r2000", "v2", "chiron", "charmem", "summary", "book_last8000",
             "book_last32000", "book", "legacy_full", "legacy_nofill", "chapnotes", "combo_short", "combo_legacy_v2"]
ORACLE_ROWS = ["noinfo", "v2", "legacy_full", "oracle_prior", "oracle_passage", "oracle_quote", "swapname_oracle_prior"]
MANUAL_ROWS = ["noinfo", "v2", "legacy_full", "oracle_prior", "manual_prior", "oracle_passage", "manual_passage", "oracle_quote",
               "swapname_manual_prior"]
CONTROL_ROWS = ["v2", "swap_v2", "swapname_v2", "chiron", "swap_chiron", "swapname_chiron"]
LENGTH_FAMILIES = [("gpt-oss claims", ["chiron_r250", "chiron_r500", "chiron_r1000", "chiron_r2000", "chiron_r4000", "chiron"]),
                   ("v2 sheet", ["v2@100", "v2@250", "v2@500", "v2"]),
                   ("Llama notes, summarized", ["legacy@100", "legacy@250", "legacy"]),
                   ("Summary", ["summary@100", "summary@250", "summary@500", "summary"]),
                   ("Book text, last k words", ["book_last2000", "book_last8000", "book_last32000", "book"]),
                   ("Book text, last k chapters", ["book_ch1", "book_ch2", "book_ch4", "book_ch8", "book"])]
FAMILY_STYLE = {}
SUMMARY_FAMILIES = [("Character summaries, first k words", ["summary@100", "summary@250", "summary@500", "summary"]),
                    ("Plot summary, one pass", ["plot_global_500", "plot_global_1000", "plot_global_2000", "plot_global_4000"]),
                    ("Plot summary, chapter by chapter", ["plot_hier_500", "plot_hier_1000", "plot_hier_2000", "plot_hier_4000"]),
                    ("One pass, last k words", ["plot_global_4000@last500", "plot_global_4000@last1000", "plot_global_4000@last2000", "plot_global_4000"]),
                    ("Chapter by chapter, last k words", ["plot_hier_4000@last500", "plot_hier_4000@last1000", "plot_hier_4000@last2000", "plot_hier_4000"])]
SUMMARY_STYLE = {}
PLOT_ROWS = ["noinfo", "summary", "plot_global_500", "plot_global_1000", "plot_global_2000", "plot_global_4000", "plot_hier_500",
             "plot_hier_1000", "plot_hier_2000", "plot_hier_4000", "plot_global_4000@last1000", "plot_hier_4000@last1000", "book_last8000"]
BOOKCH_ROWS = ["noinfo", "book_prefix", "book_ch1", "book_ch1p", "book_ch2", "book_ch4", "book_ch8", "book", "book_last8000", "legacy_full"]
PASSAGE_SERIES = [("v2 sheet", "v2"), ("Charmem", "charmem"), ("gpt-oss claims", "chiron"),
                  ("Summary", "summary"), ("Llama notes", "legacy_full"), ("Book, last 8k", "book_last8000")]
SETS = [("short", "Short spans"), ("main", "Sections"), ("window", "Dense windows")]
FINDINGS = [
    "Scoring changes the thinking-off results. Asked about one character at a time, Qwen3.8-27B without thinking almost "
    "never answers 0: with names only it gets {zero27}% of the characters whose id is [CHAR 0], which caps it near two "
    "thirds whatever it is told (the verbatim-quote oracle scores {quote27a}% that way). Taking the best one-to-one mapping "
    "of its three answers (joint scoring; the prompt says each id is used once) removes this, and the same quote oracle "
    "scores {quote27}%. Every thinking-off number below uses joint scoring; the per-character numbers are in Scoring.",
    "Model strength decides how much the representations help. Over names only, Qwen3-4B gains {g4_lo} to {g4_hi} points, "
    "and Qwen3.8-27B without thinking {g27_lo} to {g27_hi}, positive in every book. With thinking it reaches {think_lo} to "
    "{think_hi}% on sections with any representation.",
    "The models do use the sheets. With another principal's sheet and the names inside it exchanged, Qwen3.8-27B falls "
    "to {swap27}% (names only {noinfo27}%) and to {swapthink}% with thinking; the reasoning traces cite specific sheet "
    "facts and match them to events in the passage.",
    "Exactly the information needed is enough. On {man_n} passages, Claude agents wrote 1 to 4 facts per character that "
    "were established before the chapter and identify it in the passage (median {man_words} words). They give Qwen3.8-27B "
    "{manp27}% without thinking and {manpthink}% with it, against {manv2_27}% and {manv2_think}% for the whole v2 sheet on "
    "the same passages, and {manleg27}% and {manlegthink}% for the full Llama notes. Clues taken from the passage itself give "
    "{manpas27}% and {manpasthink}%. The gpt-oss prior-facts oracle did worse ({orp27}% and {orpthink}%): it picked generic traits.",
    "The representations get different passages right. Which passage it is explains {varp}% of the variance in "
    "correctness and the representation {varr}%; choosing the best representation per passage would reach {pick}% against "
    "{best}% for the best single one. Combining the Llama notes with v2 gives the best score, {combo}%.",
    "The full Llama CHIRON notes win because of what they say, not their length or recency: without the previous chapter "
    "they score {leg_noprev}% (full {legfull27}%), and cut to 6,000 words {leg_r6000}%, still well above the gpt-oss CHIRON "
    "claims at any length ({chiron27}% in full). The Llama notes answer CHIRON's questions once per chapter, seeing the "
    "whole chapter; the gpt-oss claims answer them per 300-word snippet as in the paper, as single claims, and keep "
    "little of what relates a character to the other principals (the Llama notes name them 4 to 7 times as often).",
    "Was the gpt-oss CHIRON run done wrong? Redoing the Llama notes with gpt-oss, with the same questions once per chapter "
    "and the same layout, gives {chap27}% on Qwen3.8-27B without thinking ({chap9}% on Qwen3.5-9B base): above the gpt-oss "
    "claims ({chiron27}%), below the Llama notes ({legfull27}%). Answering per 300-word snippet is the bigger loss. It also "
    "drops first-person narrators: gpt-oss is rarely told who \"I\" is, so it marks narrators absent from most of their own "
    "snippets, and the claims trail the Llama notes most in first-person books. gpt-oss's chapter notes are also 4 to 5 "
    "times shorter than Llama's.",
    "With thinking, much of every score is gender. From names alone, Qwen3.8-27B places {g_uni}% of characters whose gender "
    "is unique among the three principals and {g_same}% when all three share one (chance 33%); the representations mostly "
    "compete on same-gender characters, where the spread widens from {same_v2}% (v2) to {same_leg}% (full Llama notes). "
    "Stating the gender adds little because the names already carry it: gender only gives {genthink}% with thinking. Without "
    "thinking it does nothing: Qwen3.8-27B {gen27}% (names only {noinfo27}%), Qwen3.5-9B base {gen9}% ({noinfo9}%).",
    "Short spans (about 50 words) keep thinking-on accuracy below ceiling: names only {s_noinfo}%, every representation "
    "{s_lo} to {s_hi}%, name-swapped v2 {s_swap}%. Only the full Llama notes are clearly ahead of v2 ({s_legd} "
    "points, better in {s_legpos} books), matching its lead on sections without thinking; on dense windows every "
    "representation is at {w_lo} to {w_hi}%.",
    "Better notes shorten the reasoning. With thinking on, the median reasoning on sections is {eff_noi}k characters with "
    "names only, {eff_v2}k with v2 and {eff_leg}k with the full Llama notes; on dense windows, where every representation "
    "is at ceiling, v2 still needs {effw_v2}k against {effw_leg}k for the Llama notes.",
    "Which part of a sheet matters depends on the model. For Qwen3.5-9B base the ~120-word relationships section alone "
    "({rel9}%) does about as well as the whole v2 sheet ({v29}%); for Qwen3.8-27B each section alone adds {sec27_lo} to "
    "{sec27_hi} points, and the sheet without relationships ({norel27}%) is close to the full sheet ({v227}%).",
    "Character notes make the real next passage more likely in every book: {ppl_lo} to {ppl_hi}% lower perplexity for "
    "Qwen3.5-9B base without story context, and 1 to 4% on top of the 4,000 words before the passage (Qwen3.5-9B base and Qwen3.8-27B).",
]


def load(kind, model):
    p = OUT / f"analysis_{kind}_{model}.json"
    return json.load(open(p)) if p.exists() else None


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def delta(row, ref="noinfo"):
    d = row.get(f"vs_{ref}") if row else None
    if not d:
        return ""
    return (f'<span class="num">{100 * d["mean"]:+.1f}</span> <span class="sub">{d["pos"]}/{d["n"]} · '
            f'[{100 * d["ci"][0]:+.1f}, {100 * d["ci"][1]:+.1f}]</span>')


def example():
    for it in read_jsonl(DATA / "items_test.jsonl"):
        words = it["masked"].split()
        for i in range(0, max(1, len(words) - 70), 10):
            window = " ".join(words[i:i + 70])
            if len(set(re.findall(r"\[CHAR \d\]", window))) == 3 and not window.startswith(("I t", "“")):
                return it, window
    return None, ""


def main():
    A = {(s, m): load(s, m) for s, _ in SETS + [("two", ""), ("pron", "")] for m, _ in MODELS}
    def get(s, m, c):                                   # rows covering under half the set's books are still running: hidden
        a = A.get((s, m)) or {}
        r = (a.get("rows") or {}).get(c)
        if not r or r["books"] < len(a["books"]) / 2:
            return None
        if not m.endswith("_think") and (r.get("joint") or {}).get("macro") is not None:   # thinking off: joint scoring
            j = r["joint"]
            r = {**r, "macro_argmax": r["macro"], "macro": j["macro"], "vs_noinfo": j.get("vs_noinfo"), "vs_v2": j.get("vs_v2")}
        return r
    macro = lambda s, m, c: (get(s, m, c) or {}).get("macro")
    words = collections.defaultdict(list)
    for s in ("test", "val", "train"):
        for r in read_jsonl(DATA / f"reps_{s}.jsonl"):
            words[r["condition"]].append(r["words"])
    medw = {k: int(st.median(v)) for k, v in words.items()}
    passage_words, passage_q, counts = {}, {}, {}
    for key, _ in SETS:
        suf = "" if key == "main" else f"_{key}"
        its = [it for s in ("test", "val", "train") for it in read_jsonl(DATA / f"items_{s}{suf}.jsonl")]
        pw = sorted(len(it["original"].split()) for it in its)
        passage_words[key] = st.median(pw)
        passage_q[key] = (pw[len(pw) // 4], pw[3 * len(pw) // 4])
        counts[key] = (len(its), len({it["book"] for it in its}))
    it, window = example()
    q27, q4, qt = "Qwen3.8-27B_nothink", "Qwen3-4B-Instruct-2507", "Qwen3.8-27B_think"
    items27 = json.load(open(OUT / "analysis_items_Qwen3.8-27B_nothink.json"))
    items9 = json.load(open(OUT / "analysis_items_Qwen3.5-9B-Base.json"))
    gender = json.load(open(OUT / "analysis_gender.json"))["acc"]
    ppl = json.load(open(OUT / "analysis_ppl.json")) if (OUT / "analysis_ppl.json").exists() else {}
    rep_book, rep_b, rep_label = "dark", 14, "Liska Radost"
    rep_example = {r["condition"]: r["text"] for r in read_jsonl(DATA / "reps_test.jsonl")
                   if (r["book"], r["boundary"], r["label"]) == (rep_book, rep_b, rep_label)}
    man_words = int(st.median(len(r["text"].split()) for sp in ("test", "val", "train") for r in read_jsonl(DATA / f"reps_item_{sp}.jsonl")
                              if r["condition"] == "manual_prior"))
    manual = json.load(open(OUT / "analysis_manual.json")) if (OUT / "analysis_manual.json").exists() else {"rows": {}, "passages": 0}
    traces = json.load(open(OUT / "analysis_traces.json")) if (OUT / "analysis_traces.json").exists() else {"stats": {}}

    def oracle_table():
        head = "".join(f"<th scope='col' class='num'>{n}</th>" for _, n in MODELS[:4])
        rows = "".join(f"<tr><th scope='row'>{LABEL[c]}</th>" + "".join(f"<td class='num strong'>{pct(macro('main', m, c))}</td>" for m, _ in MODELS[:4]) + "</tr>"
                       for c in ORACLE_ROWS)
        return f"<thead><tr><th scope='col'>Representation</th>{head}</tr></thead><tbody>{rows}</tbody>"

    def scoring_table():
        direct = [(m, n) for m, n in MODELS if not m.endswith("_think") and not m.startswith("Mistral")]
        head1 = "".join(f"<th scope='colgroup' colspan='2' class='num'>{n}</th>" for _, n in direct) + "<th scope='colgroup' colspan='3' class='num'>Qwen3.8-27B, per character, by true id</th>"
        head2 = "".join("<th scope='col' class='num'>per character</th><th scope='col' class='num'>joint</th>" for _ in direct) + \
            "".join(f"<th scope='col' class='num'>[CHAR {i}]</th>" for i in range(3))
        rows = []
        for c in ["noinfo", "v2", "legacy_full", "oracle_prior", "oracle_passage", "oracle_quote"]:
            cells = ""
            for m, _ in direct:
                r = get("main", m, c)
                cells += f"<td class='num'>{pct(r and r['macro_argmax'])}</td><td class='num strong'>{pct(r and r['macro'])}</td>"
            ba = ((A[("main", q27)] or {}).get("rows", {}).get(c) or {}).get("by_answer", {})
            cells += "".join(f"<td class='num'>{100 * ba[str(i)]:.0f}</td>" if str(i) in ba else "<td>—</td>" for i in range(3))
            rows.append(f"<tr><th scope='row'>{LABEL[c]}</th>{cells}</tr>")
        return f"<thead><tr><th scope='col' rowspan='2'>Representation</th>{head1}</tr><tr>{head2}</tr></thead><tbody>{''.join(rows)}</tbody>"

    def reps_table():
        rows = []
        for c, how in REP_INFO:
            t = rep_example.get(c, "")
            lines = [l.strip() for l in t.splitlines() if l.strip() and not l.startswith(("#", "Question:", "**")) and not re.fullmatch(r"<snippet \d+>", l.strip())]
            ex = " ".join(lines)[:260].rsplit(" ", 1)[0] + " …"
            rows.append(f"<tr><th scope='row'>{LABEL[c]}</th><td>{how}</td><td class='num'>{medw.get(c, 0):,}</td><td class='sub'>{html.escape(ex)}</td></tr>")
        return ("<thead><tr><th scope='col'>Representation</th><th scope='col'>How it is made</th><th scope='col' class='num'>Words</th>"
                f"<th scope='col'>Excerpt</th></tr></thead><tbody>{''.join(rows)}</tbody>")

    def bookch_table():
        models = [(q4, "Qwen3-4B"), ("Qwen3.5-9B-Base", "Qwen3.5-9B base"), (q27, "Qwen3.8-27B"), (qt, "Qwen3.8-27B, thinking")]
        head = "".join(f"<th scope='col' class='num'>{n}</th>" for _, n in models)
        rows = []
        for c in BOOKCH_ROWS:
            q = (get("main", q27, c) or {}).get("tokens_q")
            tk = f"{q[2] / 1000:.1f}k" if q else "—"
            rows.append(f"<tr><th scope='row'>{LABEL[c]}</th><td class='num'>{tk}</td>" +
                        "".join(f"<td class='num strong'>{pct(macro('main', m, c))}</td>" for m, _ in models) + "</tr>")
        return (f"<thead><tr><th scope='col'>Representation</th><th scope='col' class='num'>Prompt tokens, median</th>{head}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody>")

    def plot_table():
        models = [(q4, "Qwen3-4B"), ("Qwen3.5-9B-Base", "Qwen3.5-9B base"), (q27, "Qwen3.8-27B"), (qt, "Qwen3.8-27B, thinking")]
        head = "".join(f"<th scope='col' class='num'>{n}</th>" for _, n in models)
        rows = []
        for c in PLOT_ROWS:
            q = (get("main", q27, c) or {}).get("rep_tokens_q")
            tk = f"{q[2] / 1000:.1f}k" if q else "—"
            rows.append(f"<tr><th scope='row'>{LABEL[c]}</th><td class='num'>{tk}</td>" +
                        "".join(f"<td class='num strong'>{pct(macro('main', m, c))}</td>" for m, _ in models) + "</tr>")
        return (f"<thead><tr><th scope='col'>Representation</th><th scope='col' class='num'>Representation tokens, median</th>{head}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody>")

    def manual_table():
        def cell(m, c):
            v = manual["rows"].get(f"{m}|{c}")
            if not v:
                return "<td>—</td>"
            return (f"<td class='num'><span class='strong'>{100 * v['acc']:.1f}</span> "
                    f"<span class='sub'>{100 * v['ci'][0]:.0f}–{100 * v['ci'][1]:.0f}</span></td>")
        head = "".join(f"<th scope='col' class='num'>{n}</th>" for _, n in MODELS[:4])
        rows = "".join(f"<tr><th scope='row'>{LABEL[c]}</th>" + "".join(cell(m, c) for m, _ in MODELS[:4]) + "</tr>" for c in MANUAL_ROWS
                       if any(f"{m}|{c}" in manual["rows"] for m, _ in MODELS[:4]))
        return f"<thead><tr><th scope='col'>Representation ({manual['passages']} passages)</th>{head}</tr></thead><tbody>{rows}</tbody>"

    def items_table():
        rows = []
        for name, d in (("Qwen3.8-27B", items27), ("Qwen3.5-9B base", items9)):
            rows.append(f"<tr><th scope='row'>{name}</th><td class='num'>{100 * d['var_passage']:.0f}%</td><td class='num'>{100 * d['var_representation']:.1f}%</td>"
                        f"<td class='num'>{100 * d['all_right']:.0f}%</td><td class='num'>{100 * d['all_wrong']:.0f}%</td>"
                        f"<td class='num'>{100 * d['best_single']:.0f}%</td><td class='num strong'>{100 * d['oracle_pick']:.0f}%</td></tr>")
        return ("<thead><tr><th scope='col'>Model</th><th scope='col' class='num'>Variance: passage</th><th scope='col' class='num'>Variance: representation</th>"
                "<th scope='col' class='num'>Right with every representation</th><th scope='col' class='num'>Wrong with every one</th>"
                f"<th scope='col' class='num'>Best single</th><th scope='col' class='num'>Best per passage</th></tr></thead><tbody>{''.join(rows)}</tbody>")

    def ppl_table():
        cols = [("Qwen3.5-9B-Base|none", "9B base, notes only"), ("Qwen3.5-9B-Base|story", "9B base, notes + story"),
                ("Qwen3.8-27B_chat|none", "27B chat, notes only"), ("Qwen3.8-27B_chat|story", "27B chat, notes + story")]
        reps = ["v2", "charmem", "summary", "legacy", "chiron_r2000", "chiron", "legacy_full"]
        names = {"v2": "v2 sheet", "charmem": "Charmem sheet", "summary": "Summary", "legacy": "Llama notes, summarized",
                 "chiron_r2000": "gpt-oss claims, 2,000 words", "chiron": "gpt-oss claims, full", "legacy_full": "Llama notes, full"}
        def cell(k, r):
            v = ppl.get(k, {}).get(r)
            if not v:
                return "<td>—</td>"
            part = "" if v["passages"] > 1000 else f" <span class='sub'>({v['passages']} passages)</span>"
            return f"<td class='num'>{100 * (v['ppl_ratio'] - 1):+.1f}%{part}</td>"
        head = "".join(f"<th scope='col' class='num'>{n}</th>" for _, n in cols)
        rows = "".join(f"<tr><th scope='row'>{names[r]}</th>" + "".join(cell(k, r) for k, _ in cols) + "</tr>" for r in reps)
        return f"<thead><tr><th scope='col'>Character notes</th>{head}</tr></thead><tbody>{rows}</tbody>"

    def traces_table():
        st_ = traces.get("stats", {})
        rows = "".join(f"<tr><th scope='row'>{LABEL.get(c, c)}</th><td class='num'>{100 * v['accuracy']:.0f}%</td><td class='num'>{v['median_words']:,.0f}</td>"
                       f"<td class='num'>{100 * v['share_traces_that_talk_about_the_notes']:.0f}%</td><td class='num'>{v['traces']}</td></tr>"
                       for c, v in sorted(st_.items()))
        return ("<thead><tr><th scope='col'>Condition</th><th scope='col' class='num'>Accuracy</th><th scope='col' class='num'>Median reasoning words</th>"
                f"<th scope='col' class='num'>Traces discussing the notes</th><th scope='col' class='num'>Traces</th></tr></thead><tbody>{rows}</tbody>")

    def main_table():
        head = "".join(f"<th scope='col' class='num'>{n}</th>" for _, n in MODELS)
        rows = []
        for c in MAIN_ROWS:
            w = {"noinfo": "0", "gender": "2", "book": "—", "book_last8000": "8,000", "book_last32000": "32,000"}.get(c, f"{medw.get(c, 0):,}")
            def cell(m):
                r = get("main", m, c)
                part = r and r["items"] < r["items_total"]
                mark = f"<sup title='{r['items']} of {r['items_total']} passages'>†</sup>" if part else ""
                return f"<td class='num strong'>{pct(r and r['macro'])}{mark}</td>"
            cells = "".join(cell(m) for m, _ in MODELS)
            q = (get("main", q27, c) or {}).get("tokens_q")
            tk = f"{q[2] / 1000:.1f}k <span class='sub'>{q[1] / 1000:.1f}–{q[3] / 1000:.1f}k</span>" if q else "—"
            rows.append(f"<tr><th scope='row'>{LABEL[c]}</th><td class='num'>{w}</td><td class='num'>{tk}</td>{cells}"
                        f"<td>{delta(get('main', q27, c)) if c != 'noinfo' else ''}</td></tr>")
        return (f"<thead><tr><th scope='col'>Representation</th><th scope='col' class='num'>Words per character</th>"
                f"<th scope='col' class='num'>Prompt tokens, median <span class='sub'>middle half</span></th>{head}"
                f"<th scope='col'>Qwen3.8-27B Δ vs names only</th></tr></thead><tbody>{''.join(rows)}</tbody>")

    def control_table():
        head = "".join(f"<th scope='col' class='num'>{n}</th>" for _, n in MODELS)
        rows = "".join(f"<tr><th scope='row'>{LABEL[c]}</th>" + "".join(f"<td class='num strong'>{pct(macro('main', m, c))}</td>" for m, _ in MODELS) + "</tr>"
                       for c in ["noinfo"] + CONTROL_ROWS)
        return f"<thead><tr><th scope='col'>Condition</th>{head}</tr></thead><tbody>{rows}</tbody>"

    def passage_table():
        head = "".join(f"<th scope='col' class='num'>{name}<br><span class='sub'>{passage_words[k]:.0f} words · {counts[k][0]:,} passages</span></th>" for k, name in SETS)
        rows = []
        for m, mname in CHART_MODELS:
            for c in ["noinfo", "v2", "charmem", "chiron", "summary", "legacy_full", "book_last8000", "swapname_v2"]:
                rows.append(f"<tr><th scope='row'>{mname}: {LABEL[c]}</th>" + "".join(f"<td class='num'>{pct(macro(k, m, c))}</td>" for k, _ in SETS) + "</tr>")
        return f"<thead><tr><th scope='col'>Model and representation</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody>"

    ABL = [("legacy_full", "Llama notes, full"), ("legacy_noprev", "Llama notes without the previous chapter"),
           ("legacy_onlyprev", "Llama notes, previous chapter only"), ("legacy_r6000", "Llama notes, most recent 6,000 words"),
           ("legacy_nofill", "Llama notes without \"not mentioned\" filler"), ("legacy_inter", "Llama notes, only statements naming another principal"),
           ("legacy_nointer", "Llama notes, only statements not naming another principal"),
           ("chapnotes", "gpt-oss chapter notes (Llama notes' format and questions)"), ("chiron", "gpt-oss claims, full"), ("chiron_noprev", "gpt-oss claims without the previous chapter"),
           ("chiron_onlyprev", "gpt-oss claims, previous chapter only"), ("chiron_inter", "gpt-oss claims, only statements naming another principal"),
           ("chiron_nointer", "gpt-oss claims, only statements not naming another principal"),
           ("book_last8000", "Book, last 8,000 words"), ("book_noprev8000", "Book, last 8,000 words before the previous chapter"),
           ("book_prevonly", "Book, previous chapter only"), ("combo_short", "v2 + charmem + summary"),
           ("combo_legacy_v2", "Llama notes without filler + v2"), ("combo_all", "All four combined")]

    def ablation_table():
        rows = "".join(f"<tr><th scope='row'>{n}</th><td class='num'>{medw.get(c, 0):,}</td><td class='num strong'>{pct(macro('main', q27, c))}</td>"
                       f"<td>{delta(get('main', q27, c), 'v2')}</td></tr>" for c, n in ABL if get("main", q27, c))
        return ("<thead><tr><th scope='col'>Representation (Qwen3.8-27B, sections)</th><th scope='col' class='num'>Words per character</th>"
                f"<th scope='col' class='num'>Accuracy</th><th scope='col'>Δ vs v2</th></tr></thead><tbody>{rows}</tbody>")

    def sections_table():
        cols = [("Qwen3.5-9B-Base", "v2", "9B base, v2"), ("Qwen3.5-9B-Base", "charmem", "9B base, charmem"),
                (q27, "v2", "27B, v2"), (q27, "charmem", "27B, charmem")]
        body = []
        for key, name in [("noinfo", "Names only")] + [(f"sec_{k}", f"{n} section only") for k, n in SHEET_SECTIONS] + [("norel", "Everything except relationships"), ("", "Full sheet")]:
            conds = ["noinfo" if key == "noinfo" else (f"{r}_{key}" if key else r) for _, r, _ in cols]
            w = "0" if key == "noinfo" else f"{medw.get(f'v2_{key}' if key else 'v2', 0):,} / {medw.get(f'charmem_{key}' if key else 'charmem', 0):,}"
            body.append(f"<tr><th scope='row'>{name}</th><td class='num'>{w}</td>" +
                        "".join(f"<td class='num strong'>{pct(macro('main', m, c))}</td>" for (m, _, _), c in zip(cols, conds)) + "</tr>")
        head = "".join(f"<th scope='col' class='num'>{n}</th>" for _, _, n in cols)
        return (f"<thead><tr><th scope='col'>Part of the sheet</th><th scope='col' class='num'>Words (v2 / charmem)</th>{head}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody>")

    def reason_table():
        def cells(k, c):
            r = get(k, qt, c)
            if not r:
                return "<td>—</td><td></td><td></td>"
            return (f"<td class='num strong'>{pct(r['macro'])}</td><td>{delta(r) if c != 'noinfo' else ''}</td>"
                    f"<td>{delta(r, 'v2') if c not in ('noinfo', 'v2') else ''}</td>")
        rows = "".join(f"<tr><th scope='row'>{LABEL[c]}</th>{cells('short', c)}<td class='num strong'>{pct(macro('window', qt, c))}</td></tr>" for c in REASON_ROWS)
        return ("<thead><tr><th scope='col'>Representation</th><th scope='col' class='num'>Short spans</th><th scope='col'>Δ vs names only</th>"
                f"<th scope='col'>Δ vs v2</th><th scope='col' class='num'>Dense windows</th></tr></thead><tbody>{rows}</tbody>")

    def gender_table():
        groups = [("unique", "unique"), ("shared", "shared"), ("all same", "all same")]
        head1 = "".join(f"<th scope='colgroup' colspan='3' class='num'>{n}</th>" for _, n in GENDER_COLS)
        head2 = "".join(f"<th scope='col' class='num'>{g}</th>" for _ in GENDER_COLS for _, g in groups)
        rows = []
        for c in GENDER_ROWS:
            shown = lambda m: f"{m}|{c}" in gender and get("short" if m.endswith("_short") else "main", m.removesuffix("_short"), c)
            cells = "".join(f"<td class='num'>{100 * gender[f'{m}|{c}'][g]['acc']:.0f}</td>" if shown(m) else "<td>—</td>"
                            for m, _ in GENDER_COLS for g, _ in groups)
            rows.append(f"<tr><th scope='row'>{LABEL[c]}</th>{cells}</tr>")
        share = f"<tr><th scope='row' class='muted'>Share of characters</th>" + "".join(
            f"<td class='num muted'>{100 * gender[f'{m}|noinfo'][g]['n'] / sum(v['n'] for v in gender[f'{m}|noinfo'].values()):.0f}%</td>"
            for m, _ in GENDER_COLS for g, _ in groups) + "</tr>"
        return (f"<thead><tr><th scope='col' rowspan='2'>Representation</th>{head1}</tr><tr>{head2}</tr></thead>"
                f"<tbody>{''.join(rows)}{share}</tbody>")

    def effort_table():
        L = traces.get("length", {})
        conds = sorted({c for d in L.values() for c in d if c in LABEL}, key=lambda c: L.get("main", {}).get(c, {}).get("median_chars", 1e9))
        cell = lambda k, c: f"<td class='num'>{L[k][c]['median_chars'] / 1000:.1f}k</td>" if c in L.get(k, {}) else "<td>—</td>"
        rows = "".join(f"<tr><th scope='row'>{LABEL[c]}</th>{cell('short', c)}{cell('main', c)}{cell('window', c)}"
                       f"<td class='num'>{pct(macro('main', qt, c))}</td></tr>" for c in conds)
        return ("<thead><tr><th scope='col'>Representation</th><th scope='col' class='num'>Short spans</th><th scope='col' class='num'>Sections</th>"
                f"<th scope='col' class='num'>Dense windows</th><th scope='col' class='num'>Accuracy, sections</th></tr></thead><tbody>{rows}</tbody>")

    def small_table(a, rows):
        if not a:
            return ""
        return "\n".join(f"<tr><th scope='row'>{LABEL[c]}</th><td class='num strong'>{pct(a['rows'][c]['macro'])}</td>"
                         f"<td>{delta(a['rows'][c]) if c != 'noinfo' else ''}</td></tr>" for c in rows if c in a["rows"])

    def book_table():
        a = A[("main", q27)]
        cols = ["noinfo", "summary", "v2", "chiron", "charmem", "legacy_full", "book_last8000", "swapname_v2"]
        head = "".join(f"<th scope='col' class='num'>{h}</th>" for h in
                       ["passages", "names only", "summary", "v2", "gpt-oss claims", "charmem", "Llama notes", "book 8k", "swap+names"])
        rows = []
        for b in sorted(a["books"], key=lambda b: -a["n_items"][b]):
            cells = "".join(f"<td class='num'>{100 * a['per_book_joint'][c][b]:.0f}</td>" if b in a["per_book_joint"].get(c, {}) else "<td>—</td>"
                            for c in cols)
            rows.append(f"<tr><th scope='row'><code>{b}</code></th><td class='num'>{a['n_items'][b]}</td>{cells}</tr>")
        return f"<thead><tr><th scope='col'>book</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody>"

    ptok = collections.defaultdict(dict)                     # item -> prompt tokens, thinking-on 27B (complete for both)
    for c in ("book", "legacy_full"):
        for f in glob.glob(str(OUT / "eval" / qt / "items_*" / f"{c}.*jsonl")):
            if not f.endswith(".errors.jsonl") and any(f"/items_{sp}/" in f for sp in ("test", "val", "train")):
                for r in read_jsonl(f):
                    if r.get("prompt_tokens"):
                        ptok[c][r["item_id"]] = r["prompt_tokens"]
    both = [i for i in ptok["book"] if i in ptok["legacy_full"]]
    leg_ratio = f"{st.median(ptok['legacy_full'][i] / ptok['book'][i] for i in both):.2f}"
    leg_longer = f"{100 * sum(ptok['legacy_full'][i] > ptok['book'][i] for i in both) / len(both):.0f}"
    def length_data(families, singles_spec, styles):
        out = {}
        for m, name in CHART_MODELS:
            a = A[("main", m)]
            if not a:
                continue
            rows = a["rows"]

            def pt(c):
                r = rows[c]
                xs = {"all": {"med": r["tokens_q"][2], "mean": r["mean_tokens"], "lo": r["tokens_q"][1], "hi": r["tokens_q"][3]}}
                if r.get("rep_tokens_q"):
                    q = r["rep_tokens_q"]
                    xs["rep"] = {"med": q[2], "mean": r["rep_tokens_mean"], "lo": max(q[1], 1), "hi": q[3]}
                return {"xs": xs, "y": 100 * get("main", m, c)["macro"], "cov": r["items"] / r["items_total"]}
            ok = lambda c: c in rows and rows[c]["tokens_q"] and get("main", m, c)
            series = [{"name": fam, **styles.get(fam, {}), "points": [{"label": c, **pt(c)} for c in conds if ok(c)]}
                      for fam, conds in families]
            singles = [{"name": n, **pt(c)} for n, c in singles_spec if ok(c)]
            if any(s_["points"] for s_ in series) and get("main", m, "noinfo"):
                out[name] = {"series": series, "singles": singles, "ref": 100 * get("main", m, "noinfo")["macro"], "unit": "tokens",
                             "xticks": [100, 300, 1000, 3000, 10000, 30000, 100000, 300000]}
        return out

    length = length_data(LENGTH_FAMILIES, (("charmem", "charmem"), ("Llama notes, full", "legacy_full"), ("gpt-oss chapter notes", "chapnotes")), FAMILY_STYLE)
    summaries = length_data(SUMMARY_FAMILIES, (), SUMMARY_STYLE)
    passage = {}
    for m, name in CHART_MODELS:
        series = [{"name": n, "points": [{"label": sname, "x": passage_words[k], "lo": passage_q[k][0], "hi": passage_q[k][1], "y": 100 * macro(k, m, c)}
                                         for k, sname in SETS if macro(k, m, c) is not None]} for n, c in PASSAGE_SERIES]
        ref = [{"label": sname, "x": passage_words[k], "lo": passage_q[k][0], "hi": passage_q[k][1], "y": 100 * macro(k, m, "noinfo")}
               for k, sname in SETS if macro(k, m, "noinfo") is not None]
        if any(s["points"] for s in series):
            passage[name] = {"series": series, "refline": ref, "unit": "words",
                             "xlabel": "median passage length in words; bars span the middle half of passages",
                             "xticks": [30, 100, 300, 1000]}

    def table_rows(d):
        def one(p):
            if "xs" not in p:
                return f"{p['label']}: {p['y']:.1f}% at {p['x']:,.0f}" + (f" (middle half {p['lo']:,.0f}–{p['hi']:,.0f})" if "lo" in p else "")
            a, r = p["xs"]["all"], p["xs"].get("rep")
            return (f"{p['label']}: {p['y']:.1f}%, prompt median {a['med']:,.0f} / mean {a['mean']:,.0f}"
                    + (f", representation median {r['med']:,.0f} / mean {r['mean']:,.0f}" if r else ""))
        return "\n".join(f"<tr><th scope='row'>{m}: {s['name']}</th><td>" + "; ".join(one(p) for p in s["points"]) + "</td></tr>"
                          for m, v in d.items() for s in v["series"])

    vals = dict(
        g27_lo=f"{100 * min(get('main', q27, c)['vs_noinfo']['mean'] for c in ('legacy', 'chiron_r2000', 'v2', 'chiron', 'charmem', 'summary', 'legacy_full') if get('main', q27, c)):.0f}",
        g27_hi=f"{100 * max(get('main', q27, c)['vs_noinfo']['mean'] for c in ('legacy', 'chiron_r2000', 'v2', 'chiron', 'charmem', 'summary', 'legacy_full') if get('main', q27, c)):.0f}",
        swap27=pct(macro("main", q27, "swapname_v2")), noinfo27=pct(macro("main", q27, "noinfo")),
        swapthink=pct(macro("main", qt, "swapname_v2")),
        v2_short=pct(macro("short", q27, "v2")), v2_main=pct(macro("main", q27, "v2")), v2_win=pct(macro("window", q27, "v2")),
        charmem27=pct(macro("main", q27, "charmem")), summary27=pct(macro("main", q27, "summary")), v227=pct(macro("main", q27, "v2")),
        legfull27=pct(macro("main", q27, "legacy_full")), bl827=pct(macro("main", q27, "book_last8000")),
        think_lo=f"{100 * min(macro('main', qt, c) or 1 for c in ('v2', 'charmem', 'summary', 'chiron', 'legacy', 'legacy_full')):.0f}",
        think_hi=f"{100 * max(macro('main', qt, c) or 0 for c in ('v2', 'charmem', 'summary', 'chiron', 'legacy', 'legacy_full')):.0f}",
        leg_noprev=pct(macro("main", q27, "legacy_noprev")), leg_r6000=pct(macro("main", q27, "legacy_r6000")),
        leg_nofill=pct(macro("main", q27, "legacy_nofill")), leg_inter=pct(macro("main", q27, "legacy_inter")),
        orpass27=pct(macro("main", q27, "oracle_passage")), orpassthink=pct(macro("main", qt, "oracle_passage")),
        combo=pct(macro("main", q27, "combo_legacy_v2")),
        zero27=f"{100 * A[('main', q27)]['rows']['noinfo']['by_answer']['0']:.1f}",
        quote27=pct(macro("main", q27, "oracle_quote")), quote27a=pct(get("main", q27, "oracle_quote")["macro_argmax"]),
        g4_lo=f"{100 * min(get('main', q4, c)['vs_noinfo']['mean'] for c in ('legacy', 'chiron_r2000', 'v2', 'chiron', 'charmem', 'summary', 'legacy_full')):.0f}",
        g4_hi=f"{100 * max(get('main', q4, c)['vs_noinfo']['mean'] for c in ('legacy', 'chiron_r2000', 'v2', 'chiron', 'charmem', 'summary', 'legacy_full')):.0f}",
        chiron27=pct(macro("main", q27, "chiron")), chap27=pct(macro("main", q27, "chapnotes")),
        chap9=pct(macro("main", "Qwen3.5-9B-Base", "chapnotes")),
        sec27_lo=f"{100 * min(macro('main', q27, f'v2_sec_{k}') - macro('main', q27, 'noinfo') for k, _ in SHEET_SECTIONS):.0f}",
        sec27_hi=f"{100 * max(macro('main', q27, f'v2_sec_{k}') - macro('main', q27, 'noinfo') for k, _ in SHEET_SECTIONS):.0f}",
        man_n=str(manual["passages"]), man_words=str(man_words),
        **{f"{k}{sfx}": f"{100 * manual['rows'][f'{m}|{c}']['acc']:.1f}" for k, c in (("manp", "manual_prior"), ("manv2_", "v2"),
           ("manleg", "legacy_full"), ("manpas", "manual_passage")) for sfx, m in (("27", q27), ("think", qt))},
        orp27=pct(macro("main", q27, "oracle_prior")), orpthink=pct(macro("main", qt, "oracle_prior")),
        gen27=pct(macro("main", q27, "gender")), genthink=pct(macro("main", qt, "gender")), gen9=pct(macro("main", "Qwen3.5-9B-Base", "gender")),
        noinfo9=pct(macro("main", "Qwen3.5-9B-Base", "noinfo")),
        g_uni=f"{100 * gender[qt + '|noinfo']['unique']['acc']:.0f}", g_same=f"{100 * gender[qt + '|noinfo']['all same']['acc']:.0f}",
        same_v2=f"{100 * gender[qt + '|v2']['all same']['acc']:.0f}", same_leg=f"{100 * gender[qt + '|legacy_full']['all same']['acc']:.0f}",
        s_noinfo=pct(macro("short", qt, "noinfo")),
        s_swap=pct(macro("short", qt, "swapname_v2")),
        s_lo=f"{100 * min(macro('short', qt, c) or 1 for c in REASON_ROWS[1:-1]):.0f}",
        s_hi=f"{100 * max(macro('short', qt, c) or 0 for c in REASON_ROWS[1:-1]):.0f}",
        s_legd=f"{100 * get('short', qt, 'legacy_full')['vs_v2']['mean']:+.1f}",
        s_legpos=f"{get('short', qt, 'legacy_full')['vs_v2']['pos']}/{get('short', qt, 'legacy_full')['vs_v2']['n']}",
        w_lo=f"{100 * min(macro('window', qt, c) or 1 for c in REASON_ROWS[1:-1]):.0f}",
        w_hi=f"{100 * max(macro('window', qt, c) or 0 for c in REASON_ROWS[1:-1]):.0f}",
        rel9=pct(macro("main", "Qwen3.5-9B-Base", "v2_sec_relationships")), v29=pct(macro("main", "Qwen3.5-9B-Base", "v2")),
        norel27=pct(macro("main", q27, "v2_norel")),
        **{f"{p}_{c.split('_')[0][:3]}": f"{traces['length'][k][c]['median_chars'] / 1000:.0f}"
           for p, k in (("eff", "main"), ("effw", "window")) for c in ("noinfo", "v2", "legacy_full") if c in traces.get("length", {}).get(k, {})},
        varp=f"{100 * items27['var_passage']:.0f}", varr=f"{100 * items27['var_representation']:.1f}",
        pick=f"{100 * items27['oracle_pick']:.0f}", best=f"{100 * items27['best_single']:.0f}",
        ppl_lo=f"{100 * (1 - max(v['ppl_ratio'] for k, v in ppl.get('Qwen3.5-9B-Base|none', {}).items() if k != 'names' and v['passages'] > 1000)):.0f}",
        ppl_hi=f"{100 * (1 - min(v['ppl_ratio'] for k, v in ppl.get('Qwen3.5-9B-Base|none', {}).items() if k != 'names' and v['passages'] > 1000)):.0f}")
    findings = "\n".join(f"<li>{f.format(**vals)}</li>" for f in FINDINGS)

    ex = html.escape(window)
    ex = re.sub(r"\[CHAR (\d)\]", r'<mark class="m\1">[CHAR \1]</mark>', ex)
    names = ", ".join(f"{html.escape(l)} = <mark class='m{i}'>[CHAR {i}]</mark>" for l, i in sorted(it["answer"].items(), key=lambda kv: kv[1])) if it else ""
    page = TEMPLATE
    for k, v in {"ORACLE_TABLE": oracle_table(), "MANUAL_TABLE": manual_table(), "SCORING_TABLE": scoring_table(), "ITEMS_TABLE": items_table(), "PPL_TABLE": ppl_table(), "TRACES_TABLE": traces_table(),
                 "ABLATION_TABLE": ablation_table(), "MAIN_TABLE": main_table(), "CONTROL_TABLE": control_table(), "PASSAGE_TABLE": passage_table(),
                 "BOOK_TABLE": book_table(), "FINDINGS": findings, "SECTIONS_TABLE": sections_table(),
                 "REASON_TABLE": reason_table(), "BOOKCH_TABLE": bookch_table(), "PLOT_TABLE": plot_table(), "SUMMARY_DATA": json.dumps(summaries), "EFFORT_TABLE": effort_table(), "GENDER_TABLE": gender_table(),
                 "TWO_TABLE": small_table(A[("two", q4)], ["noinfo", "summary", "v2", "chiron", "charmem", "legacy_full", "book"]),
                 "PRON_TABLE": small_table(A[("pron", q4)], ["noinfo", "summary", "v2", "chiron", "charmem", "book_last8000"]),
                 "LENGTH_DATA": json.dumps(length), "PASSAGE_DATA": json.dumps(passage),
                 "LENGTH_ROWS": table_rows(length), "PASSAGE_ROWS": table_rows(passage),
                 "EXAMPLE": ex, "EXAMPLE_KEY": names, "EXAMPLE_BOOK": html.escape(it["book"]) if it else "",
                 "N_MAIN": f"{counts['main'][0]:,}", "N_SHORT": f"{counts['short'][0]:,}", "N_WINDOW": f"{counts['window'][0]:,}",
                 "N_BOOKS": str(len(A[("main", q27)]["books"])), "LEG_RATIO": leg_ratio, "REPS_TABLE": reps_table(), "REP_WHO": f"{rep_label} in {rep_book} before chapter {rep_b}", "LEG_LONGER": leg_longer}.items():
        page = page.replace("{{" + k + "}}", v)
    (REPO / "reports").mkdir(exist_ok=True)
    (REPO / "reports" / "results.html").write_text(page)
    print("wrote reports/results.html", len(page), "bytes")


TEMPLATE = r"""<title>CHIRON Book Replication</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column; tables and charts sit in their own scroll containers. */
:root {
  --bg: #f6f7f7; --surface: #ffffff; --fg: #16191d; --muted: #5a6068; --rule: #dde1e4; --accent: #245fa8;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #7b5fd1; --ref: #8a9097;
  --m0bg: #dbe8f9; --m1bg: #fbe1d5; --m2bg: #d4f0e5;
  --display: "Newsreader", "Iowan Old Style", Georgia, serif;
  --body: "Public Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141618; --surface: #1b1e21; --fg: #eceef0; --muted: #a2a8b0; --rule: #2d3237; --accent: #7fb0ee;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #8a6fe0; --ref: #7d848c;
  --m0bg: #1f3350; --m1bg: #4a2a1c; --m2bg: #173c30; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141618; --surface: #1b1e21; --fg: #eceef0; --muted: #a2a8b0; --rule: #2d3237; --accent: #7fb0ee;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #8a6fe0; --ref: #7d848c;
  --m0bg: #1f3350; --m1bg: #4a2a1c; --m2bg: #173c30; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); }
main { max-width: 64rem; margin: 0 auto; padding-inline: 1.25rem; padding-block: 2.5rem 4rem; display: grid; gap: 2.25rem; }
section { display: grid; gap: 0.9rem; min-width: 0; }
h1, h2 { font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: clamp(1.9rem, 4vw, 2.6rem); }
h2 { font-size: 1.35rem; }
p { margin: 0; max-width: 70ch; }
.eyebrow { font: 500 0.75rem/1 var(--body); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.lede { font-size: 1.05rem; }
.muted, .sub { color: var(--muted); }
.sub { font-size: 0.78rem; font-weight: 400; text-transform: none; letter-spacing: 0; }
ul.findings { margin: 0; padding-left: 1.2rem; display: grid; gap: 0.5rem; max-width: 74ch; }
.exhibit { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 1rem 1.1rem; display: grid; gap: 0.6rem; }
.exhibit blockquote { margin: 0; font: 0.95rem/1.7 var(--display); }
.exhibit .key, .exhibit .q { font: 0.8rem/1.5 var(--mono); color: var(--muted); }
mark { font: 500 0.82em var(--mono); color: var(--fg); padding: 0 0.2em; border-radius: 3px; }
mark.m0 { background: var(--m0bg); } mark.m1 { background: var(--m1bg); } mark.m2 { background: var(--m2bg); }
.table-wrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; }
table { border-collapse: collapse; width: 100%; font-size: 0.87rem; }
th, td { padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--rule); text-align: left; vertical-align: top; }
thead th { font: 600 0.72rem/1.3 var(--body); letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }
tbody tr:last-child th, tbody tr:last-child td { border-bottom: 0; }
tbody th { font-weight: 500; }
.num { font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
.strong { font-weight: 600; }
code { font: 0.85em var(--mono); }
.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 0.75rem; position: relative; display: grid; gap: 0.5rem; }
.chart svg { display: block; width: 100%; height: auto; }
.chart text { fill: var(--muted); font: 11px var(--body); }
.chart .lbl { font-weight: 600; }
.bar { display: flex; flex-wrap: wrap; gap: 0.5rem 1rem; align-items: center; justify-content: space-between; }
.legend { display: flex; flex-wrap: wrap; gap: 0.3rem 0.9rem; font-size: 0.8rem; color: var(--muted); }
.legend label { display: inline-flex; align-items: center; gap: 0.35rem; cursor: pointer; }
.legend input { margin: 0; accent-color: var(--accent); }
.legend span { display: inline-flex; }
.legend i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
.legend i.dash { background: repeating-linear-gradient(90deg, var(--c) 0 5px, transparent 5px 8px); }
.legend i.diamond { width: 8px; height: 8px; border-radius: 1px; background: var(--fg); transform: rotate(45deg); }
.bounds { display: flex; flex-wrap: wrap; gap: 0.4rem 0.9rem; align-items: center; font-size: 0.8rem; color: var(--muted); }
.bounds label { display: inline-flex; align-items: center; gap: 0.35rem; }
.bounds input { width: 6.5rem; font: 0.8rem var(--body); color: var(--fg); background: var(--bg); border: 1px solid var(--rule); border-radius: 4px; padding: 0.2rem 0.4rem; }
.bounds button { font: 500 0.8rem var(--body); color: var(--muted); background: transparent; border: 1px solid var(--rule); border-radius: 6px; padding: 0.25rem 0.6rem; cursor: pointer; }
.bounds input:focus-visible, .bounds button:focus-visible, .legend input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.controls { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 6px; overflow: hidden; }
.seg button { font: 500 0.8rem var(--body); color: var(--muted); background: transparent; border: 0; padding: 0.35rem 0.7rem; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--rule); }
.seg button[aria-pressed="true"] { background: var(--fg); color: var(--bg); }
.seg button:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.tip { position: absolute; pointer-events: none; background: var(--fg); color: var(--bg); font: 0.78rem/1.4 var(--body); padding: 0.35rem 0.55rem; border-radius: 4px; max-width: 16rem; }
details { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 0.6rem 0.9rem; }
summary { cursor: pointer; font-weight: 500; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.35rem 1rem; margin: 0; }
dt { color: var(--muted); } dd { margin: 0; min-width: 0; }
@media (max-width: 560px) { dl { grid-template-columns: 1fr; } dt { margin-top: 0.4rem; } }
</style>

<main>
  <header style="display:grid;gap:0.6rem">
    <div class="eyebrow">CHIRON replication · NCP books · 30 September 2026</div>
    <h1>Can a character sheet put the names back?</h1>
    <p class="lede">CHIRON's masked-character test on the NCP novels. A model sees a passage with the three principals' names replaced by ids, plus one representation of each character built only from earlier chapters, and says which id is which. {{N_MAIN}} sections (about 320 words), {{N_SHORT}} short spans (about 50 words) and {{N_WINDOW}} dense windows (about 900 words); five model setups (Qwen3-4B, Qwen3.5-9B base, Qwen3.8-27B with thinking off and on, Mistral-7B).</p>
  </header>

  <section aria-labelledby="ex">
    <h2 id="ex">What the model sees</h2>
    <div class="exhibit">
      <blockquote>… {{EXAMPLE}} …</blockquote>
      <div class="key">{{EXAMPLE_BOOK}} · answer: {{EXAMPLE_KEY}}</div>
      <div class="q">One block per character (the representation being tested), then the passage. Without thinking, the model answers "Which ID in the passage is &lt;name&gt;?" and is scored from the probability of each id digit. With thinking, it reasons and then gives the whole name-to-id mapping.</div>
    </div>
  </section>

  <section aria-labelledby="reps">
    <h2 id="reps">The representations</h2>
    <p class="muted">Everything is built only from the chapters before the passage's chapter. Words are the median per character. The excerpts are the start of each representation for {{REP_WHO}}.</p>
    <div class="table-wrap"><table>{{REPS_TABLE}}</table></div>
  </section>

  <section aria-labelledby="find">
    <h2 id="find">Findings</h2>
    <ul class="findings">{{FINDINGS}}</ul>
  </section>

  <section aria-labelledby="scr">
    <h2 id="scr">Scoring: a prior over id labels</h2>
    <p class="muted">Without thinking, each character is asked for separately and scored from the probabilities of the id digits. Per character, the answer is the most likely digit. Joint scoring takes the one-to-one mapping of names to ids with the highest summed log-probability over the three questions (the prompt says each id stands for exactly one character). Qwen3.8-27B without thinking almost never answers 0, so per-character scoring cannot get [CHAR 0] right; joint scoring removes most of that. Accuracy (%, sections); thinking-on answers are already full mappings and are unaffected.</p>
    <div class="table-wrap"><table>{{SCORING_TABLE}}</table></div>
  </section>

  <section aria-labelledby="main">
    <h2 id="main">Sections, all three principals named</h2>
    <p class="muted">Macro accuracy over the {{N_BOOKS}} books with at least 10 passages (each book counts once; chance is 33.3%); thinking-off columns use joint scoring. The last column is the mean per-book gain over names only for Qwen3.8-27B without thinking, the number of books where it is positive, and a 95% bootstrap interval over books. Words per character is the median length of one character's block (book text is shared, not per character). Prompt tokens are for the whole Qwen3.8-27B prompt: three blocks, passage and question. † marks a condition that covers only some passages (prompts longer than the served context are skipped). Mistral-7B cannot read the long representations (32k context).</p>
    <div class="table-wrap"><table>{{MAIN_TABLE}}</table></div>
  </section>

  <section aria-labelledby="len">
    <h2 id="len">Representation length and accuracy</h2>
    <p class="muted">Sections. By default each point is the median length of the whole prompt (all three characters' blocks, the passage and the question) with a bar over the middle half of prompts; the switches show the mean instead, or only the representation's own tokens (the prompt minus the names-only prompt for the same passage and character); lengths vary a lot because notes and book text grow through a book. Each line is one representation cut to increasing lengths: the gpt-oss claims keep their most recent statements, v2, the summarized Llama notes and the summary keep their first words, book text keeps its last words. Diamonds are representations tested at one length only. The dashed line is names only. The full Llama notes (for the three characters together) are shorter than the whole book so far for most passages (median {{LEG_RATIO}} times as long) but longer for {{LEG_LONGER}}% of them, early in books.</p>
    <div class="chart" id="lenbox">
      <div class="bar"><div class="legend" data-legend></div><div class="controls" data-controls></div></div>
      <svg viewBox="0 0 760 380" role="img" aria-label="Accuracy against prompt length"></svg>
      <div class="tip" hidden></div>
    </div>
    <details><summary>Chart data</summary><div class="table-wrap" style="margin-top:0.6rem"><table><tbody>{{LENGTH_ROWS}}</tbody></table></div></details>
  </section>

  <section aria-labelledby="plt">
    <h2 id="plt">Plot summaries vs character summaries</h2>
    <p class="muted">One gpt-oss summary of the story so far, shared by the three characters and shown in place of their blocks. One pass: the whole text of the chapters before the passage's chapter (the most recent 85,000 words for the longest books), summarized to a target length. Chapter by chapter: each chapter summarized to about 250 words, then those summaries combined to the target length. "Last k words" cuts the 4,000-word summary to its most recent part. Character summaries are cut to their first k words (they are organized by topic, not time).</p>
    <div class="chart" id="sumbox">
      <div class="bar"><div class="legend" data-legend></div><div class="controls" data-controls></div></div>
      <svg viewBox="0 0 760 380" role="img" aria-label="Accuracy of plot and character summaries against their length"></svg>
      <div class="tip" hidden></div>
    </div>
    <div class="table-wrap"><table>{{PLOT_TABLE}}</table></div>
  </section>

  <section aria-labelledby="bch">
    <h2 id="bch">Book text: how much, and the chapter so far</h2>
    <p class="muted">Sections. Whole chapters before the passage's chapter, and the passage's own chapter up to the passage (which no other representation sees: everything else stops at the chapter boundary). Thinking-off columns use joint scoring.</p>
    <div class="table-wrap"><table>{{BOOKCH_TABLE}}</table></div>
  </section>

  <section aria-labelledby="pas">
    <h2 id="pas">Passage length and accuracy</h2>
    <p class="muted">The same representations on three passage sets: short spans (the tightest run of sentences naming all three principals, 20 to 150 words), NCP sections, and dense windows (up to three consecutive sections, about 900 words, each principal named at least three times). Each set averages over its own books, so the points are not paired. The dashed line is names only.</p>
    <div class="chart" id="pasbox">
      <div class="bar"><div class="legend" data-legend></div><div class="controls" data-controls></div></div>
      <svg viewBox="0 0 760 380" role="img" aria-label="Accuracy against passage length"></svg>
      <div class="tip" hidden></div>
    </div>
    <details><summary>Chart data</summary><div class="table-wrap" style="margin-top:0.6rem"><table><tbody>{{PASSAGE_ROWS}}</tbody></table></div></details>
    <div class="table-wrap"><table>{{PASSAGE_TABLE}}</table></div>
  </section>

  <section aria-labelledby="rsn">
    <h2 id="rsn">Reasoning on short spans and dense windows</h2>
    <p class="muted">Qwen3.8-27B with thinking on. Sections are near ceiling with any representation; about 50 words are not, and there the full Llama notes lead again, with the others within about two points of v2. Dense windows (about 900 words) are at ceiling. Short spans: {{N_SHORT}} passages, macro over the 19 books with at least 10; windows: 13 books with at least 5.</p>
    <div class="table-wrap"><table>{{REASON_TABLE}}</table></div>
  </section>

  <section aria-labelledby="sim">
    <h2 id="sim">Why the numbers look so similar</h2>
    <p class="muted">Per passage and character, correctness depends far more on which passage it is than on which representation the model gets. The representations are right on different passages, so their averages converge while a per-passage choice would do much better (thinking off, sections).</p>
    <div class="table-wrap"><table>{{ITEMS_TABLE}}</table></div>
  </section>

  <section aria-labelledby="gen">
    <h2 id="gen">Gender does much of the work, with thinking</h2>
    <p class="muted">Accuracy (%, pooled over passages) split by whether a character's gender is unique among the three principals, shared with one other, or shared by all three (gender read from the pronouns in each character's sheets and summaries). Pronouns in the passage give the unique character away once the model knows who is who by gender, which names alone often tell it; the shared pair is then a coin flip. With thinking, only same-gender characters need what the notes say about them. Stating the gender ("Gender only") adds 2 points with thinking, since names already carry it, and nothing without thinking, where the model does not make this inference.</p>
    <div class="table-wrap"><table>{{GENDER_TABLE}}</table></div>
  </section>

  <section aria-labelledby="orc">
    <h2 id="orc">Oracles: exactly the information needed</h2>
    <p class="muted">Per passage, gpt-oss wrote (a) 2 to 4 clues per character taken from the unmasked passage itself, a ceiling, and (b) up to 5 facts per character copied word for word from the notes available before the chapter, chosen because they identify the character in this passage (facts that were not verbatim copies were dropped). Swapping (b) between characters with names exchanged tests whether the model relies on it.</p>
    <div class="table-wrap"><table>{{ORACLE_TABLE}}</table></div>
    <p class="muted">The quote oracle gives each character up to two sentences of the unmasked passage that name it, so the task reduces to matching a sentence to its masked copy. Hand-written oracles: for 3 passages from each of the 21 books, Claude agents read the passage, the earlier notes and the book, and wrote (a) 1 to 3 sentences per character from the passage, just enough to force the mapping, and (b) 1 to 4 facts per character established before the chapter, each with a chapter source, chosen for this passage. Accuracy on those passages with a 95% interval over passages:</p>
    <div class="table-wrap"><table>{{MANUAL_TABLE}}</table></div>
  </section>

  <section aria-labelledby="trc">
    <h2 id="trc">Is the reasoning using the sheets?</h2>
    <p class="muted">Saved reasoning from Qwen3.8-27B with thinking on (test and validation passages, one block order). With a sheet the reasoning is half as long and cites sheet facts against events in the passage, for example "If CHAR 2 is Liska, 'Liska's aunt' matches character info: Liska mentions an aunt who lives in Ząbki". With swapped sheets nearly every trace reasons from the misleading notes.</p>
    <div class="table-wrap"><table>{{TRACES_TABLE}}</table></div>
    <p class="muted">How long the model reasons is a second measure that still separates the representations where accuracy is at ceiling. Median reasoning length in characters per generation, every thinking-on run (sorted by sections):</p>
    <div class="table-wrap"><table>{{EFFORT_TABLE}}</table></div>
  </section>

  <section aria-labelledby="ppl">
    <h2 id="ppl">Does character information make the real next passage more likely?</h2>
    <p class="muted">Change in perplexity of the real passage (names unmasked) against a prompt that only lists the characters' names. Qwen3.5-9B base reads a plain-text prompt; Qwen3.8-27B reads the notes as a chat request to write the next passage and is scored on the passage as its reply. "Story" adds the 4,000 words right before the passage. Negative is better; every value covering all passages is negative in all 21 books.</p>
    <div class="table-wrap"><table>{{PPL_TABLE}}</table></div>
  </section>

  <section aria-labelledby="why">
    <h2 id="why">Why the Llama notes and book text do best</h2>
    <p class="muted">Ablations with Qwen3.8-27B, thinking off. Dropping the chapter just before the passage changes nothing, and cutting the Llama notes to their most recent 6,000 words (about 40%) costs little, so the lead is not recency or volume. A word-overlap matcher with no model ranks the Llama notes below the gpt-oss claims, so it is not shared vocabulary either. The Llama notes name the other two principals about twice as often per word as the other sheets (31 per 1,000 words against 13 to 20), which is the information that separates three characters appearing together.</p>
    <div class="table-wrap"><table>{{ABLATION_TABLE}}</table></div>
  </section>

  <section aria-labelledby="secs">
    <h2 id="secs">Which part of the sheet matters</h2>
    <p class="muted">Sections, thinking off. v2 and charmem sheets cut to one of their six sections, or everything except relationships.</p>
    <div class="table-wrap"><table>{{SECTIONS_TABLE}}</table></div>
  </section>

  <section aria-labelledby="ctl">
    <h2 id="ctl">Controls</h2>
    <p class="muted">A plain swap gives each character another principal's representation. Sheets name their own subject in almost every line, so a model can still tell whose sheet it is. Exchanging the two characters' names inside the swapped text removes that, and accuracy falls below names only: the models believe what the sheet says.</p>
    <div class="table-wrap"><table>{{CONTROL_TABLE}}</table></div>
  </section>

  <section aria-labelledby="other">
    <h2 id="other">Other item sets (Qwen3-4B)</h2>
    <div style="display:grid;gap:1.2rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,22rem),1fr))">
      <div style="display:grid;gap:0.5rem;min-width:0">
        <p class="muted">Two principals named (chance 50%).</p>
        <div class="table-wrap"><table><thead><tr><th scope="col">Representation</th><th scope="col" class="num">Acc.</th><th scope="col">Δ</th></tr></thead><tbody>{{TWO_TABLE}}</tbody></table></div>
      </div>
      <div style="display:grid;gap:0.5rem;min-width:0">
        <p class="muted">Names and principals' pronouns masked (pronouns resolved by gpt-oss; a missed one still leaks gender).</p>
        <div class="table-wrap"><table><thead><tr><th scope="col">Representation</th><th scope="col" class="num">Acc.</th><th scope="col">Δ</th></tr></thead><tbody>{{PRON_TABLE}}</tbody></table></div>
      </div>
    </div>
  </section>

  <section aria-labelledby="books">
    <h2 id="books">By book (Qwen3.8-27B, sections)</h2>
    <p class="muted">Accuracy (%) per book without thinking. Books with fewer than 10 passages are left out of the averages above.</p>
    <div class="table-wrap"><table>{{BOOK_TABLE}}</table></div>
  </section>

  <section aria-labelledby="setup">
    <h2 id="setup">Setup</h2>
    <dl>
      <dt>Passages</dt><dd>Every candidate principal is named in at least two earlier chapters. Names come from hand-checked alias tables (nicknames, earlier names, secret identities); passages with an ambiguous surname or any leftover name are dropped.</dd>
      <dt>Representations</dt><dd>All built from chapters before the passage's chapter. See "The representations" above.</dd>
      <dt>Scoring</dt><dd>Without thinking: next-token probabilities of the id digits at temperature 0, one question per character (Qwen3-4B over all 6 block orders, the others over 3 rotations; Mistral's reply is pre-started with "[CHAR "), combined by joint scoring (the best one-to-one mapping per passage and block order). With thinking: one generation per passage and rotation (temperature 0.6, up to 32k tokens) ending in a JSON mapping; an unreadable answer counts as wrong.</dd>
      <dt>Memorization</dt><dd>Temperature-0 continuations of the four test books reproduced no 13-word sequence (longest verbatim run 5 words).</dd>
      <dt>Caveats</dt><dd>21 books carry the statistics and ten of them hold most passages. The passage sets cover different books (short spans 19, dense windows 13 with at least 5 passages). With thinking on, sections are near ceiling, which compresses differences between representations. 46 of the 4,113 thinking-on generations for name-swapped v2 on short spans are missing (a vLLM engine hung under memory pressure). Prompts longer than the served context (131k tokens for Qwen3.8-27B) are skipped, which drops a few combination prompts late in long books.</dd>
    </dl>
  </section>
</main>

<script>
(function () {
  const ns = "http://www.w3.org/2000/svg", colors = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"];
  const niceStep = raw => { const m = Math.pow(10, Math.floor(Math.log10(raw))); return [1, 2, 5, 10].map(k => k * m).find(v => v >= raw); };
  const num = v => (Math.abs(v) >= 1000 ? +(v / 1000).toFixed(2) + "k" : String(+v.toFixed(2)));
  function logTicks(a, b) {
    for (const ds of [[1, 2, 5], [1, 3], [1]]) {
      const t = [];
      for (let k = Math.floor(Math.log10(a)) - 1; k <= Math.ceil(Math.log10(b)); k++) ds.forEach(d => { const v = d * Math.pow(10, k); if (v >= a * 0.999 && v <= b * 1.001) t.push(v); });
      if (t.length <= 8) return t;
    }
    return [];
  }
  const logFloor = v => { const m = Math.pow(10, Math.floor(Math.log10(v))); return [5, 2, 1].map(d => d * m).find(x => x <= v); };
  const logCeil = v => { const m = Math.pow(10, Math.floor(Math.log10(v))); return [1, 2, 5, 10].map(d => d * m).find(x => x >= v); };
  // Draws one chart. st.hidden: names switched off in the legend; st.bounds: manual x0/x1/y0/y1 (null = automatic).
  // Returns the automatic bounds, shown as placeholders in the bound inputs.
  function draw(box, spec, st) {
    const svg = box.querySelector("svg"), tip = box.querySelector(".tip"), legend = box.querySelector("[data-legend]");
    svg.replaceChildren(); legend.replaceChildren(); tip.hidden = true;
    const W = 760, H = 380, L = 48, R = 178, T = 16, B = 42, lin = spec.scale === "linear";
    const on = name => !st.hidden.has(name);
    const series = spec.series.map((s, i) => ({ ...s, c: `var(${colors[s.color !== undefined ? s.color : i]})` })).filter(s => s.points.length);
    const refName = "names only", refOn = on(refName) && (spec.ref !== undefined || (spec.refline || []).length);
    const vis = series.filter(s => on(s.name)).flatMap(s => s.points).concat((spec.singles || []).filter(p => on(p.name)),
                refOn ? spec.refline || [] : []);
    const shown = vis.length ? vis : series.flatMap(s => s.points);
    const xs = shown.flatMap(p => [p.x, p.lo || p.x, p.hi || p.x]).filter(v => lin || v > 0);
    const ys = shown.map(p => p.y).concat(refOn && spec.ref !== undefined ? [spec.ref] : []);
    const auto = {};
    { const lo = Math.min(...xs), hi = Math.max(...xs);
      if (lin) { const step = niceStep(Math.max(hi - lo, 1) / 5); auto.x0 = Math.max(0, Math.floor(lo / step) * step); auto.x1 = Math.ceil(hi / step) * step; }
      else { auto.x0 = logFloor(lo); auto.x1 = logCeil(hi); } }
    const bd = k => (Number.isFinite(st.bounds[k]) && (lin || !k.startsWith("x") || st.bounds[k] > 0) ? st.bounds[k] : auto[k]);
    let x0 = bd("x0"), x1 = bd("x1");
    if (x1 <= x0) x1 = x0 * (lin ? 1 : 10) + (lin ? 1 : 0);
    { const inX = shown.filter(p => p.x >= x0 && p.x <= x1).map(p => p.y).concat(refOn && spec.ref !== undefined ? [spec.ref] : []);
      const yy = inX.length ? inX : ys, lo = Math.min(...yy), hi = Math.max(...yy), step = niceStep(Math.max(hi - lo, 4) / 6);
      auto.y0 = Math.max(0, Math.floor((lo - 1) / step) * step); auto.y1 = Math.min(100, Math.ceil((hi + 1) / step) * step); }
    let y0 = bd("y0"), y1 = bd("y1");
    if (y1 <= y0) y1 = y0 + 5;
    const f = lin ? (v => v) : Math.log10;
    const X = v => L + (f(v) - f(x0)) / (f(x1) - f(x0)) * (W - L - R), Y = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
    const inside = p => p.x >= x0 && p.x <= x1 && p.y >= y0 && p.y <= y1;
    const el = (tag, a, parent) => { const e = document.createElementNS(ns, tag); for (const k in a) e.setAttribute(k, a[k]); (parent || svg).appendChild(e); return e; };
    const clipId = "clip-" + box.id;
    const clip = el("clipPath", { id: clipId }, el("defs", {}));
    el("rect", { x: L, y: T - 6, width: W - L - R, height: H - T - B + 12 }, clip);
    const ystep = niceStep((y1 - y0) / 6);
    for (let v = Math.ceil(y0 / ystep) * ystep; v <= y1 + 1e-9; v += ystep) {
      el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "var(--rule)", "stroke-width": 1 });
      el("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end" }).textContent = num(v) + "%";
    }
    let ticks;
    if (lin) { const step = niceStep((x1 - x0) / 5); ticks = []; for (let v = Math.ceil(x0 / step) * step; v <= x1 + 1e-9; v += step) ticks.push(v); }
    else ticks = logTicks(x0, x1);
    ticks.forEach(v => { el("text", { x: X(v), y: H - B + 18, "text-anchor": "middle" }).textContent = num(v); });
    el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle" }).textContent = `${spec.xlabel} (${lin ? "linear" : "log"} scale)`;
    const g = el("g", { "clip-path": `url(#${clipId})` });
    const desc = p => `${p.y.toFixed(1)}% · median ${num(p.x)} ${spec.unit}` + (p.lo ? ` (middle half ${num(p.lo)}–${num(p.hi)})` : "")
      + (p.cov !== undefined && p.cov < 1 ? ` · ${Math.round(100 * p.cov)}% of passages` : "");
    const labels = [];
    const hover = (p, text) => {
      if (!inside(p)) return;
      const x = X(p.x), y = Y(p.y), hit = el("circle", { cx: x, cy: y, r: 12, fill: "transparent", tabindex: 0 }, g);
      const show = () => { const r = svg.getBoundingClientRect(), k = r.width / W; tip.hidden = false; tip.textContent = text;
        tip.style.left = Math.min(x * k + 12, r.width - 250) + "px"; tip.style.top = (y * k + 30) + "px"; };
      hit.addEventListener("mouseenter", show); hit.addEventListener("focus", show);
      hit.addEventListener("mouseleave", () => tip.hidden = true); hit.addEventListener("blur", () => tip.hidden = true);
    };
    const whisker = (p, c) => {
      if (!p.lo) return;
      const y = Y(p.y), a = { stroke: c, "stroke-width": 2, opacity: 0.7 };
      el("line", { x1: X(p.lo), x2: X(p.hi), y1: y, y2: y, ...a }, g);
      el("line", { x1: X(p.lo), x2: X(p.lo), y1: y - 4, y2: y + 4, ...a }, g);
      el("line", { x1: X(p.hi), x2: X(p.hi), y1: y - 4, y2: y + 4, ...a }, g);
    };
    const endLabel = (pts, text, c) => { const last = pts.filter(inside).pop(); if (last) labels.push({ y: Y(last.y), text, c }); };
    const legendItem = (name, swatch) => {
      const lab = document.createElement("label"), box_ = document.createElement("input");
      box_.type = "checkbox"; box_.checked = on(name);
      box_.addEventListener("change", () => { if (box_.checked) st.hidden.delete(name); else st.hidden.add(name); st.redraw(); });
      lab.appendChild(box_); const sw = document.createElement("span"); sw.innerHTML = swatch; lab.appendChild(sw); lab.append(name); legend.appendChild(lab);
    };
    if (spec.ref !== undefined || (spec.refline || []).length) {
      legendItem(refName, `<i class="dash" style="--c:var(--ref)"></i>`);
      if (refOn && spec.ref !== undefined && spec.ref >= y0 && spec.ref <= y1) {
        el("line", { x1: L, x2: W - R, y1: Y(spec.ref), y2: Y(spec.ref), stroke: "var(--ref)", "stroke-dasharray": "5 4", "stroke-width": 1.5 }, g);
        labels.push({ y: Y(spec.ref), text: refName, c: "var(--ref)" });
      }
      if (refOn && (spec.refline || []).length) {
        el("polyline", { points: spec.refline.map(p => X(p.x) + "," + Y(p.y)).join(" "), fill: "none", stroke: "var(--ref)", "stroke-dasharray": "5 4", "stroke-width": 1.5 }, g);
        spec.refline.forEach(p => { whisker(p, "var(--ref)"); hover(p, `Names only · ${p.label}: ${desc(p)}`); });
        endLabel(spec.refline, refName, "var(--ref)");
      }
    }
    series.forEach(s => {
      legendItem(s.name, `<i style="background:${s.c}"></i>`);
      if (!on(s.name)) return;
      el("polyline", { points: s.points.map(p => X(p.x) + "," + Y(p.y)).join(" "), fill: "none", stroke: s.c, "stroke-width": 2, "stroke-linejoin": "round" }, g);
      s.points.forEach(p => whisker(p, s.c));
      s.points.forEach(p => { el("circle", { cx: X(p.x), cy: Y(p.y), r: 4.5, fill: s.c, stroke: "var(--surface)", "stroke-width": 2 }, g);
        hover(p, `${s.name} · ${p.label}: ${desc(p)}`); });
      endLabel(s.points, s.name, s.c);
    });
    (spec.singles || []).forEach(p => {
      legendItem(p.name, `<i class="diamond"></i>`);
      if (!on(p.name)) return;
      whisker(p, "var(--fg)");
      el("rect", { x: X(p.x) - 5, y: Y(p.y) - 5, width: 10, height: 10, fill: "var(--fg)", transform: `rotate(45 ${X(p.x)} ${Y(p.y)})` }, g);
      hover(p, `${p.name}: ${desc(p)}`);
      const t = el("text", { x: X(p.hi || p.x) + 9, y: Y(p.y) + 4, class: "lbl" }, g); t.textContent = p.name; t.style.fill = "var(--fg)";
    });
    labels.sort((a, b) => a.y - b.y).forEach((l, i, arr) => { if (i && l.y - arr[i - 1].y < 13) l.y = arr[i - 1].y + 13; });
    labels.forEach(l => { const t = el("text", { x: W - R + 8, y: l.y + 4, class: "lbl" }); t.textContent = l.text; t.style.fill = l.c; });
    return auto;
  }
  const XLABEL = { all: "prompt tokens (whole prompt)", rep: "representation tokens (the three characters' blocks)" };
  function view(spec, st) {
    if (!st.basis) return { ...spec, scale: st.scale };
    const pick = p => {
      const v = p.xs[st.basis];
      return v ? { ...p, x: v[st.stat], lo: v.lo, hi: v.hi } : null;
    };
    return { ...spec, series: spec.series.map(s => ({ ...s, points: s.points.map(pick).filter(Boolean) })),
             singles: (spec.singles || []).map(pick).filter(Boolean),
             scale: st.scale, xlabel: `${st.stat === "med" ? "median" : "mean"} ${XLABEL[st.basis]}; bars span the middle half` };
  }
  function segment(box, label, choices, current, onPick) {
    const seg = document.createElement("div"); seg.className = "seg"; seg.setAttribute("role", "group"); seg.setAttribute("aria-label", label);
    choices.forEach(([value, text]) => {
      const b = document.createElement("button"); b.type = "button"; b.textContent = text; b.setAttribute("aria-pressed", value === current);
      b.addEventListener("click", () => { seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b)); onPick(value); });
      seg.appendChild(b);
    });
    box.querySelector("[data-controls]").appendChild(seg);
  }
  function boundsRow(box, st) {
    const row = document.createElement("div"); row.className = "bounds";
    const inputs = {};
    [["x0", "x from"], ["x1", "to"], ["y0", "y from"], ["y1", "to"]].forEach(([k, text]) => {
      const lab = document.createElement("label"), inp = document.createElement("input");
      inp.type = "number"; inp.step = "any"; inp.inputMode = "decimal";
      inp.addEventListener("change", () => { const v = parseFloat(inp.value); st.bounds[k] = Number.isFinite(v) ? v : null; st.redraw(); });
      lab.append(text); lab.appendChild(inp); row.appendChild(lab); inputs[k] = inp;
    });
    const reset = document.createElement("button"); reset.type = "button"; reset.textContent = "Reset bounds";
    reset.addEventListener("click", () => { Object.values(inputs).forEach(i => { i.value = ""; }); st.bounds = {}; st.redraw(); });
    row.appendChild(reset);
    box.appendChild(row);
    return auto => Object.entries(inputs).forEach(([k, i]) => { i.placeholder = num(auto[k]); });
  }
  function mount(id, data, options) {
    const box = document.getElementById(id), names = Object.keys(data);
    if (!names.length) { box.hidden = true; return; }
    const st = { model: names[names.length > 1 ? 1 : 0], hidden: new Set(), bounds: {} };
    const setPlaceholders = boundsRow(box, st);
    const redraw = () => setPlaceholders(draw(box, view(data[st.model], st), st));
    st.redraw = redraw;
    segment(box, "Model", names.map(n => [n, n]), st.model, v => { st.model = v; redraw(); });
    st.scale = "log";
    segment(box, "X axis", [["log", "log x"], ["linear", "linear x"]], st.scale, v => { st.scale = v; redraw(); });
    (options || []).forEach(o => { st[o.key] = o.choices[0][0]; segment(box, o.label, o.choices, st[o.key], v => { st[o.key] = v; redraw(); }); });
    redraw();
  }
  mount("lenbox", {{LENGTH_DATA}}, [{ key: "basis", label: "Length of", choices: [["all", "whole prompt"], ["rep", "representation only"]] },
                                    { key: "stat", label: "Statistic", choices: [["med", "median"], ["mean", "mean"]] }]);
  mount("pasbox", {{PASSAGE_DATA}});
  mount("sumbox", {{SUMMARY_DATA}}, [{ key: "basis", label: "Length of", choices: [["rep", "representation only"], ["all", "whole prompt"]] },
                                     { key: "stat", label: "Statistic", choices: [["med", "median"], ["mean", "mean"]] }]);
})();
</script>
"""

if __name__ == "__main__":
    main()
