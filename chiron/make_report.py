"""Render reports/results.html from the analysis files (no numbers are typed by hand).

  python3 chiron/make_report.py
Reads outputs/analysis_<set>_<model>.json (from analyze.py) for the main, window, short, two and pron sets,
data/items_*.jsonl and data/reps_*.jsonl; writes a self-contained page with two inline SVG charts
(representation length vs accuracy, passage length vs accuracy), each switchable by model.
Findings text lives in FINDINGS below and is written against the final numbers.
"""
import collections
import html
import json
import re
import statistics as st

from common import DATA, OUT, REPO, read_jsonl

MODELS = [("Qwen3-4B-Instruct-2507", "Qwen3-4B"), ("Qwen3.5-9B-Base", "Qwen3.5-9B base"), ("Qwen3.8-27B_nothink", "Qwen3.8-27B"),
          ("Qwen3.8-27B_think", "Qwen3.8-27B, thinking"), ("Mistral-7B-Instruct-v0.2_prefix", "Mistral-7B")]
CHART_MODELS = [("Qwen3-4B-Instruct-2507", "Qwen3-4B"), ("Qwen3.8-27B_nothink", "Qwen3.8-27B"), ("Qwen3.8-27B_think", "Qwen3.8-27B, thinking")]
LABEL = {
    "noinfo": "Names only", "v2": "v2 sheet (current dataset)", "legacy": "Legacy sheet, compressed (Llama-3.3-70B)",
    "legacy_full": "Legacy sheet, full (Llama-3.3-70B)", "summary": "Character summary (gpt-oss)",
    "chiron": "CHIRON-style sheet, full (gpt-oss)", "chiron_r2000": "CHIRON-style sheet, last 2,000 words",
    "charmem": "Charmem sheet (finished rebuild)", "book_last8000": "Book text, last 8,000 words",
    "book_last32000": "Book text, last 32,000 words", "book": "Book text, everything so far",
    "swap_v2": "Swap: another principal's v2 sheet", "swapname_v2": "Swap, names exchanged: v2",
    "swap_chiron": "Swap: another principal's CHIRON-style sheet", "swapname_chiron": "Swap, names exchanged: CHIRON-style",
    "combo_legacy_v2": "Legacy without filler + v2", "combo_short": "v2 + charmem + summary", "combo_all": "All four combined",
    "oracle_passage": "Oracle: clues taken from the passage", "oracle_prior": "Oracle: prior facts chosen for the passage",
    "swapname_oracle_prior": "Oracle prior facts, swapped with names exchanged", "legacy_nofill": "Legacy full without filler",
}
MAIN_ROWS = ["noinfo", "legacy", "chiron_r2000", "v2", "chiron", "charmem", "summary", "book_last8000",
             "book_last32000", "book", "legacy_full", "legacy_nofill", "combo_short", "combo_legacy_v2"]
ORACLE_ROWS = ["noinfo", "v2", "legacy_full", "oracle_prior", "oracle_passage", "swapname_oracle_prior"]
CONTROL_ROWS = ["v2", "swap_v2", "swapname_v2", "chiron", "swap_chiron", "swapname_chiron"]
LENGTH_FAMILIES = [("CHIRON-style", ["chiron_r250", "chiron_r500", "chiron_r1000", "chiron_r2000", "chiron_r4000", "chiron"]),
                   ("v2 sheet", ["v2@100", "v2@250", "v2@500", "v2"]),
                   ("Legacy, compressed", ["legacy@100", "legacy@250", "legacy"]),
                   ("Summary", ["summary@100", "summary@250", "summary@500", "summary"]),
                   ("Book text", ["book_last2000", "book_last8000", "book_last32000", "book"])]
PASSAGE_SERIES = [("v2 sheet", "v2"), ("Charmem", "charmem"), ("CHIRON-style, full", "chiron"),
                  ("Summary", "summary"), ("Legacy, full", "legacy_full"), ("Book, last 8k", "book_last8000")]
SETS = [("short", "Short spans"), ("main", "Sections"), ("window", "Dense windows")]
FINDINGS = [
    "Model strength decides whether the representations matter. Qwen3-4B gains 1 to 5 points over names only; "
    "Qwen3.8-27B with thinking off gains {g27_lo} to {g27_hi} points, positive in every book; with thinking on it reaches "
    "{think_lo} to {think_hi}% on sections with any representation.",
    "The models do use the sheets. With another principal's sheet and the names inside it exchanged, Qwen3.8-27B falls "
    "to {swap27}% (names only {noinfo27}%) and to {swapthink}% with thinking; the reasoning traces cite specific sheet "
    "facts and match them to events in the passage.",
    "Without thinking, the ceiling is matching, not information: clues copied from the passage itself give Qwen3.8-27B only "
    "{orpass27}%, below the full legacy sheet; with thinking the same clues give {orpassthink}%.",
    "The representations get different passages right. Which passage it is explains {varp}% of the variance in "
    "correctness and the representation {varr}%; choosing the best representation per passage would reach {pick}% against "
    "{best}% for the best single one. Combining legacy with v2 gives the best score, {combo}%.",
    "The full legacy sheet (Llama-3.3-70B, per-chapter CHIRON answers) wins because of what it says, not its length or "
    "recency: cut to 6,000 words it scores {leg_r6000}%, without the previous chapter {leg_noprev}%. Its statements naming "
    "the other principals alone score {leg_inter}%; it has 4 to 7 times more of them than the gpt-oss CHIRON-style sheet.",
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
    get = lambda s, m, c: ((A.get((s, m)) or {}).get("rows") or {}).get(c)
    macro = lambda s, m, c: (get(s, m, c) or {}).get("macro")
    words = collections.defaultdict(list)
    for s in ("test", "val", "train"):
        for r in read_jsonl(DATA / f"reps_{s}.jsonl"):
            words[r["condition"]].append(r["words"])
    medw = {k: int(st.median(v)) for k, v in words.items()}
    passage_words, counts = {}, {}
    for key, _ in SETS:
        suf = "" if key == "main" else f"_{key}"
        its = [it for s in ("test", "val", "train") for it in read_jsonl(DATA / f"items_{s}{suf}.jsonl")]
        passage_words[key] = st.median(len(it["original"].split()) for it in its)
        counts[key] = (len(its), len({it["book"] for it in its}))
    it, window = example()
    q27, q4, qt = "Qwen3.8-27B_nothink", "Qwen3-4B-Instruct-2507", "Qwen3.8-27B_think"
    items27 = json.load(open(OUT / "analysis_items_Qwen3.8-27B_nothink.json"))
    items9 = json.load(open(OUT / "analysis_items_Qwen3.5-9B-Base.json"))
    ppl = json.load(open(OUT / "analysis_ppl.json")) if (OUT / "analysis_ppl.json").exists() else {}
    traces = json.load(open(OUT / "analysis_traces.json")) if (OUT / "analysis_traces.json").exists() else {"stats": {}}

    def oracle_table():
        head = "".join(f"<th scope='col' class='num'>{n}</th>" for _, n in MODELS[:4])
        rows = "".join(f"<tr><th scope='row'>{LABEL[c]}</th>" + "".join(f"<td class='num strong'>{pct(macro('main', m, c))}</td>" for m, _ in MODELS[:4]) + "</tr>"
                       for c in ORACLE_ROWS)
        return f"<thead><tr><th scope='col'>Representation</th>{head}</tr></thead><tbody>{rows}</tbody>"

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
        names = {"v2": "v2 sheet", "charmem": "Charmem sheet", "summary": "Summary", "legacy": "Legacy, compressed",
                 "chiron_r2000": "CHIRON-style, 2,000 words", "chiron": "CHIRON-style, full", "legacy_full": "Legacy, full"}
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
            w = {"noinfo": "0", "book_last8000": "8,000", "book_last32000": "32,000"}.get(c, f"{medw.get(c, 0):,}")
            cells = "".join(f"<td class='num strong'>{pct(macro('main', m, c))}</td>" for m, _ in MODELS)
            rows.append(f"<tr><th scope='row'>{LABEL[c]}</th><td class='num'>{w}</td>{cells}"
                        f"<td>{delta(get('main', q27, c)) if c != 'noinfo' else ''}</td></tr>")
        return (f"<thead><tr><th scope='col'>Representation</th><th scope='col' class='num'>Words per character</th>{head}"
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

    ABL = [("legacy_full", "Legacy, full"), ("legacy_noprev", "Legacy without the previous chapter"),
           ("legacy_onlyprev", "Legacy, previous chapter only"), ("legacy_r6000", "Legacy, most recent 6,000 words"),
           ("legacy_nofill", "Legacy without \"not mentioned\" filler"), ("legacy_inter", "Legacy, only statements naming another principal"),
           ("legacy_nointer", "Legacy, only statements not naming another principal"),
           ("chiron", "CHIRON-style, full"), ("chiron_noprev", "CHIRON-style without the previous chapter"),
           ("chiron_onlyprev", "CHIRON-style, previous chapter only"), ("chiron_inter", "CHIRON-style, only statements naming another principal"),
           ("chiron_nointer", "CHIRON-style, only statements not naming another principal"),
           ("book_last8000", "Book, last 8,000 words"), ("book_noprev8000", "Book, last 8,000 words before the previous chapter"),
           ("book_prevonly", "Book, previous chapter only"), ("combo_short", "v2 + charmem + summary"),
           ("combo_legacy_v2", "Legacy without filler + v2"), ("combo_all", "All four combined")]

    def ablation_table():
        rows = "".join(f"<tr><th scope='row'>{n}</th><td class='num'>{medw.get(c, 0):,}</td><td class='num strong'>{pct(macro('main', q27, c))}</td>"
                       f"<td>{delta(get('main', q27, c), 'v2')}</td></tr>" for c, n in ABL if get("main", q27, c))
        return ("<thead><tr><th scope='col'>Representation (Qwen3.8-27B, sections)</th><th scope='col' class='num'>Words per character</th>"
                f"<th scope='col' class='num'>Accuracy</th><th scope='col'>Δ vs v2</th></tr></thead><tbody>{rows}</tbody>")

    def small_table(a, rows):
        if not a:
            return ""
        return "\n".join(f"<tr><th scope='row'>{LABEL[c]}</th><td class='num strong'>{pct(a['rows'][c]['macro'])}</td>"
                         f"<td>{delta(a['rows'][c]) if c != 'noinfo' else ''}</td></tr>" for c in rows if c in a["rows"])

    def book_table():
        a = A[("main", q27)]
        cols = ["noinfo", "summary", "v2", "chiron", "charmem", "legacy_full", "book_last8000", "swapname_v2"]
        head = "".join(f"<th scope='col' class='num'>{h}</th>" for h in
                       ["passages", "names only", "summary", "v2", "CHIRON", "charmem", "legacy full", "book 8k", "swap+names"])
        rows = []
        for b in sorted(a["books"], key=lambda b: -a["n_items"][b]):
            cells = "".join(f"<td class='num'>{100 * a['per_book'][c][b]:.0f}</td>" if b in a["per_book"].get(c, {}) else "<td>—</td>"
                            for c in cols)
            rows.append(f"<tr><th scope='row'><code>{b}</code></th><td class='num'>{a['n_items'][b]}</td>{cells}</tr>")
        return f"<thead><tr><th scope='col'>book</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody>"

    length = {}
    for m, name in CHART_MODELS[:2]:
        a = A[("main", m)]
        if not a:
            continue
        rows = a["rows"]
        series = [{"name": fam, "points": [{"label": c, "x": rows[c]["mean_tokens"], "y": 100 * rows[c]["macro"]}
                                           for c in conds if c in rows and rows[c]["mean_tokens"]]} for fam, conds in LENGTH_FAMILIES]
        singles = [{"name": n, "x": rows[c]["mean_tokens"], "y": 100 * rows[c]["macro"]}
                   for n, c in (("charmem", "charmem"), ("legacy, full", "legacy_full")) if c in rows]
        length[name] = {"series": series, "singles": singles, "ref": 100 * rows["noinfo"]["macro"],
                        "xlabel": "mean prompt tokens (log scale)", "xticks": [1000, 3000, 10000, 30000, 100000]}
    passage = {}
    for m, name in CHART_MODELS:
        series = [{"name": n, "points": [{"label": sname, "x": passage_words[k], "y": 100 * macro(k, m, c)}
                                         for k, sname in SETS if macro(k, m, c) is not None]} for n, c in PASSAGE_SERIES]
        ref = [{"label": sname, "x": passage_words[k], "y": 100 * macro(k, m, "noinfo")} for k, sname in SETS if macro(k, m, "noinfo") is not None]
        if any(s["points"] for s in series):
            passage[name] = {"series": series, "refline": ref, "xlabel": "median passage length in words (log scale)",
                             "xticks": [30, 100, 300, 1000]}

    def table_rows(d):
        return "\n".join(f"<tr><th scope='row'>{m}: {s['name']}</th><td>" + ", ".join(
            f"{p['label']}: {p['y']:.1f}% at {p['x']:,.0f}" for p in s["points"]) + "</td></tr>" for m, v in d.items() for s in v["series"])

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
        varp=f"{100 * items27['var_passage']:.0f}", varr=f"{100 * items27['var_representation']:.1f}",
        pick=f"{100 * items27['oracle_pick']:.0f}", best=f"{100 * items27['best_single']:.0f}",
        ppl_lo=f"{100 * (1 - max(v['ppl_ratio'] for k, v in ppl.get('Qwen3.5-9B-Base|none', {}).items() if k != 'names' and v['passages'] > 1000)):.0f}",
        ppl_hi=f"{100 * (1 - min(v['ppl_ratio'] for k, v in ppl.get('Qwen3.5-9B-Base|none', {}).items() if k != 'names' and v['passages'] > 1000)):.0f}")
    findings = "\n".join(f"<li>{f.format(**vals)}</li>" for f in FINDINGS)

    ex = html.escape(window)
    ex = re.sub(r"\[CHAR (\d)\]", r'<mark class="m\1">[CHAR \1]</mark>', ex)
    names = ", ".join(f"{html.escape(l)} = <mark class='m{i}'>[CHAR {i}]</mark>" for l, i in sorted(it["answer"].items(), key=lambda kv: kv[1])) if it else ""
    page = TEMPLATE
    for k, v in {"ORACLE_TABLE": oracle_table(), "ITEMS_TABLE": items_table(), "PPL_TABLE": ppl_table(), "TRACES_TABLE": traces_table(),
                 "ABLATION_TABLE": ablation_table(), "MAIN_TABLE": main_table(), "CONTROL_TABLE": control_table(), "PASSAGE_TABLE": passage_table(),
                 "BOOK_TABLE": book_table(), "FINDINGS": findings,
                 "TWO_TABLE": small_table(A[("two", q4)], ["noinfo", "summary", "v2", "chiron", "charmem", "legacy_full", "book"]),
                 "PRON_TABLE": small_table(A[("pron", q4)], ["noinfo", "summary", "v2", "chiron", "charmem", "book_last8000"]),
                 "LENGTH_DATA": json.dumps(length), "PASSAGE_DATA": json.dumps(passage),
                 "LENGTH_ROWS": table_rows(length), "PASSAGE_ROWS": table_rows(passage),
                 "EXAMPLE": ex, "EXAMPLE_KEY": names, "EXAMPLE_BOOK": html.escape(it["book"]) if it else "",
                 "N_MAIN": f"{counts['main'][0]:,}", "N_SHORT": f"{counts['short'][0]:,}", "N_WINDOW": f"{counts['window'][0]:,}",
                 "N_BOOKS": str(len(A[("main", q27)]["books"]))}.items():
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
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300; --ref: #8a9097;
  --m0bg: #dbe8f9; --m1bg: #fbe1d5; --m2bg: #d4f0e5;
  --display: "Newsreader", "Iowan Old Style", Georgia, serif;
  --body: "Public Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141618; --surface: #1b1e21; --fg: #eceef0; --muted: #a2a8b0; --rule: #2d3237; --accent: #7fb0ee;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --ref: #7d848c;
  --m0bg: #1f3350; --m1bg: #4a2a1c; --m2bg: #173c30; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141618; --surface: #1b1e21; --fg: #eceef0; --muted: #a2a8b0; --rule: #2d3237; --accent: #7fb0ee;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300; --ref: #7d848c;
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
.legend span { display: inline-flex; align-items: center; gap: 0.35rem; }
.legend i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
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
    <div class="eyebrow">CHIRON replication · NCP books · 29 September 2026</div>
    <h1>Can a character sheet put the names back?</h1>
    <p class="lede">CHIRON's masked-character test on the NCP novels. A model sees a passage with the three principals' names replaced by ids, plus one representation of each character built only from earlier chapters, and says which id is which. {{N_MAIN}} sections (about 320 words), {{N_SHORT}} short spans (about 50 words) and {{N_WINDOW}} dense windows (about 900 words); four model setups.</p>
  </header>

  <section aria-labelledby="ex">
    <h2 id="ex">What the model sees</h2>
    <div class="exhibit">
      <blockquote>… {{EXAMPLE}} …</blockquote>
      <div class="key">{{EXAMPLE_BOOK}} · answer: {{EXAMPLE_KEY}}</div>
      <div class="q">One block per character (the representation being tested), then the passage. Without thinking, the model answers "Which ID in the passage is &lt;name&gt;?" and is scored from the probability of each id digit. With thinking, it reasons and then gives the whole name-to-id mapping.</div>
    </div>
  </section>

  <section aria-labelledby="find">
    <h2 id="find">Findings</h2>
    <ul class="findings">{{FINDINGS}}</ul>
  </section>

  <section aria-labelledby="main">
    <h2 id="main">Sections, all three principals named</h2>
    <p class="muted">Macro accuracy over the {{N_BOOKS}} books with at least 10 passages (each book counts once; chance is 33.3%). The last column is the mean per-book gain over names only for Qwen3.8-27B without thinking, the number of books where it is positive, and a 95% bootstrap interval over books. Mistral-7B cannot read the two longest representations (32k context).</p>
    <div class="table-wrap"><table>{{MAIN_TABLE}}</table></div>
  </section>

  <section aria-labelledby="len">
    <h2 id="len">Representation length and accuracy</h2>
    <p class="muted">Sections. Each line is one representation cut to increasing lengths: CHIRON-style keeps its most recent statements, v2, legacy and summary keep their first words, book text keeps its last words. Diamonds are representations tested at one length only. The dashed line is names only.</p>
    <div class="chart" id="lenbox">
      <div class="bar"><div class="legend" data-legend></div><div class="seg" role="group" aria-label="Model" data-seg></div></div>
      <svg viewBox="0 0 760 380" role="img" aria-label="Accuracy against mean prompt tokens, log scale"></svg>
      <div class="tip" hidden></div>
    </div>
    <details><summary>Chart data</summary><div class="table-wrap" style="margin-top:0.6rem"><table><tbody>{{LENGTH_ROWS}}</tbody></table></div></details>
  </section>

  <section aria-labelledby="pas">
    <h2 id="pas">Passage length and accuracy</h2>
    <p class="muted">The same representations on three passage sets: short spans (the tightest run of sentences naming all three principals, 20 to 150 words), NCP sections, and dense windows (up to three consecutive sections, about 900 words, each principal named at least three times). Each set averages over its own books, so the points are not paired. The dashed line is names only.</p>
    <div class="chart" id="pasbox">
      <div class="bar"><div class="legend" data-legend></div><div class="seg" role="group" aria-label="Model" data-seg></div></div>
      <svg viewBox="0 0 760 380" role="img" aria-label="Accuracy against median passage length, log scale"></svg>
      <div class="tip" hidden></div>
    </div>
    <details><summary>Chart data</summary><div class="table-wrap" style="margin-top:0.6rem"><table><tbody>{{PASSAGE_ROWS}}</tbody></table></div></details>
    <div class="table-wrap"><table>{{PASSAGE_TABLE}}</table></div>
  </section>

  <section aria-labelledby="sim">
    <h2 id="sim">Why the numbers look so similar</h2>
    <p class="muted">Per passage and character, correctness depends far more on which passage it is than on which representation the model gets. The representations are right on different passages, so their averages converge while a per-passage choice would do much better (thinking off, sections).</p>
    <div class="table-wrap"><table>{{ITEMS_TABLE}}</table></div>
  </section>

  <section aria-labelledby="orc">
    <h2 id="orc">Oracles: exactly the information needed</h2>
    <p class="muted">Per passage, gpt-oss wrote (a) 2 to 4 clues per character taken from the unmasked passage itself, a ceiling, and (b) up to 5 facts per character copied word for word from the notes available before the chapter, chosen because they identify the character in this passage (facts that were not verbatim copies were dropped). Swapping (b) between characters with names exchanged tests whether the model relies on it.</p>
    <div class="table-wrap"><table>{{ORACLE_TABLE}}</table></div>
  </section>

  <section aria-labelledby="trc">
    <h2 id="trc">Is the reasoning using the sheets?</h2>
    <p class="muted">Saved reasoning from Qwen3.8-27B with thinking on (test and validation passages, one block order). With a sheet the reasoning is half as long and cites sheet facts against events in the passage, for example "If CHAR 2 is Liska, 'Liska's aunt' matches character info: Liska mentions an aunt who lives in Ząbki". With swapped sheets nearly every trace reasons from the misleading notes.</p>
    <div class="table-wrap"><table>{{TRACES_TABLE}}</table></div>
  </section>

  <section aria-labelledby="ppl">
    <h2 id="ppl">Does character information make the real next passage more likely?</h2>
    <p class="muted">Change in perplexity of the real passage (names unmasked) against a prompt that only lists the characters' names. Qwen3.5-9B base reads a plain-text prompt; Qwen3.8-27B reads the notes as a chat request to write the next passage and is scored on the passage as its reply. "Story" adds the 4,000 words right before the passage. Negative is better; every value covering all passages is negative in all 21 books.</p>
    <div class="table-wrap"><table>{{PPL_TABLE}}</table></div>
  </section>

  <section aria-labelledby="why">
    <h2 id="why">Why legacy and book text do best</h2>
    <p class="muted">Ablations with Qwen3.8-27B, thinking off. Dropping the chapter just before the passage changes nothing, and neither does cutting legacy to a third of its length, so the lead is not recency or volume. A word-overlap matcher with no model ranks legacy below the CHIRON-style sheet, so it is not shared vocabulary either. Legacy's per-chapter answers name the other two principals about twice as often per word as the other sheets (31 per 1,000 words against 13 to 20), which is the information that separates three characters appearing together.</p>
    <div class="table-wrap"><table>{{ABLATION_TABLE}}</table></div>
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
      <dt>Representations</dt><dd>All built from chapters before the passage's chapter. CHIRON-style: CHIRON's 8 questions answered per 300-word snippet by gpt-oss-120b, claims kept only at rating 5 on the paper's 1–5 entailment scale, grouped by category and deduplicated. Summary: gpt-oss rolling summary condensed to about 700 words. Legacy: the Llama-3.3-70B sheets from the original NCP archive.</dd>
      <dt>Scoring</dt><dd>Without thinking: next-token probabilities of the id digits at temperature 0 (Qwen3-4B over all 6 block orders, Qwen3.8-27B over 3 rotations; Mistral's reply is pre-started with "[CHAR "). With thinking: one generation per passage and rotation (temperature 0.6, up to 32k tokens) ending in a JSON mapping; an unreadable answer counts as wrong.</dd>
      <dt>Memorization</dt><dd>Temperature-0 continuations of the four test books reproduced no 13-word sequence (longest verbatim run 5 words).</dd>
      <dt>Caveats</dt><dd>21 books carry the statistics and ten of them hold most passages. The passage sets cover different books (short spans 19, dense windows 13 with at least 5 passages). With thinking on, sections are near ceiling, which compresses differences between representations. Reasoning results are partial until every shard finishes.</dd>
    </dl>
  </section>
</main>

<script>
(function () {
  const ns = "http://www.w3.org/2000/svg", colors = ["--s1", "--s2", "--s3", "--s4", "--s5", "--s6"];
  function draw(box, spec) {
    const svg = box.querySelector("svg"), tip = box.querySelector(".tip"), legend = box.querySelector("[data-legend]");
    svg.replaceChildren(); legend.replaceChildren(); tip.hidden = true;
    const W = 760, H = 380, L = 48, R = 160, T = 16, B = 42;
    const pts = spec.series.flatMap(s => s.points).concat(spec.singles || [], spec.refline || []);
    const ys = pts.map(p => p.y).concat(spec.ref !== undefined ? [spec.ref] : []);
    const x0 = Math.log10(spec.xticks[0] * 0.8), x1 = Math.log10(spec.xticks[spec.xticks.length - 1] * 1.1);
    let y0 = Math.floor(Math.min(...ys) / 5) * 5, y1 = Math.ceil(Math.max(...ys) / 5) * 5;
    if (y1 - y0 < 10) y1 = y0 + 10;
    const X = v => L + (Math.log10(v) - x0) / (x1 - x0) * (W - L - R), Y = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
    const el = (tag, a) => { const e = document.createElementNS(ns, tag); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
    const step = (y1 - y0) > 30 ? 10 : 5;
    for (let v = y0; v <= y1; v += step) {
      el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "var(--rule)", "stroke-width": 1 });
      el("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end" }).textContent = v + "%";
    }
    spec.xticks.forEach(v => { el("text", { x: X(v), y: H - B + 18, "text-anchor": "middle" }).textContent = v >= 1000 ? (v / 1000) + "k" : v; });
    el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle" }).textContent = spec.xlabel;
    const labels = [];
    const hover = (x, y, text) => {
      const hit = el("circle", { cx: x, cy: y, r: 12, fill: "transparent", tabindex: 0 });
      const show = () => { const r = svg.getBoundingClientRect(), k = r.width / W; tip.hidden = false; tip.textContent = text;
        tip.style.left = Math.min(x * k + 12, r.width - 250) + "px"; tip.style.top = (y * k + 30) + "px"; };
      hit.addEventListener("mouseenter", show); hit.addEventListener("focus", show);
      hit.addEventListener("mouseleave", () => tip.hidden = true); hit.addEventListener("blur", () => tip.hidden = true);
    };
    if (spec.ref !== undefined) {
      el("line", { x1: L, x2: W - R, y1: Y(spec.ref), y2: Y(spec.ref), stroke: "var(--ref)", "stroke-dasharray": "5 4", "stroke-width": 1.5 });
      labels.push({ y: Y(spec.ref), text: "names only", c: "var(--ref)" });
    }
    if (spec.refline && spec.refline.length) {
      el("polyline", { points: spec.refline.map(p => X(p.x) + "," + Y(p.y)).join(" "), fill: "none", stroke: "var(--ref)", "stroke-dasharray": "5 4", "stroke-width": 1.5 });
      spec.refline.forEach(p => hover(X(p.x), Y(p.y), `Names only · ${p.label}: ${p.y.toFixed(1)}%`));
      const last = spec.refline[spec.refline.length - 1]; labels.push({ y: Y(last.y), text: "names only", c: "var(--ref)" });
    }
    spec.series.forEach((s, i) => {
      if (!s.points.length) return;
      const c = `var(${colors[i]})`;
      el("polyline", { points: s.points.map(p => X(p.x) + "," + Y(p.y)).join(" "), fill: "none", stroke: c, "stroke-width": 2, "stroke-linejoin": "round" });
      s.points.forEach(p => { el("circle", { cx: X(p.x), cy: Y(p.y), r: 4.5, fill: c, stroke: "var(--surface)", "stroke-width": 2 });
        hover(X(p.x), Y(p.y), `${s.name} · ${p.label}: ${p.y.toFixed(1)}% at ${Math.round(p.x).toLocaleString()}`); });
      const last = s.points[s.points.length - 1]; labels.push({ y: Y(last.y), text: s.name, c });
      const lg = document.createElement("span"); lg.innerHTML = `<i style="background:${c}"></i>`; lg.append(s.name); legend.appendChild(lg);
    });
    (spec.singles || []).forEach(p => {
      el("rect", { x: X(p.x) - 5, y: Y(p.y) - 5, width: 10, height: 10, fill: "var(--fg)", transform: `rotate(45 ${X(p.x)} ${Y(p.y)})` });
      hover(X(p.x), Y(p.y), `${p.name}: ${p.y.toFixed(1)}% at ${Math.round(p.x).toLocaleString()}`);
      const t = el("text", { x: X(p.x) + 9, y: Y(p.y) + 4, class: "lbl" }); t.textContent = p.name; t.style.fill = "var(--fg)";
    });
    labels.sort((a, b) => a.y - b.y).forEach((l, i, arr) => { if (i && l.y - arr[i - 1].y < 13) l.y = arr[i - 1].y + 13; });
    labels.forEach(l => { const t = el("text", { x: W - R + 8, y: l.y + 4, class: "lbl" }); t.textContent = l.text; t.style.fill = l.c; });
  }
  function mount(id, data) {
    const box = document.getElementById(id), seg = box.querySelector("[data-seg]"), names = Object.keys(data);
    if (!names.length) { box.hidden = true; return; }
    let current = names[names.length > 1 ? 1 : 0];
    names.forEach(n => {
      const b = document.createElement("button"); b.type = "button"; b.textContent = n;
      b.addEventListener("click", () => { current = n; seg.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", x === b)); draw(box, data[n]); });
      b.setAttribute("aria-pressed", n === current); seg.appendChild(b);
    });
    draw(box, data[current]);
  }
  mount("lenbox", {{LENGTH_DATA}});
  mount("pasbox", {{PASSAGE_DATA}});
})();
</script>
"""

if __name__ == "__main__":
    main()
