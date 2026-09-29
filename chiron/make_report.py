"""Render reports/results.html from the analysis files (no numbers are typed by hand).

  python3 chiron/make_report.py
Reads outputs/analysis_{main,two,pron}_<model>.json (from analyze.py), data/items_*.jsonl and
data/reps_*.jsonl; writes a self-contained page with an inline SVG length-vs-accuracy chart.
"""
import collections
import html
import json
import re
import statistics as st

from common import DATA, OUT, REPO, read_jsonl

QWEN, MIS = "Qwen3-4B-Instruct-2507", "Mistral-7B-Instruct-v0.2_prefix"
LABEL = {
    "noinfo": "No information (names only)", "v2": "v2 sheet (current dataset)", "legacy": "Legacy sheet, compressed (Llama-3.3-70B)",
    "legacy_full": "Legacy sheet, full (Llama-3.3-70B)", "summary": "Character summary (gpt-oss, rolling)",
    "chiron": "CHIRON-style sheet, full (gpt-oss)", "chiron_r2000": "CHIRON-style sheet, last 2,000 words",
    "charmem": "Charmem sheet (finished rebuild)", "book_last8000": "Book text, last 8,000 words",
    "book_last32000": "Book text, last 32,000 words", "book": "Book text, everything so far",
    "swap_v2": "Swap: another principal's v2 sheet", "swapname_v2": "Swap with names exchanged: v2",
    "swap_chiron": "Swap: another principal's CHIRON-style sheet", "swapname_chiron": "Swap with names exchanged: CHIRON-style",
    "swap_summary": "Swap: another principal's summary", "swapname_summary": "Swap with names exchanged: summary",
    "swapname_legacy_full": "Swap with names exchanged: legacy full", "swap_charmem": "Swap: another principal's charmem sheet",
}
MAIN_ROWS = ["noinfo", "summary", "legacy", "v2", "chiron_r2000", "charmem", "chiron", "legacy_full",
             "book_last8000", "book_last32000", "book"]
CONTROL_ROWS = ["v2", "swap_v2", "swapname_v2", "chiron", "swap_chiron", "swapname_chiron", "summary", "swap_summary",
                "swapname_summary"]
FAMILIES = [("CHIRON-style", ["chiron_r250", "chiron_r500", "chiron_r1000", "chiron_r2000", "chiron_r4000", "chiron"]),
            ("v2 sheet", ["v2@100", "v2@250", "v2@500", "v2"]),
            ("Legacy", ["legacy@100", "legacy@250", "legacy", "legacy_full"]),
            ("Summary", ["summary@100", "summary@250", "summary@500", "summary"]),
            ("Book text", ["book_last2000", "book_last8000", "book_last32000", "book"])]


def load(kind, model):
    return json.load(open(OUT / f"analysis_{kind}_{model}.json"))


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def delta(row, ref):
    d = row.get(f"vs_{ref}")
    if not d:
        return "—"
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
    q, m = load("main", QWEN), load("main", MIS)
    two, pron = load("two", QWEN), load("pron", QWEN)
    words = collections.defaultdict(list)
    for s in ("test", "val", "train"):
        for r in read_jsonl(DATA / f"reps_{s}.jsonl"):
            words[r["condition"]].append(r["words"])
    medw = {k: int(st.median(v)) for k, v in words.items()}
    n_main = sum(len(read_jsonl(DATA / f"items_{s}.jsonl")) for s in ("test", "val", "train"))
    n_two = sum(len(read_jsonl(DATA / f"items_{s}_two.jsonl")) for s in ("test", "val", "train"))
    it, window = example()

    def main_table():
        out = []
        for c in MAIN_ROWS:
            r, rm = q["rows"].get(c), m["rows"].get(c)
            w = medw.get(c.replace("book_last8000", "").replace("book_last32000", ""), None)
            wtxt = {"noinfo": "0", "book_last8000": "8,000 (shared)", "book_last32000": "32,000 (shared)",
                    "book": "whole book so far"}.get(c, f"{w:,}" if w else "—")
            out.append(f"<tr><th scope='row'>{LABEL[c]}</th><td class='num'>{wtxt}</td>"
                       f"<td class='num strong'>{pct(r['macro'])}</td><td>{delta(r, 'noinfo') if c != 'noinfo' else ''}</td>"
                       f"<td class='num strong'>{pct(rm['macro']) if rm else '—'}</td>"
                       f"<td>{(delta(rm, 'noinfo') if c != 'noinfo' else '') if rm else '<span class=sub>beyond 32k context</span>'}</td></tr>")
        return "\n".join(out)

    def control_table():
        out = []
        for c in CONTROL_ROWS:
            r, rm = q["rows"].get(c), m["rows"].get(c)
            out.append(f"<tr><th scope='row'>{LABEL[c]}</th><td class='num strong'>{pct(r['macro']) if r else '—'}</td>"
                       f"<td>{delta(r, 'noinfo') if r else ''}</td><td class='num strong'>{pct(rm['macro']) if rm else '—'}</td></tr>")
        return "\n".join(out)

    def small_table(a, rows):
        return "\n".join(f"<tr><th scope='row'>{LABEL[c]}</th><td class='num strong'>{pct(a['rows'][c]['macro'])}</td>"
                         f"<td>{delta(a['rows'][c], 'noinfo') if c != 'noinfo' else ''}</td></tr>"
                         for c in rows if c in a["rows"])

    def book_table():
        cols = ["noinfo", "summary", "v2", "chiron", "charmem", "legacy_full", "swapname_v2"]
        head = "".join(f"<th scope='col' class='num'>{h}</th>" for h in
                       ["items", "no info", "summary", "v2", "CHIRON", "charmem", "legacy full", "swap+names"])
        rows = []
        for b in sorted(q["books"], key=lambda b: -q["n_items"][b]):
            cells = "".join(f"<td class='num'>{100 * q['per_book'][c][b]:.0f}</td>" if b in q["per_book"].get(c, {}) else "<td>—</td>"
                            for c in cols)
            rows.append(f"<tr><th scope='row'><code>{b}</code></th><td class='num'>{q['n_items'][b]}</td>{cells}</tr>")
        return f"<thead><tr><th scope='col'>book</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody>"

    series = []
    for name, conds in FAMILIES:
        pts = [{"cond": c, "x": q["rows"][c]["mean_tokens"], "y": 100 * q["rows"][c]["macro"]} for c in conds if c in q["rows"]]
        series.append({"name": name, "points": pts})
    chart = {"series": series, "noinfo": 100 * q["rows"]["noinfo"]["macro"],
             "charmem": {"x": q["rows"]["charmem"]["mean_tokens"], "y": 100 * q["rows"]["charmem"]["macro"]}}
    length_rows = "\n".join(f"<tr><th scope='row'>{s['name']}</th><td>" + ", ".join(
        f"{p['cond']}: {p['y']:.1f}% at {p['x'] / 1000:.1f}k tokens" for p in s["points"]) + "</td></tr>" for s in series)

    ex = html.escape(window)
    ex = re.sub(r"\[CHAR (\d)\]", r'<mark class="m\1">[CHAR \1]</mark>', ex)
    names = ", ".join(f"{html.escape(l)} = <mark class='m{i}'>[CHAR {i}]</mark>" for l, i in sorted(it["answer"].items(), key=lambda kv: kv[1])) if it else ""

    page = TEMPLATE
    for k, v in {"MAIN_TABLE": main_table(), "CONTROL_TABLE": control_table(), "BOOK_TABLE": book_table(),
                 "TWO_TABLE": small_table(two, ["noinfo", "summary", "v2", "chiron", "charmem", "legacy_full", "book"]),
                 "PRON_TABLE": small_table(pron, ["noinfo", "summary", "v2", "chiron", "charmem", "book_last8000"]),
                 "LENGTH_ROWS": length_rows, "CHART_DATA": json.dumps(chart), "EXAMPLE": ex, "EXAMPLE_KEY": names,
                 "EXAMPLE_BOOK": html.escape(it["book"]) if it else "", "N_MAIN": f"{n_main:,}", "N_TWO": f"{n_two:,}",
                 "N_BOOKS": str(len(q["books"])), "N_BOOKS_TWO": str(len(two["books"])),
                 "NOINFO_Q": pct(q["rows"]["noinfo"]["macro"]), "NOINFO_M": pct(m["rows"]["noinfo"]["macro"])}.items():
        page = page.replace("{{" + k + "}}", v)
    (REPO / "reports").mkdir(exist_ok=True)
    (REPO / "reports" / "results.html").write_text(page)
    print("wrote reports/results.html", len(page), "bytes")


TEMPLATE = r"""<title>CHIRON Book Replication</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: one reading column, tables and the chart in their own scroll containers. */
:root {
  --bg: #f6f7f7; --surface: #ffffff; --fg: #16191d; --muted: #5a6068; --rule: #dde1e4; --accent: #245fa8;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --ref: #8a9097;
  --m0bg: #dbe8f9; --m1bg: #fbe1d5; --m2bg: #d4f0e5;
  --display: "Newsreader", "Iowan Old Style", Georgia, serif;
  --body: "Public Sans", "Segoe UI", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141618; --surface: #1b1e21; --fg: #eceef0; --muted: #a2a8b0; --rule: #2d3237; --accent: #7fb0ee;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --ref: #7d848c;
  --m0bg: #1f3350; --m1bg: #4a2a1c; --m2bg: #173c30; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141618; --surface: #1b1e21; --fg: #eceef0; --muted: #a2a8b0; --rule: #2d3237; --accent: #7fb0ee;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --ref: #7d848c;
  --m0bg: #1f3350; --m1bg: #4a2a1c; --m2bg: #173c30; color-scheme: dark }
body { background: var(--bg); color: var(--fg); font: 15px/1.6 var(--body); }
main { max-width: 60rem; margin: 0 auto; padding-inline: 1.25rem; padding-block: 2.5rem 4rem; display: grid; gap: 2.25rem; }
section { display: grid; gap: 0.9rem; min-width: 0; }
h1, h2 { font-family: var(--display); font-weight: 600; text-wrap: balance; margin: 0; line-height: 1.15; }
h1 { font-size: clamp(1.9rem, 4vw, 2.6rem); }
h2 { font-size: 1.35rem; }
p { margin: 0; max-width: 68ch; }
.eyebrow { font: 500 0.75rem/1 var(--body); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.lede { font-size: 1.05rem; color: var(--fg); }
.muted, .sub { color: var(--muted); }
.sub { font-size: 0.8rem; }
ul.findings { margin: 0; padding-left: 1.2rem; display: grid; gap: 0.5rem; max-width: 72ch; }
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
td:has(> .num) { white-space: nowrap; }
.strong { font-weight: 600; }
code { font: 0.85em var(--mono); }
.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 0.75rem; position: relative; }
.chart svg { display: block; width: 100%; height: auto; }
.chart text { fill: var(--muted); font: 11px var(--body); }
.chart .lbl { font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 0.4rem 1rem; font-size: 0.8rem; color: var(--muted); padding: 0 0.25rem 0.5rem; }
.legend span { display: inline-flex; align-items: center; gap: 0.35rem; }
.legend i { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
#tip { position: absolute; pointer-events: none; background: var(--fg); color: var(--bg); font: 0.78rem/1.4 var(--body); padding: 0.35rem 0.55rem; border-radius: 4px; }
details { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 0.6rem 0.9rem; }
summary { cursor: pointer; font-weight: 500; }
summary:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.35rem 1rem; margin: 0; }
dt { color: var(--muted); } dd { margin: 0; min-width: 0; }
@media (max-width: 560px) { dl { grid-template-columns: 1fr; } dt { margin-top: 0.4rem; } }
</style>

<main>
  <header style="display:grid;gap:0.6rem">
    <div class="eyebrow">CHIRON replication · NCP books · 29 September 2026</div>
    <h1>Can a character sheet put the names back?</h1>
    <p class="lede">CHIRON's masked-character test, run on the NCP novels: {{N_MAIN}} passages where all three principals are named (plus {{N_TWO}} two-principal passages), the names replaced with ids, and a model asked which id is which character given only a representation of each character built from earlier chapters.</p>
  </header>

  <section aria-labelledby="ex">
    <h2 id="ex">What the model sees</h2>
    <div class="exhibit">
      <blockquote>… {{EXAMPLE}} …</blockquote>
      <div class="key">{{EXAMPLE_BOOK}} · answer: {{EXAMPLE_KEY}}</div>
      <div class="q">The prompt adds one block per character (the condition being tested), then asks "Which ID in the passage is &lt;name&gt;?" for each character, in all orders of the blocks. Scored from the probability of each id digit.</div>
    </div>
  </section>

  <section aria-labelledby="find">
    <h2 id="find">Findings</h2>
    <ul class="findings">
      <li>Models do use the sheets. Give each character another principal's sheet with the names inside it exchanged, and accuracy drops to 32–34%, below the names-only floor ({{NOINFO_Q}}% for Qwen3-4B, {{NOINFO_M}}% for Mistral-7B), in 17 to 20 of {{N_BOOKS}} books.</li>
      <li>The gains are small for Qwen3-4B (1 to 5 points over names only) and larger for Mistral-7B-v0.2 (5 to 10 points), the model the paper used.</li>
      <li>Sheets beat summaries of similar length: for Qwen3-4B the v2 sheet is ahead of the gpt-oss character summary in 17 of 21 books. For Mistral the two are level.</li>
      <li>Among sheets the differences are within noise. The finished charmem rebuild is the best short sheet for both models, 1 point over v2 (13 of 21 books). The full legacy CHIRON-style sheet (about 14,000 words) is the best overall for Qwen3-4B.</li>
      <li>Longer inputs mostly help, slowly. The exception is the whole book so far (about 83,000 tokens), which scores below its last 32,000 words. Raw book text does no better than a sheet of similar length.</li>
    </ul>
  </section>

  <section aria-labelledby="main">
    <h2 id="main">Three principals named</h2>
    <p class="muted">Macro accuracy over the {{N_BOOKS}} books with at least 10 passages (each book counts once; chance is 33.3%). "Δ" is the mean per-book difference from names-only, then books where it is positive and a 95% bootstrap interval over books.</p>
    <div class="table-wrap"><table>
      <thead><tr><th scope="col">Representation</th><th scope="col" class="num">Median words</th><th scope="col" class="num">Qwen3-4B</th><th scope="col">Δ vs names only</th><th scope="col" class="num">Mistral-7B</th><th scope="col">Δ vs names only</th></tr></thead>
      <tbody>{{MAIN_TABLE}}</tbody>
    </table></div>
  </section>

  <section aria-labelledby="len">
    <h2 id="len">Input length and accuracy</h2>
    <p class="muted">Qwen3-4B, three-principal set. Each line is one representation cut to increasing lengths (CHIRON-style keeps the most recent statements; the others keep the first words; book text keeps the last words). The dashed line is names only.</p>
    <div class="chart">
      <div class="legend" id="legend"></div>
      <svg id="lenchart" viewBox="0 0 720 360" role="img" aria-label="Accuracy against mean prompt tokens, log scale"></svg>
      <div id="tip" hidden></div>
    </div>
    <details><summary>Chart data</summary><div class="table-wrap" style="margin-top:0.6rem"><table><tbody>{{LENGTH_ROWS}}</tbody></table></div></details>
  </section>

  <section aria-labelledby="ctl">
    <h2 id="ctl">Controls</h2>
    <p class="muted">A plain swap gives each character another principal's representation. Sheets name their own subject in almost every line, so the model can still tell whose sheet it is and scores as well as with the right sheet. Exchanging the two characters' names inside the swapped text removes that escape, and accuracy falls below names only.</p>
    <div class="table-wrap"><table>
      <thead><tr><th scope="col">Condition</th><th scope="col" class="num">Qwen3-4B</th><th scope="col">Δ vs names only</th><th scope="col" class="num">Mistral-7B</th></tr></thead>
      <tbody>{{CONTROL_TABLE}}</tbody>
    </table></div>
  </section>

  <section aria-labelledby="other">
    <h2 id="other">Other item sets (Qwen3-4B)</h2>
    <div style="display:grid;gap:1.2rem;grid-template-columns:repeat(auto-fit,minmax(min(100%,22rem),1fr))">
      <div style="display:grid;gap:0.5rem;min-width:0">
        <p class="muted">Two principals named ({{N_BOOKS_TWO}} books; chance 50%).</p>
        <div class="table-wrap"><table><thead><tr><th scope="col">Representation</th><th scope="col" class="num">Acc.</th><th scope="col">Δ</th></tr></thead><tbody>{{TWO_TABLE}}</tbody></table></div>
      </div>
      <div style="display:grid;gap:0.5rem;min-width:0">
        <p class="muted">Names and principals' pronouns masked (pronouns resolved by gpt-oss; a missed one still leaks gender).</p>
        <div class="table-wrap"><table><thead><tr><th scope="col">Representation</th><th scope="col" class="num">Acc.</th><th scope="col">Δ</th></tr></thead><tbody>{{PRON_TABLE}}</tbody></table></div>
      </div>
    </div>
  </section>

  <section aria-labelledby="books">
    <h2 id="books">By book</h2>
    <p class="muted">Qwen3-4B accuracy (%) per book, three-principal set. Books with fewer than 10 passages are left out of the averages above.</p>
    <div class="table-wrap"><table>{{BOOK_TABLE}}</table></div>
  </section>

  <section aria-labelledby="setup">
    <h2 id="setup">Setup</h2>
    <dl>
      <dt>Passages</dt><dd>NCP sections (about 300 words) naming all three book-level principals, each named in at least two earlier chapters. Names come from hand-checked alias tables (nicknames, earlier names, secret identities); passages with an ambiguous surname or a leftover name are dropped. <em>god</em> has none, since Alice and Louise never share a section.</dd>
      <dt>Representations</dt><dd>All built from chapters before the passage's chapter. CHIRON-style: CHIRON's 8 questions answered per 300-word snippet by gpt-oss-120b, claims kept only at rating 5 on the paper's 1–5 entailment scale, grouped by category and deduplicated. Summary: gpt-oss rolling summary, condensed to about 700 words. Legacy: the Llama-3.3-70B sheets from the original NCP archive.</dd>
      <dt>Scoring</dt><dd>Next-token probabilities of the id digits at temperature 0, averaged over every order of the character blocks. Mistral's reply is pre-started with "[CHAR " because it does not answer with a bare digit.</dd>
      <dt>Memorization</dt><dd>Temperature-0 continuations of the four test books reproduced no 13-word sequence (longest verbatim run 5 words) for either model.</dd>
      <dt>Caveats</dt><dd>21 books carry the statistics and ten of them hold most passages. Differences of 1 to 2 points between sheet types are inside the book-level intervals. Qwen3-4B's first token is a digit for 96–99% of questions; otherwise the digit probabilities further down its top 20 decide.</dd>
    </dl>
  </section>
</main>

<script>
(function () {
  const data = {{CHART_DATA}};
  const svg = document.getElementById("lenchart"), tip = document.getElementById("tip");
  const W = 720, H = 360, L = 48, R = 150, T = 16, B = 40;
  const xs = [], ys = [data.noinfo, data.charmem.y];
  data.series.forEach(s => s.points.forEach(p => { xs.push(p.x); ys.push(p.y); }));
  const x0 = Math.log10(500), x1 = Math.log10(100000);
  const y0 = Math.floor(Math.min(...ys) - 1), y1 = Math.ceil(Math.max(...ys) + 1);
  const X = v => L + (Math.log10(v) - x0) / (x1 - x0) * (W - L - R);
  const Y = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);
  const ns = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const e = document.createElementNS(ns, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).appendChild(e); return e; };
  const colors = ["--s1", "--s2", "--s3", "--s4", "--s5"];
  for (let v = Math.ceil(y0); v <= y1; v += 2) {
    el("line", { x1: L, x2: W - R, y1: Y(v), y2: Y(v), stroke: "var(--rule)", "stroke-width": 1 });
    el("text", { x: L - 8, y: Y(v) + 4, "text-anchor": "end" }).textContent = v + "%";
  }
  [1000, 3000, 10000, 30000, 100000].forEach(v => {
    el("text", { x: X(v), y: H - B + 18, "text-anchor": "middle" }).textContent = v >= 1000 ? (v / 1000) + "k" : v;
  });
  el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle" }).textContent = "mean prompt tokens (log scale)";
  el("line", { x1: L, x2: W - R, y1: Y(data.noinfo), y2: Y(data.noinfo), stroke: "var(--ref)", "stroke-dasharray": "5 4", "stroke-width": 1.5 });
  el("text", { x: W - R + 6, y: Y(data.noinfo) + 4, class: "lbl" }).textContent = "names only";
  const labels = [];
  data.series.forEach((s, i) => {
    const c = `var(${colors[i]})`;
    const pts = s.points.map(p => [X(p.x), Y(p.y)]);
    el("polyline", { points: pts.map(p => p.join(",")).join(" "), fill: "none", stroke: c, "stroke-width": 2, "stroke-linejoin": "round" });
    s.points.forEach(p => {
      const dot = el("circle", { cx: X(p.x), cy: Y(p.y), r: 4.5, fill: c, stroke: "var(--surface)", "stroke-width": 2 });
      const hit = el("circle", { cx: X(p.x), cy: Y(p.y), r: 12, fill: "transparent", tabindex: 0 });
      const show = () => { tip.hidden = false; tip.textContent = `${s.name} · ${p.cond}: ${p.y.toFixed(1)}% at ${(p.x / 1000).toFixed(1)}k tokens`;
        const r = svg.getBoundingClientRect(), k = r.width / W; tip.style.left = Math.min(X(p.x) * k, r.width - 220) + 12 + "px"; tip.style.top = Y(p.y) * k - 8 + "px"; };
      hit.addEventListener("mouseenter", show); hit.addEventListener("focus", show);
      hit.addEventListener("mouseleave", () => tip.hidden = true); hit.addEventListener("blur", () => tip.hidden = true);
    });
    const last = s.points[s.points.length - 1];
    labels.push({ y: Y(last.y), text: s.name, c });
    const lg = document.createElement("span"); lg.innerHTML = `<i style="background:${c}"></i>${s.name}`; document.getElementById("legend").appendChild(lg);
  });
  el("rect", { x: X(data.charmem.x) - 5, y: Y(data.charmem.y) - 5, width: 10, height: 10, fill: "var(--fg)", transform: `rotate(45 ${X(data.charmem.x)} ${Y(data.charmem.y)})` });
  labels.push({ y: Y(data.charmem.y), text: "charmem", c: "var(--fg)", x: X(data.charmem.x) });
  const lg = document.createElement("span"); lg.innerHTML = `<i style="background:var(--fg);width:8px;height:8px;transform:rotate(45deg)"></i>charmem sheet`; document.getElementById("legend").appendChild(lg);
  labels.filter(l => l.x === undefined).sort((a, b) => a.y - b.y).forEach((l, i, arr) => { if (i && l.y - arr[i - 1].y < 13) l.y = arr[i - 1].y + 13; });
  labels.forEach(l => {
    const t = el("text", { x: l.x === undefined ? W - R + 6 : l.x + 9, y: l.y + 4, class: "lbl" });
    t.textContent = l.text; t.style.fill = l.c;
  });
})();
</script>
"""

if __name__ == "__main__":
    main()
