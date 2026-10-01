"""Queue every eval the "Best of each kind" chart needs but lacks: for each condition on any model's best-of-kind line
(make_report.BEST_CATS, chosen without the chapter so far), the condition and its +pre version on all four model
setups (Qwen3.8-27B thinking off / on, Qwen3.5-9B base, Qwen3-4B).

  python3 scripts/queue_missing.py [--dry] [--model=Qwen3.8-27B_think ...]
A (model, condition) counts as present when its analysis row covers at least half the books. Thinking-on jobs take two
conditions and are split 6 / 1 / 2 ways (train / val / test), about 1.2 GPU-hours each; prompts whose 90th-percentile length leaves under 34k tokens of a 65k context for
reasoning run at 262k on 2 GPUs. TIER1 (hybrids, Story Information, plot + sheets, the new gpt-oss summary) is queued
ahead of the rest; SKIP drops expensive ablations that sit on no line that matters. Finished items are skipped on
restart, so re-running this is safe.
"""
import json
import sys

sys.path.insert(0, "/home/toolkit/chiron_replication/chiron")
sys.path.insert(0, "/home/toolkit/chiron_replication/scripts")
import make_report as M  # noqa: E402
from queue_jobs import addraw, env, ev, srv  # noqa: E402

OUT = M.OUT
MODELS = ["Qwen3.8-27B_nothink", "Qwen3.8-27B_think", "Qwen3.5-9B-Base", "Qwen3-4B-Instruct-2507"]
A = {m: json.load(open(OUT / f"analysis_main_{m}.json")) for m in MODELS}


def ok(m, c):
    r = A[m]["rows"].get(c)
    if r is None:
        return False
    books = r["joint"]["books"] if m != "Qwen3.8-27B_think" else r["books"]
    return books >= len(A[m]["books"]) / 2


front = set()
for m in ("Qwen3.8-27B_nothink", "Qwen3.5-9B-Base", "Qwen3-4B-Instruct-2507", "Qwen3.8-27B_think"):
    rows = A[m]["rows"]
    conds = [c for c in rows if not c.endswith("+pre") and rows[c]["tokens_q"] and c != "noinfo" and ok(m, c)]
    score = (lambda c: rows[c]["joint"]["macro"]) if m != "Qwen3.8-27B_think" else (lambda c: rows[c]["macro"])
    for cat, col, dash, shape, pred in M.BEST_CATS:
        top = -1
        for x, ny, c in sorted((rows[c]["tokens_q"][2], -score(c), c) for c in conds if pred(c)):
            if -ny > top:
                front.add(c)
                top = -ny
# skipped: expensive ablations off every line that matters (107k-token Llama-notes ablations), superseded nested sheets
SKIP = {"legacy_nofill", "legacy_noprev", "book_noprev8000", "plot_recap300", "sheet_bible_leg", "sheet_chiron_leg",
        "plot_h_global_500", "summary_h", "legacy_gptoss_ent4", "chiron_r250"}
# first: what could be on a best-of-kind line or answers a question we are asking (hybrids, Story Information)
TIER1 = {"legacy&book_last500", "legacy&book_last1000", "legacy&book_last1500", "charmem&book_last500", "charmem&book_last1000",
         "summary&book_last1000", "summary@500&book_last1000", "summary_h&book_last1000", "sheet_legsum_ent_500_flat",
         "sheet_legsum_ent_500_flat&book_last1000", "sheet_legsum_ent_500_flat&book_last1500", "sheet_chiron_leg_flat",
         "charmem&ncp_plot", "charmem&ncp_plot@last1000", "v2&ncp_plot", "summary&ncp_plot", "v2&book_ch2", "charmem&ncp_story",
         "v2&ncp_story", "ncp_story", "ncp_plot", "ncp_plot@last1000", "legacy&plot_global_1000", "legacy_r3000", "legacy_r6000",
         "legacy_gptoss_ent"}
front = (front | TIER1) - SKIP
p90 = {c: max((A[m]["rows"].get(c) or {}).get("tokens_q", [0] * 5)[4] for m in MODELS if c in A[m]["rows"]) for c in front}
missing = {m: sorted(c2 for c in front for c2 in (c, c + "+pre") if not ok(m, c2)) for m in MODELS}
for m, cs in missing.items():
    print(m, len(cs), cs)
if "--dry" in sys.argv:
    raise SystemExit
long = lambda c: p90.get(c.removesuffix("+pre"), 0) + (3000 if c.endswith("+pre") else 0) > 31000
tier = lambda c: 1 if c.removesuffix("+pre") in TIER1 else 2
only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")]
for m, cs in missing.items():
    if only and m not in only:
        continue
    for lng, t in ((False, 1), (True, 1), (False, 2), (True, 2)):
        group = [c for c in cs if long(c) == lng and tier(c) == t]
        if not group:
            continue
        g = 2 if lng else 1
        per = 2 if m == "Qwen3.8-27B_think" else 4                  # thinking: ~1.2 h jobs pack better than ~5 h ones
        for i in range(0, len(group), per):
            chunk = group[i:i + per]
            tag = f"miss_{m.split('_')[-1].split('-')[0].lower()}_t{t}{'L' if lng else 'S'}{i}"
            for split in ("test", "val", "train"):
                if m == "Qwen3.8-27B_think":
                    n = {"train": 6, "val": 1, "test": 2}[split]
                    for k in range(n):
                        addraw(f"{tag}_{split}_s{k}of{n}", g, env(CHIRON_THINKING=1) + srv("qwen27", lng)
                               + ev("eval_reason.py", split, f"items_{split}", chunk, k, n), -1 if t == 1 else 2)
                elif m == "Qwen3.8-27B_nothink":
                    n = 3 if split == "train" else 1
                    for k in range(n):
                        addraw(f"{tag}_{split}_s{k}of{n}", g, env(CHIRON_THINKING=0) + srv("qwen27", lng)
                               + ev("eval_mcp.py", split, f"items_{split}", chunk, k, n, ["--rotations"]), -2)
                elif m == "Qwen3.5-9B-Base":
                    addraw(f"{tag}_{split}", g, env(CHIRON_BASE=1) + srv("qwen9base", lng)
                           + ev("eval_mcp.py", split, f"items_{split}", chunk, extra=["--rotations"]), -2)
                else:
                    addraw(f"{tag}_{split}", g, srv("qwen4b", lng) + ev("eval_mcp.py", split, f"items_{split}", chunk), -2)
