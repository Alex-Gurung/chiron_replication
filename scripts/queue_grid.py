"""Queue the same-model grid's evaluations: every generator (l70 = Llama-3.3-70B, gptoss = gpt-oss-120b, q4b = Qwen3-4B)
writes the same kinds of representation, and each is scored on character identification and on the NCP writer ruler.

  python3 scripts/queue_grid.py charid GEN [--judges 27 small llama]   masked-character evals of GEN's representations
  python3 scripts/queue_grid.py ncp TAG EVERY MODEL [--lean] GEN...    NCP ruler wave for those generators' conditions
CHIRON_LANE picks the worker lane for the 27 / small / ncp jobs (default chiron).
Representations per generator (GRID): CH = CHIRON condensed sheet, RS = rolling character summary, PS = the CHIRON
paper's character summary, PL = plot summary of about 1,000 words (the matched budget), FULL = CHIRON notes in full.
charid conditions: CH, RS, PS, PL, two more plot summaries, CH&PL, RS&PL (and FULL on the 27B at 262k).
ncp conditions: ship|X and none|X for X in CH, RS, PS; PL|none, PL|CH, PL|RS; with the references ship|v2, ship|none,
none|v2, none|none.
Judges: 27 = Qwen3.8-27B thinking off (lane chiron), small = Qwen3.5-9B base and Qwen3-4B (lane chiron), llama =
Llama-3.3-70B on 4 GPUs (lane chiron_l70; also scores the references noinfo, v2, charmem).
"""
import sys
import time

sys.path.insert(0, "/home/toolkit/eaiexp")
sys.path.insert(0, "/home/toolkit/chiron_replication/scripts")
import jobqueue as q  # noqa: E402
from queue_jobs import REPO, addraw, env, ev, srv  # noqa: E402

GRID = {
    "l70": {"CH": "legacy", "RS": "summary_l70", "PS": "sheet_csum_l70_flat", "PL": "plot_l70_global_1000", "FULL": "legacy_full",
            "PLX": ["plot_l70_hier_1000", "plot_l70_global_2000"]},
    "gptoss": {"CH": "sheet_legsum_x_500_flat", "RS": "summary", "PS": "sheet_csum_flat", "PSF": "sheet_csum_fd_flat", "PL": "plot_global_1000", "FULL": "legacy_gptoss_x",
               "PLX": ["plot_hier_1000", "plot_global_2000"]},
    "q4b": {"CH": "sheet_legsum_q4b_x_500_flat", "CHL": "sheet_legsum_q4b_x_flat", "RS": "summary_q4b", "PS": "sheet_csum_q4b_flat", "PSF": "sheet_csum_q4b_fd_flat", "PL": "plot_q4b_global_1000", "FULL": "legacy_q4b_x",
            "PLX": ["plot_q4b_hier_1000", "plot_q4b_global_2000"]},
}
REFS = ["ship|v2", "ship|none", "none|v2", "none|none"]


def charid_conds(g):
    """+ the rolling summary cut to 500 words, the paper's summary filtered and de-duplicated (PSF), an uncapped CHIRON
    condensed sheet (CHL) where they exist."""
    r = GRID[g]
    return ([r["CH"], r["RS"], r["PS"], r["PL"], *r["PLX"], f"{r['CH']}&{r['PL']}", f"{r['RS']}&{r['PL']}", f"{r['RS']}@500"]
            + [r[k] for k in ("PSF", "CHL") if k in r])


def ncp_conds(g, lean=False):
    """lean: without the paper's summary and the rolling summary + plot pair (6 conditions instead of 9)."""
    r = GRID[g]
    kinds = ("CH", "RS") if lean else ("CH", "RS", "PS")
    return ([f"{p}|{r[x]}" for p in ("ship", "none") for x in kinds]
            + [f"{r['PL']}|none", f"{r['PL']}|{r['CH']}"] + ([] if lean else [f"{r['PL']}|{r['RS']}"]))


def main():
    what = sys.argv[1]
    stamp = time.strftime("%m%d%H%M", time.gmtime())
    if what == "charid":
        g = sys.argv[2]
        judges = sys.argv[sys.argv.index("--judges") + 1:] if "--judges" in sys.argv else ["27", "small"]
        conds = charid_conds(g)
        for split in ("test", "val", "train"):
            if "27" in judges:
                n = 3 if split == "train" else 1
                for k in range(n):
                    for part, cs in enumerate([conds[i:i + 4] for i in range(0, len(conds), 4)]):   # short conditions, four a job
                        addraw(f"gc27_{g}_{part}_{split}_s{k}of{n}", 1, env(CHIRON_THINKING=0) + srv("qwen27", False)
                               + ev("eval_mcp.py", split, f"items_{split}", cs, k, n, ["--rotations"]), -3)
                    if g == "q4b":                                         # the other generators' full notes are already scored
                        addraw(f"gc27_{g}_full_{split}_s{k}of{n}", 2, env(CHIRON_THINKING=0) + srv("qwen27", True)
                               + ev("eval_mcp.py", split, f"items_{split}", [GRID[g]["FULL"]], k, n, ["--rotations"]), -2)
            if "small" in judges:
                addraw(f"gcs_{g}_{split}_q9b", 1, env(CHIRON_BASE=1) + srv("qwen9base", False) + ev("eval_mcp.py", split, f"items_{split}", conds, extra=["--rotations"]), -3)
                addraw(f"gcs_{g}_{split}_q4b", 1, srv("qwen4b", False) + ev("eval_mcp.py", split, f"items_{split}", conds), -3)
            if "llama" in judges:
                cs = conds + (["noinfo", "v2", "charmem"] if g == "l70" else [])
                n = 2 if split == "train" else 1
                for k in range(n):
                    name = f"chiron_gcl_{g}_{split}_s{k}of{n}_{stamp}"
                    ok = q.add(q.conn(), name, srv("llama70", False) + ev("eval_mcp.py", split, f"items_{split}", cs, k, n), gpus=4, lane="chiron_l70", priority=-3)
                    print(("queued " if ok else "exists ") + name)
    elif what == "ncp":
        tag, every, model = sys.argv[2], sys.argv[3], sys.argv[4]
        lean, gens = "--lean" in sys.argv, [a for a in sys.argv[5:] if a != "--lean"]
        conds = REFS + [c for g in gens for c in ncp_conds(g, lean)]
        short = "4b" if "4B" in model else "27"
        for k in range(8):
            addraw(f"grid{short}_{tag}_s{k}of8", 1, ["bash", f"{REPO}/scripts/ncp_ruler_grid.sh", str(k), "8", tag, every, model, *conds], -3)
        print(len(conds), "conditions")


if __name__ == "__main__":
    main()
