"""Add chiron_replication jobs to the eaiexp file queue on lane "chiron".

  python3 scripts/queue_jobs.py smoke           one small job of each kind
  python3 scripts/queue_jobs.py chiron          CHIRON-style statements, all books (test books first)
  python3 scripts/queue_jobs.py summary         rolling character summaries, all books
  python3 scripts/queue_jobs.py charmem         finish the Sep 8 charmem rebuild, all books
  python3 scripts/queue_jobs.py eval NAME MAXLEN MODEL ITEMS COND...   one eval job (MODEL: qwen4b|mistral)
  python3 scripts/queue_jobs.py final SPLIT...  every condition for the given splits (needs items/reps built)
  python3 scripts/queue_jobs.py reshard         re-split the slow thinking-on short/window shards wider
  python3 scripts/queue_jobs.py sections SPLIT...  sheet-section ablation on the 27B, two conditions a job
  python3 scripts/queue_jobs.py long27          27B prompts over 131k tokens, rerun at 262k on 2 GPUs
  python3 scripts/queue_jobs.py quote|manual   item-level oracles from manual_oracle.py on every model
  python3 scripts/queue_jobs.py bookch          book text by last 1/2/4/8 chapters and the chapter so far, every model
  python3 scripts/queue_jobs.py sweep_think     the representation-length sweep with thinking on
  python3 scripts/queue_jobs.py chapnotes       the Llama CHIRON notes redone with gpt-oss (per chapter)
  python3 scripts/queue_jobs.py chapnotes_eval|plot_eval   evaluate the gpt-oss chapter notes / plot summaries, every model
  python3 scripts/queue_jobs.py narrator        who narrates each chapter in the first person (gpt-oss)
  python3 scripts/queue_jobs.py headings_gen    every gpt-oss representation again, with chapter headings and narrators
  python3 scripts/queue_jobs.py headings_eval   evaluate the with-headings representations on every model
  python3 scripts/queue_jobs.py pre             every representation + the passage's chapter so far (<cond>+pre)
  python3 scripts/queue_jobs.py legacy_gptoss   the Llama notes' extraction prompt run with gpt-oss
  python3 scripts/queue_jobs.py long_notes      thorough gpt-oss chapter notes (with headings)
  python3 scripts/queue_jobs.py eval_conds TAG LONG COND...   evaluate conditions on every model
  python3 scripts/queue_jobs.py sheet VARIANT SOURCE WORDS STYLE NJOBS [--limit N]   new gpt-oss sheets (gen_sheet.py)
  python3 scripts/queue_jobs.py eval27 TAG LONG COND...   Qwen3.8-27B without thinking only (fast iteration)
  python3 scripts/queue_jobs.py legsum VARIANT SOURCE [WORDS]   the Llama notes' summary step with gpt-oss (gen_legsum.py)
  python3 scripts/queue_jobs.py charsynth VARIANT SOURCE [WORDS]   charmem's exact synthesis on other notes (gen_charsynth.py)
  python3 scripts/queue_jobs.py evalsmall TAG LONG COND...   Qwen3.5-9B base and Qwen3-4B only
  python3 scripts/queue_jobs.py merge VARIANT WORDS V1 V2 V3...   consensus of sheet samples (gen_merge.py)
  python3 scripts/queue_jobs.py entail          gpt-oss entailment ratings for the gpt-oss Llama-prompt notes (gen_entail.py)
  python3 scripts/queue_jobs.py plot            plot summaries with gpt-oss (one pass; chapter by chapter), 6 book groups each
  python3 scripts/queue_jobs.py oracle_shards   thinking-on prior-facts oracle on train, split 8 ways
  python3 scripts/queue_jobs.py gender_think    only the thinking-on gender-only shards, split wide
  python3 scripts/queue_jobs.py gender          gender-only representation on every model and set
Lower priority number is claimed first. Job names are stamped so reruns never collide.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/home/toolkit/eaiexp")
import jobqueue as q  # noqa: E402

REPO = "/home/toolkit/chiron_replication"
SERVE = ["python3", "-u", f"{REPO}/scripts/serve_and_run.py"]
TEST = ["dark", "god", "mercy", "witch"]
STAMP = time.strftime("%m%d%H%M", time.gmtime())
BASE = q.conn()


def principals():
    return json.load(open(f"{REPO}/data/principals.json"))


def add(name, gpus, model_args, client, priority):
    ok = q.add(BASE, f"chiron_{name}_{STAMP}", SERVE + model_args + ["--", "python3", "-u", *client],
               gpus=gpus, lane="chiron", priority=priority)
    print(("queued " if ok else "exists ") + f"chiron_{name}_{STAMP}")


def addraw(name, gpus, cmd, priority):
    ok = q.add(BASE, f"chiron_{name}_{STAMP}", cmd, gpus=gpus, lane="chiron", priority=priority)
    print(("queued " if ok else "exists ") + f"chiron_{name}_{STAMP}")


def env(**kv):
    return ["env", *[f"{k}={v}" for k, v in kv.items()]]


def srv(model, tp_long):
    return ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", model, "--max-model-len", "262144" if tp_long else "65536", "--"]


def ev(script, split, stem, c, k=0, n=1, extra=()):
    conds = [c] if isinstance(c, str) else list(c)
    return ["python3", "-u", f"{REPO}/chiron/{script}", "--split", split, "--items", stem, "--conditions", *conds,
            "--workers", "128", "--shard", str(k), "--nshards", str(n), *extra]


def tag(c):
    return c.replace(":", "_")


def gen_client(books, shard, nshards, *extra):
    return [f"{REPO}/chiron/gen_chiron.py", "--books", *books, "--shard", str(shard), "--nshards", str(nshards), *extra]


def main():
    what = sys.argv[1]
    oss1 = ["--model", "gptoss", "--max-model-len", "32768"]
    oss1_long = ["--model", "gptoss", "--max-model-len", "40960"]
    oss2 = ["--model", "gptoss", "--max-model-len", "131072"]
    if what == "smoke":
        add("smoke_gen", 1, oss1, gen_client(["dark"], 0, 50, "--limit", "24"), -10)
        add("smoke_sum", 1, oss1_long, [f"{REPO}/chiron/gen_summary.py", "--books", "dark", "--tag", "smoke", "--upto", "3"], -10)
        add("smoke_charmem", 2, oss2, [f"{REPO}/chiron/charmem_finish.py", "--books", "blue", "--width", "8"], -10)
    elif what == "chiron":
        # Mixed tensor-parallel on purpose: 1-GPU copies are KV-limited (~100k tokens), 2-GPU copies
        # hold ~1.9M; compare generation throughput per GPU in the vllm logs.
        p = principals()
        others = sorted(b for b in p if b not in TEST)
        oss2s = ["--model", "gptoss", "--max-model-len", "32768"]
        for k in range(12):
            g = 1 if k < 8 else 2
            add(f"gen_test_{k:02d}", g, oss1 if g == 1 else oss2s, gen_client(TEST, k, 12, "--workers", "48" if g == 1 else "192"), 0)
        for k in range(32):
            g = 1 if k < 24 else 2
            add(f"gen_rest_{k:02d}", g, oss1 if g == 1 else oss2s, gen_client(others, k, 32, "--workers", "48" if g == 1 else "192"), 5)
    elif what == "summary":
        p = principals()
        add("sum_test", 1, oss1_long, [f"{REPO}/chiron/gen_summary.py", "--books", *TEST, "--tag", "test"], 0)
        others = sorted((b for b in p if b not in TEST), key=lambda b: -p[b]["n_chapters"])
        for k in range(3):
            books = others[k::3]
            add(f"sum_rest_{k}", 1, oss1_long, [f"{REPO}/chiron/gen_summary.py", "--books", *books, "--tag", f"rest{k}"], 5)
    elif what == "charmem":
        p = principals()
        skip = set(sys.argv[2:])                      # books another job already owns
        books = sorted((b for b in p if b not in skip), key=lambda b: -p[b]["n_chapters"])
        groups = [books[k::10] for k in range(10)]
        for k, g in enumerate(groups):
            add(f"charmem_{k}", 2, oss2, [f"{REPO}/chiron/charmem_finish.py", "--books", *g, "--width", "16"], 3)
    elif what == "memo":
        for model in ("qwen4b", "mistral"):
            add(f"memo_{model}", 1, ["--model", model, "--max-model-len", "8192"],
                [f"{REPO}/chiron/memo_probe.py", "--books", *TEST, "--n", "20"], -1)
    elif what == "requeue":                        # rerun a job's exact command under a fresh name (outputs resume)
        old = json.load(open(q._path(BASE, sys.argv[2], q._find(BASE, sys.argv[2]))))
        name = old["name"].rsplit("_", 1)[0].replace("chiron_", "", 1)
        ok = q.add(BASE, f"chiron_{name}_{STAMP}", old["cmd"], gpus=old["gpus"], lane="chiron", priority=old["priority"])
        print(("queued " if ok else "exists ") + f"chiron_{name}_{STAMP}")
    elif what == "pronouns":
        for split in sys.argv[2:] or ["test"]:
            add(f"pronouns_{split}", 1, oss1, [f"{REPO}/chiron/pronouns.py", "--split", split], -1)
    elif what == "final":
        short = ["noinfo", "v2", "swap:v2", "legacy", "summary", "swap:summary", "charmem", "swap:charmem",
                 "chiron_r250", "chiron_r500", "chiron_r1000", "chiron_r2000", "chiron_r4000",
                 "v2@100", "v2@250", "v2@500", "legacy@100", "legacy@250", "summary@100", "summary@250", "summary@500",
                 "book_last2000", "book_last8000"]
        long = ["chiron", "swap:chiron", "chiron_physical", "chiron_dialogue", "chiron_knowledge", "chiron_goals",
                "legacy_full", "book_last32000", "book"]
        pron = ["noinfo", "v2", "legacy", "summary", "charmem", "chiron_r2000", "chiron", "book_last8000"]
        mistral = ["noinfo", "v2", "swap:v2", "legacy", "summary", "charmem", "chiron_r500", "chiron_r2000",
                   "book_last2000", "book_last8000"]
        two_short = ["noinfo", "v2", "swap:v2", "legacy", "summary", "swap:summary", "charmem", "swap:charmem",
                     "chiron_r500", "chiron_r2000", "book_last8000"]
        two_long = ["chiron", "swap:chiron", "legacy_full", "book"]
        splits = [a for a in sys.argv[2:] if not a.startswith("--")]
        if "--ready-only" in sys.argv:              # conditions that do not need summaries or charmem
            keep = lambda c: "summary" not in c and "charmem" not in c
            short, long, pron, mistral = ([c for c in x if keep(c)] for x in (short, long, pron, mistral))
            two_short, two_long = ([c for c in x if keep(c)] for x in (two_short, two_long))
        shards = {"train": 8, "val": 2, "test": 2}
        for split in splits:
            ev4 = lambda stem, conds, k=0, n=1: [f"{REPO}/chiron/eval_mcp.py", "--split", split, "--items", stem,
                                                "--conditions", *conds, "--workers", "128", "--shard", str(k), "--nshards", str(n)]
            for c in short:
                add(f"final_{split}_{c.replace(':', '_').replace('@', 'at')}", 1, ["--model", "qwen4b", "--max-model-len", "65536"], ev4(f"items_{split}", [c]), 1)
            for c in long:
                n = shards[split]
                for k in range(n):
                    add(f"final_{split}_{c.replace(':', '_')}_s{k}", 1, ["--model", "qwen4b", "--max-model-len", "262144"], ev4(f"items_{split}", [c], k, n), 1)
            for c in pron:
                n = shards[split] if c == "chiron" else 1
                for k in range(n):
                    add(f"final_{split}_pron_{c.replace(':', '_')}_s{k}", 1, ["--model", "qwen4b", "--max-model-len", "262144"], ev4(f"items_{split}_pron", [c], k, n), 1)
            for c in mistral:
                add(f"final_{split}_mistral_{c.replace(':', '_')}", 1, ["--model", "mistral", "--max-model-len", "32768"], ev4(f"items_{split}", [c]), 2)
            for c in two_short:
                add(f"final_{split}_two_{c.replace(':', '_').replace('@', 'at')}", 1, ["--model", "qwen4b", "--max-model-len", "65536"], ev4(f"items_{split}_two", [c]), 2)
            for c in two_long:
                n = shards[split]
                for k in range(n):
                    add(f"final_{split}_two_{c.replace(':', '_')}_s{k}", 1, ["--model", "qwen4b", "--max-model-len", "262144"], ev4(f"items_{split}_two", [c], k, n), 2)
    elif what == "q27":
        # Qwen3.8-27B: direct scoring with thinking off (main, dense-window and short sets), then the thinking-on
        # run at the back of the queue (priority 9).
        short = ["noinfo", "summary", "v2", "legacy", "charmem", "chiron_r2000", "book_last8000", "swapname:v2"]
        long = ["chiron", "legacy_full", "book_last32000", "swapname:chiron"]
        reason = ["noinfo", "summary", "v2", "legacy", "charmem", "chiron_r2000", "chiron", "legacy_full", "book_last8000", "swapname:v2"]
        for split in ("test", "val", "train"):
            for stem in (f"items_{split}", f"items_{split}_window", f"items_{split}_short"):
                kind = stem.split("_")[-1] if stem.count("_") > 1 else "main"
                for c in short + long:
                    if kind != "main" and c in ("book_last32000", "swapname:chiron"):
                        continue
                    lng = c in long
                    n = {"train": 8 if lng else 2, "val": 2 if lng else 1, "test": 2 if lng else 1}[split] if kind != "window" else 1
                    for k in range(n):
                        addraw(f"q27_{kind}_{split}_{tag(c)}_s{k}", 2 if lng else 1, env(CHIRON_THINKING=0) + srv("qwen27", lng) + ev("eval_mcp.py", split, stem, c, k, n, ["--rotations"]), 1)
            for c in reason:
                lng = c in long
                n = {"train": 8 if lng else 4, "val": 2, "test": 2}[split]
                for k in range(n):
                    addraw(f"q27think_{split}_{tag(c)}_s{k}", 2 if lng else 1, env(CHIRON_THINKING=1) + srv("qwen27", lng) + ev("eval_reason.py", split, f"items_{split}", c, k, n), 9)
    elif what == "reshard":
        # slow thinking-on shards, re-split wider; finished (item, rotation) pairs are skipped on start
        for split, stem, c, n in [("train", "items_train_short", "v2", 16), ("train", "items_train_short", "swapname:v2", 16),
                                  ("train", "items_train_short", "chiron_r2000", 12), ("val", "items_val_short", "swapname:v2", 4),
                                  ("val", "items_val_short", "chiron_r2000", 4), ("train", "items_train_window", "noinfo", 4),
                                  ("train", "items_train_window", "swapname:v2", 6), ("train", "items_train_window", "chiron_r2000", 4)]:
            for k in range(n):
                addraw(f"q27think_rs_{stem[6:]}_{tag(c)}_s{k}of{n}", 1, env(CHIRON_THINKING=1) + srv("qwen27", False) + ev("eval_reason.py", split, stem, c, k, n), 1)
    elif what == "sections":
        # one v2/charmem sheet section at a time, and everything but relationships (27B, thinking off), two conditions a job
        conds = [f"{src}_{x}" for src in ("v2", "charmem") for x in ("sec_physical", "sec_dialogue", "sec_history", "sec_knowledge",
                                                                     "sec_goals", "sec_relationships", "norel")]
        for split in sys.argv[2:]:
            for k in range(0, len(conds), 2):
                addraw(f"sec27_{split}_{k // 2}", 1, env(CHIRON_THINKING=0) + srv("qwen27", False) + [
                    "python3", "-u", f"{REPO}/chiron/eval_mcp.py", "--split", split, "--items", f"items_{split}",
                    "--conditions", *conds[k:k + 2], "--workers", "128", "--rotations"], 0)
    elif what == "long27":
        # 27B, thinking off: prompts over 131k tokens (whole book, combinations) at 262k on 2 GPUs; finished ones are skipped
        for split in ("test", "val", "train"):
            for c in ("book", "combo_legacy_v2", "combo_all"):
                n = 4 if split == "train" else 1
                for k in range(n):
                    addraw(f"long27_{split}_{c}_s{k}of{n}", 2, env(CHIRON_THINKING=0) + srv("qwen27", True)
                           + ev("eval_mcp.py", split, f"items_{split}", c, k, n, ["--rotations"]), 0)
    elif what in ("quote", "manual"):
        # item-level oracles from manual_oracle.py: the verbatim-quote oracle (every item) or the agent-written ones (63 items)
        conds = ["oracle_quote"] if what == "quote" else ["manual_passage", "manual_prior", "swapname:manual_prior"]
        for split in ("test", "val", "train"):
            for c in conds:
                t = f"{what}_{split}_{tag(c)}"
                addraw(f"{t}_q27", 1, env(CHIRON_THINKING=0) + srv("qwen27", False) + ev("eval_mcp.py", split, f"items_{split}", c, extra=["--rotations"]), 0)
                addraw(f"{t}_q9b", 1, env(CHIRON_BASE=1) + srv("qwen9base", False) + ev("eval_mcp.py", split, f"items_{split}", c, extra=["--rotations"]), 0)
                addraw(f"{t}_q4b", 1, srv("qwen4b", False) + ev("eval_mcp.py", split, f"items_{split}", c), 0)
                n = 4 if split == "train" and what == "quote" else 1
                for k in range(n):
                    addraw(f"{t}_q27think_s{k}of{n}", 1, env(CHIRON_THINKING=1) + srv("qwen27", False) + ev("eval_reason.py", split, f"items_{split}", c, k, n), 1)
    elif what == "bookch":
        # book text by whole chapters (last 1/2/4/8) and the passage's own chapter so far, on every model
        for split in ("test", "val", "train"):
            for conds, lng in ((["book_ch1", "book_ch2", "book_prefix", "book_ch1p"], False), (["book_ch4", "book_ch8"], True)):
                n = 4 if split == "train" else 1
                t = f"bookch_{split}_{'long' if lng else 'short'}"
                for k in range(n):
                    addraw(f"{t}_q27_s{k}of{n}", 2 if lng else 1, env(CHIRON_THINKING=0) + srv("qwen27", lng)
                           + ev("eval_mcp.py", split, f"items_{split}", conds, k, n, ["--rotations"]), 0)
                    addraw(f"{t}_q27think_s{k}of{n}", 2 if lng else 1, env(CHIRON_THINKING=1) + srv("qwen27", lng)
                           + ev("eval_reason.py", split, f"items_{split}", conds, k, n), 1)
                addraw(f"{t}_q9b", 1, env(CHIRON_BASE=1) + srv("qwen9base", lng) + ev("eval_mcp.py", split, f"items_{split}", conds, extra=["--rotations"]), 0)
                addraw(f"{t}_q4b", 1, srv("qwen4b", lng) + ev("eval_mcp.py", split, f"items_{split}", conds), 0)
    elif what == "sweep_think":
        # the length sweep with thinking on (Qwen3.8-27B), for the representation-length chart
        groups = [(["chiron_r250", "chiron_r500", "chiron_r1000", "chiron_r4000"], False), (["v2@100", "v2@250", "v2@500"], False),
                  (["legacy@100", "legacy@250", "summary@100"], False), (["summary@250", "summary@500", "book_last2000"], False),
                  (["book_last32000"], True)]
        for split in ("test", "val", "train"):
            n = {"train": 8, "val": 1, "test": 2}[split]
            for g, (conds, lng) in enumerate(groups):
                for k in range(n):
                    addraw(f"sweepthink_{split}_g{g}_s{k}of{n}", 2 if lng else 1, env(CHIRON_THINKING=1) + srv("qwen27", lng)
                           + ev("eval_reason.py", split, f"items_{split}", conds, k, n), 2)
    elif what == "chapnotes":
        # the Llama CHIRON notes redone with gpt-oss (per chapter, unfiltered), 8 book groups on 1 GPU each
        books = sorted({json.loads(l)["book"] for s in ("test", "val", "train") for l in open(f"{REPO}/data/items_{s}.jsonl")})
        for k in range(8):
            addraw(f"chapnotes_{k}", 1, ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "65536",
                                        "--", "python3", "-u", f"{REPO}/chiron/gen_chapnotes.py", "--books", *books[k::8], "--workers", "64"], 0)
    elif what in ("chapnotes_eval", "plot_eval"):
        # evaluate the gpt-oss chapter notes (long: 262k) or the plot summaries (short) on every model
        if what == "chapnotes_eval":
            conds, lng = ["chapnotes"], True
        else:
            conds = [f"plot_{k}_{n}" for k in ("global", "hier") for n in (500, 1000, 2000, 4000)] + \
                    [f"plot_{k}_4000@last{n}" for k in ("global", "hier") for n in (500, 1000, 2000)]
            lng = False
        for split in ("test", "val", "train"):
            n = 4 if split == "train" else 1
            for k in range(n):
                addraw(f"{what}_{split}_q27_s{k}of{n}", 2 if lng else 1, env(CHIRON_THINKING=0) + srv("qwen27", lng) + ev("eval_mcp.py", split, f"items_{split}", conds, k, n, ["--rotations"]), 0)
                addraw(f"{what}_{split}_q27think_s{k}of{n}", 2 if lng else 1, env(CHIRON_THINKING=1) + srv("qwen27", lng) + ev("eval_reason.py", split, f"items_{split}", conds, k, n), 1)
            addraw(f"{what}_{split}_q9b", 1, env(CHIRON_BASE=1) + srv("qwen9base", lng) + ev("eval_mcp.py", split, f"items_{split}", conds, extra=["--rotations"]), 0)
            addraw(f"{what}_{split}_q4b", 1, srv("qwen4b", lng) + ev("eval_mcp.py", split, f"items_{split}", conds), 0)
    elif what == "narrator":
        # who narrates each chapter (for the chapter-heading/narrator hints), 4 book groups on 1 GPU each
        books = sorted({json.loads(l)["book"] for s in ("test", "val", "train") for l in open(f"{REPO}/data/items_{s}.jsonl")})
        for k in range(4):
            addraw(f"narrator_{k}", 1, ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "65536",
                                       "--", "python3", "-u", f"{REPO}/chiron/gen_narrator.py", "--books", *books[k::4], "--workers", "64"], -1)
    elif what == "headings_gen":
        # regenerate every gpt-oss representation with chapter headings and narrator hints, after the narrator jobs
        deps = sorted(f[:-5] for d in ("queued", "running", "done") for f in os.listdir(f"{q.QDIR}/{d}") if f.startswith("chiron_narrator_"))
        books = sorted({json.loads(l)["book"] for s in ("test", "val", "train") for l in open(f"{REPO}/data/items_{s}.jsonl")})
        oss = lambda n: ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", str(n), "--"]
        def dep(name, gpus, cmd, prio):
            ok = q.add(BASE, f"chiron_{name}_{STAMP}", cmd, gpus=gpus, lane="chiron", priority=prio, deps=deps)
            print(("queued " if ok else "exists ") + f"chiron_{name}_{STAMP}")
        for k in range(8):
            g = books[k::8]
            dep(f"h_chapnotes_{k}", 1, oss(65536) + ["python3", "-u", f"{REPO}/chiron/gen_chapnotes.py", "--books", *g, "--workers", "64", "--headings"], 0)
            dep(f"h_summary_{k}", 1, oss(40960) + ["python3", "-u", f"{REPO}/chiron/gen_summary.py", "--books", *g, "--tag", f"h{k}", "--headings"], 0)
        for k in range(6):
            g = books[k::6]
            dep(f"h_plot_global_{k}", 2, oss(131072) + ["python3", "-u", f"{REPO}/chiron/gen_plot.py", "global", "--books", *g, "--workers", str(len(g)), "--headings"], 1)
            dep(f"h_plot_hier_{k}", 1, oss(65536) + ["python3", "-u", f"{REPO}/chiron/gen_plot.py", "hier", "--books", *g, "--workers", str(len(g)), "--headings"], 1)
        for k in range(24):
            dep(f"h_claims_{k:02d}", 1, oss(32768) + ["python3", "-u", *gen_client(books, k, 24, "--workers", "48", "--headings")], 2)
    elif what == "headings_eval":
        # evaluate the with-headings representations on every model (long ones at 262k)
        groups = [(["chapnotes_h", "chiron_h"], True),
                  (["summary_h", "chiron_h_r2000"] + [f"plot_h_{k}_{n}" for k in ("global", "hier") for n in (500, 1000, 2000, 4000)], False)]
        if sys.argv[2:] == ["claims"]:                  # only the claims (they finish long before the rest)
            groups = [(["chiron_h"], True), (["chiron_h_r2000"], False)]
        for split in ("test", "val", "train"):
            n = 4 if split == "train" else 1
            for conds, lng in groups:
                t = f"heval{'c' if sys.argv[2:] == ['claims'] else ''}_{split}_{'long' if lng else 'short'}"
                for k in range(n):
                    addraw(f"{t}_q27_s{k}of{n}", 2 if lng else 1, env(CHIRON_THINKING=0) + srv("qwen27", lng) + ev("eval_mcp.py", split, f"items_{split}", conds, k, n, ["--rotations"]), 0)
                    addraw(f"{t}_q27think_s{k}of{n}", 2 if lng else 1, env(CHIRON_THINKING=1) + srv("qwen27", lng) + ev("eval_reason.py", split, f"items_{split}", conds, k, n), 1)
                addraw(f"{t}_q9b", 1, env(CHIRON_BASE=1) + srv("qwen9base", lng) + ev("eval_mcp.py", split, f"items_{split}", conds, extra=["--rotations"]), 0)
                addraw(f"{t}_q4b", 1, srv("qwen4b", lng) + ev("eval_mcp.py", split, f"items_{split}", conds), 0)
    elif what == "pre":
        # every representation again with the passage's chapter up to the passage (<cond>+pre); thinking on a core subset
        short = ["noinfo", "gender", "v2@100", "v2@250", "v2@500", "v2", "charmem", "legacy@100", "legacy@250", "legacy",
                 "summary@100", "summary@250", "summary@500", "summary", "chiron_r250", "chiron_r500", "chiron_r1000", "chiron_r2000",
                 "book_last2000", "book_last8000", "book_ch1", "book_ch2", "oracle_prior", "oracle_passage", "oracle_quote"] + \
                [f"plot_{k}_{n}" for k in ("global", "hier") for n in (500, 1000, 2000, 4000)] + \
                [f"plot_{k}_4000@last{n}" for k in ("global", "hier") for n in (500, 1000, 2000)]
        long = ["chiron_r4000", "chiron", "legacy_full", "legacy_nofill", "chapnotes", "combo_short", "combo_legacy_v2", "combo_all",
                "book_last32000", "book_ch4", "book_ch8", "book"]
        think = ["noinfo", "v2", "charmem", "summary", "chiron_r2000", "book_last8000", "book_ch1", "plot_global_500", "plot_hier_4000"]
        think_long = ["legacy_full", "chiron", "chapnotes"]
        P = lambda cs: [c + "+pre" for c in cs]
        for split in ("test", "val", "train"):
            n = 2 if split == "train" else 1
            groups = [(short[i::5], False) for i in range(5)] + [(long[i::3], True) for i in range(3)]
            for g, (conds, lng) in enumerate(groups):
                for k in range(n):
                    addraw(f"pre_{split}_g{g}_q27_s{k}of{n}", 2 if lng else 1, env(CHIRON_THINKING=0) + srv("qwen27", lng) + ev("eval_mcp.py", split, f"items_{split}", P(conds), k, n, ["--rotations"]), 1)
                addraw(f"pre_{split}_g{g}_q9b", 1, env(CHIRON_BASE=1) + srv("qwen9base", lng) + ev("eval_mcp.py", split, f"items_{split}", P(conds), extra=["--rotations"]), 1)
                addraw(f"pre_{split}_g{g}_q4b", 1, srv("qwen4b", lng) + ev("eval_mcp.py", split, f"items_{split}", P(conds)), 1)
            nt = {"train": 6, "val": 1, "test": 2}[split]
            for c in think + think_long:
                for k in range(nt):
                    lng = c in think_long
                    addraw(f"prethink_{split}_{c}_s{k}of{nt}", 2 if lng else 1, env(CHIRON_THINKING=1) + srv("qwen27", lng) + ev("eval_reason.py", split, f"items_{split}", c + "+pre", k, nt), 3)
    elif what == "long_notes":
        # thorough gpt-oss chapter notes (with headings), 8 book groups
        books = sorted({json.loads(l)["book"] for s in ("test", "val", "train") for l in open(f"{REPO}/data/items_{s}.jsonl")})
        for k in range(8):
            addraw(f"longnotes_{k}", 1, ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "65536",
                                        "--", "python3", "-u", f"{REPO}/chiron/gen_chapnotes.py", "--books", *books[k::8], "--workers", "64",
                                        "--headings", "--long"], -1)
    elif what == "legacy_gptoss":
        # the Llama notes' extraction prompt with gpt-oss, 8 book groups
        books = sorted({json.loads(l)["book"] for s in ("test", "val", "train") for l in open(f"{REPO}/data/items_{s}.jsonl")})
        for k in range(8):
            addraw(f"leggpt_{k}", 1, ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "65536",
                                     "--", "python3", "-u", f"{REPO}/chiron/gen_legacy_gptoss.py", "--books", *books[k::8], "--workers", "96"], -2)
    elif what == "sheet":
        # one sheet variant, NJOBS book groups, on 2 GPUs: a 1-GPU copy holds only a handful of 10-60k-token prompts at once
        variant, source, words, style, nj = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
        books = sorted({json.loads(l)["book"] for s in ("test", "val", "train") for l in open(f"{REPO}/data/items_{s}.jsonl")})
        for k in range(nj):
            addraw(f"sheet_{variant}_{k}", 2, ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "131072",
                                               "--", "python3", "-u", f"{REPO}/chiron/gen_sheet.py", "--variant", variant, "--source", source,
                                               "--words", words, "--style", style, "--books", *books[k::nj], "--workers", "128", *sys.argv[7:]], -3)
    elif what == "entail":
        # the Llama notes' entailment filter, with gpt-oss, on the gpt-oss Llama-prompt notes (6 book groups, 2 GPUs each)
        books = sorted({json.loads(l)["book"] for s in ("test", "val", "train") for l in open(f"{REPO}/data/items_{s}.jsonl")})
        for k in range(6):
            addraw(f"entail_{k}", 2, ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "131072",
                                      "--", "python3", "-u", f"{REPO}/chiron/gen_entail.py", "--books", *books[k::6], "--workers", "128"], -3)
    elif what == "legsum":
        # the Llama notes' summary step with gpt-oss: legsum VARIANT SOURCE [WORDS]
        variant, source, words = sys.argv[2], sys.argv[3], (sys.argv[4] if len(sys.argv) > 4 else "0")
        books = sorted({json.loads(l)["book"] for s in ("test", "val", "train") for l in open(f"{REPO}/data/items_{s}.jsonl")})
        for k in range(2):
            addraw(f"sheet_{variant}_{k}", 2, ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "131072",
                                               "--", "python3", "-u", f"{REPO}/chiron/gen_legsum.py", "--source", source, "--variant", variant,
                                               "--words", words, "--books", *books[k::2]], -3)
    elif what == "charsynth":
        # charmem's exact principal synthesis on other notes: charsynth VARIANT SOURCE
        variant, source, words = sys.argv[2], sys.argv[3], (sys.argv[4] if len(sys.argv) > 4 else "900")
        books = sorted({json.loads(l)["book"] for s in ("test", "val", "train") for l in open(f"{REPO}/data/items_{s}.jsonl")})
        for k in range(3):
            addraw(f"sheet_{variant}_{k}", 2, ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "131072",
                                               "--", "python3", "-u", f"{REPO}/chiron/gen_charsynth.py", "--source", source, "--variant", variant,
                                               "--words", words, "--books", *books[k::3]], -3)
    elif what == "evalsmall":
        # Qwen3.5-9B base and Qwen3-4B only: evalsmall TAG LONG COND...
        tag, lng, conds = sys.argv[2], sys.argv[3] == "1", sys.argv[4:]
        for split in ("test", "val", "train"):
            addraw(f"es_{tag}_{split}_q9b", 1, env(CHIRON_BASE=1) + srv("qwen9base", lng) + ev("eval_mcp.py", split, f"items_{split}", conds, extra=["--rotations"]), -2)
            addraw(f"es_{tag}_{split}_q4b", 1, srv("qwen4b", lng) + ev("eval_mcp.py", split, f"items_{split}", conds), -2)
    elif what == "merge":
        # consensus of sheet samples: merge VARIANT WORDS V1 V2 V3...
        variant, words, src = sys.argv[2], sys.argv[3], sys.argv[4:]
        books = sorted({json.loads(l)["book"] for s in ("test", "val", "train") for l in open(f"{REPO}/data/items_{s}.jsonl")})
        for k in range(2):
            addraw(f"sheet_{variant}_{k}", 2, ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "65536",
                                               "--", "python3", "-u", f"{REPO}/chiron/gen_merge.py", "--variant", variant, "--words", words,
                                               "--from", *src, "--books", *books[k::2]], -3)
    elif what == "eval27":
        tag, lng, conds = sys.argv[2], sys.argv[3] == "1", sys.argv[4:]
        for split in ("test", "val", "train"):
            n = 2 if split == "train" else 1
            for k in range(n):
                addraw(f"e27_{tag}_{split}_s{k}of{n}", 2 if lng else 1, env(CHIRON_THINKING=0) + srv("qwen27", lng) + ev("eval_mcp.py", split, f"items_{split}", conds, k, n, ["--rotations"]), -3)
    elif what == "eval_conds":
        # evaluate the given conditions on every model: queue_jobs.py eval_conds <tag> <long 0|1> COND...
        tag, lng, conds = sys.argv[2], sys.argv[3] == "1", sys.argv[4:]
        for split in ("test", "val", "train"):
            n = 4 if split == "train" else 1
            for k in range(n):
                addraw(f"ec_{tag}_{split}_q27_s{k}of{n}", 2 if lng else 1, env(CHIRON_THINKING=0) + srv("qwen27", lng) + ev("eval_mcp.py", split, f"items_{split}", conds, k, n, ["--rotations"]), -1)
                addraw(f"ec_{tag}_{split}_q27think_s{k}of{n}", 2 if lng else 1, env(CHIRON_THINKING=1) + srv("qwen27", lng) + ev("eval_reason.py", split, f"items_{split}", conds, k, n), 0)
            addraw(f"ec_{tag}_{split}_q9b", 1, env(CHIRON_BASE=1) + srv("qwen9base", lng) + ev("eval_mcp.py", split, f"items_{split}", conds, extra=["--rotations"]), -1)
            addraw(f"ec_{tag}_{split}_q4b", 1, srv("qwen4b", lng) + ev("eval_mcp.py", split, f"items_{split}", conds), -1)
    elif what == "plot":
        # plot summaries with gpt-oss (one pass at 131k context on 2 GPUs; chapter-by-chapter at 65k on 1 GPU)
        books = sorted({b for s in ("test", "val", "train") for b in (json.loads(l)["book"] for l in open(f"{REPO}/data/items_{s}.jsonl"))})
        groups = [books[k::6] for k in range(6)]
        for k, g in enumerate(groups):
            addraw(f"plot_global_{k}", 2, ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "131072",
                                           "--", "python3", "-u", f"{REPO}/chiron/gen_plot.py", "global", "--books", *g, "--workers", str(len(g))], 0)
            addraw(f"plot_hier_{k}", 1, ["python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "gptoss", "--max-model-len", "65536",
                                         "--", "python3", "-u", f"{REPO}/chiron/gen_plot.py", "hier", "--books", *g, "--workers", str(len(g))], 0)
    elif what == "bookch_think_rs":
        # thinking-on short book-text conditions, one condition per job and split wider (finished work is skipped)
        for split in ("test", "val", "train"):
            n = {"train": 6, "val": 1, "test": 2}[split]
            for c in ("book_ch1", "book_ch2", "book_prefix", "book_ch1p"):
                for k in range(n):
                    addraw(f"bookchrs_{split}_{c}_s{k}of{n}", 1, env(CHIRON_THINKING=1) + srv("qwen27", False) + ev("eval_reason.py", split, f"items_{split}", c, k, n), 1)
    elif what == "oracle_shards":
        # thinking-on prior-facts oracle (and its name-swapped control) on train, 8 ways each
        for c in ("oracle_prior", "swapname:oracle_prior"):
            for k in range(8):
                addraw(f"or_rs_train_{tag(c)}_s{k}of8", 1, env(CHIRON_THINKING=1) + srv("qwen27", False) + ev("eval_reason.py", "train", "items_train", c, k, 8), 1)
    elif what in ("gender", "gender_think"):
        # "Gender: female/male." as the only character information (gender_think: only the thinking-on shards)
        for split in ("test", "val", "train"):
            if what == "gender":
                for stem in (f"items_{split}", f"items_{split}_short", f"items_{split}_window"):
                    addraw(f"g_q27_{stem[6:]}", 1, env(CHIRON_THINKING=0) + srv("qwen27", False) + ev("eval_mcp.py", split, stem, "gender", extra=["--rotations"]), 0)
                addraw(f"g_q9b_{split}", 1, env(CHIRON_BASE=1) + srv("qwen9base", False) + ev("eval_mcp.py", split, f"items_{split}", "gender", extra=["--rotations"]), 0)
                addraw(f"g_q4b_{split}", 1, srv("qwen4b", False) + ev("eval_mcp.py", split, f"items_{split}", "gender"), 0)
            n = {"train": 24, "val": 3, "test": 6}[split]
            for k in range(n):
                addraw(f"g_q27think_{split}_s{k}of{n}", 1, env(CHIRON_THINKING=1) + srv("qwen27", False) + ev("eval_reason.py", split, f"items_{split}", "gender", k, n), 1)
    elif what == "reason_smoke":
        addraw("q27think_smoke", 1, ["env", "CHIRON_THINKING=1", "python3", "-u", f"{REPO}/scripts/serve_and_run.py", "--model", "qwen27",
            "--max-model-len", "65536", "--", "python3", "-u", f"{REPO}/chiron/eval_reason.py", "--split", "test", "--items", "items_test",
            "--conditions", "v2", "noinfo", "--limit", "12", "--tag", "_smoke"], -10)
    elif what == "eval":
        name, maxlen, model, stem, conds = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
        add(f"eval_{name}", 1, ["--model", model, "--max-model-len", maxlen],
            [f"{REPO}/chiron/eval_mcp.py", "--split", "test", "--items", stem, "--conditions", *conds, "--workers", "128"], -1)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
