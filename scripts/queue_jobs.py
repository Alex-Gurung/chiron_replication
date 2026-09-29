"""Add chiron_replication jobs to the eaiexp file queue on lane "chiron".

  python3 scripts/queue_jobs.py smoke           one small job of each kind
  python3 scripts/queue_jobs.py chiron          CHIRON-style statements, all books (test books first)
  python3 scripts/queue_jobs.py summary         rolling character summaries, all books
  python3 scripts/queue_jobs.py charmem         finish the Sep 8 charmem rebuild, all books
  python3 scripts/queue_jobs.py eval NAME MAXLEN MODEL ITEMS COND...   one eval job (MODEL: qwen4b|mistral)
  python3 scripts/queue_jobs.py final SPLIT...  every condition for the given splits (needs items/reps built)
Lower priority number is claimed first. Job names are stamped so reruns never collide.
"""
import json
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
        for split in sys.argv[2:]:
            ev = lambda stem, conds: [f"{REPO}/chiron/eval_mcp.py", "--split", split, "--items", stem,
                                      "--conditions", *conds, "--workers", "128"]
            for c in short:
                add(f"final_{split}_{c.replace(':', '_').replace('@', 'at')}", 1, ["--model", "qwen4b", "--max-model-len", "65536"], ev(f"items_{split}", [c]), 1)
            for c in long:
                add(f"final_{split}_{c.replace(':', '_')}", 1, ["--model", "qwen4b", "--max-model-len", "262144"], ev(f"items_{split}", [c]), 1)
            add(f"final_{split}_pron", 1, ["--model", "qwen4b", "--max-model-len", "262144"], ev(f"items_{split}_pron", pron), 1)
            add(f"final_{split}_mistral", 1, ["--model", "mistral", "--max-model-len", "32768"], ev(f"items_{split}", mistral), 2)
            for c in two_short:
                add(f"final_{split}_two_{c.replace(':', '_').replace('@', 'at')}", 1, ["--model", "qwen4b", "--max-model-len", "65536"], ev(f"items_{split}_two", [c]), 2)
            for c in two_long:
                add(f"final_{split}_two_{c.replace(':', '_')}", 1, ["--model", "qwen4b", "--max-model-len", "262144"], ev(f"items_{split}_two", [c]), 2)
    elif what == "eval":
        name, maxlen, model, stem, conds = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
        add(f"eval_{name}", 1, ["--model", model, "--max-model-len", maxlen],
            [f"{REPO}/chiron/eval_mcp.py", "--split", "test", "--items", stem, "--conditions", *conds, "--workers", "128"], -1)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
