"""Finish the stopped Sep 8 gpt-oss charmem rebuild (verify remaining ledgers, write all sheets).

Same prompts, model, roots and stage code as eaiexp/runners/ncp_finish_20260908.py; this harness
only changes what happens when a verification batch fails validation 3 times. In the Sep 8 run
that aborted 24 of 30 books (missing decisions, verdict disagreeing with flags, duplicate ids).
Here: (1) on the last attempt the verdict is derived from the four flags and unknown/duplicate
ids are dropped; (2) a batch that still fails is re-run one candidate per request; (3) a
candidate that still fails is removed with reason "verification_failed" and counted in the
ledger's source_verification.fallback. Books run concurrently; higher in-flight concurrency.
Writes into /home/toolkit/ncp_charmem_gptoss120b_reviewed_20260908 (never ncp_cohorts).
"""
import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

os.environ["CHARMEM_API_BASE"] = os.environ.get("CHIRON_API_BASE", os.environ.get("CHARMEM_API_BASE", ""))
os.environ.setdefault("CHARMEM_ROOT", "/home/toolkit/ncp_charmem_gptoss120b_reviewed_20260908")
OPS = Path("/home/toolkit/eaiexp")
CODE = OPS / "runners/charmem_gptoss_medium_20260908"
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(OPS / "runners"))
import common as C                                   # noqa: E402
import verify_gptoss_candidates_20260908 as V        # noqa: E402

RAW = Path("/home/toolkit/ncp_charmem_gptoss120b_medium_20260908")
REVIEW = Path(os.environ["CHARMEM_ROOT"])
HARNESS = {"file": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def repair(payload, error):
    items = payload.get("decisions") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    seen, kept = set(), []
    for d in items:
        if isinstance(d, dict) and d.get("id") not in seen and all(type(d.get(f)) is bool for f in V.FLAGS):
            d["verdict"] = "keep" if all(d[f] for f in V.FLAGS) else "remove"
            seen.add(d["id"])
            kept.append(d)
    payload["decisions"] = kept
    return "verdict_from_flags_dedup"


def request(chapter, chunk, rid, out):
    return {"request_id": rid, "messages": V.messages(chapter, chunk), "validator": lambda p, c=chunk: V.validate(p, c),
            "max_tokens": 12000, "out": out / "verification", "tag": rid.split("__verify")[0], "repair": repair}


def verify_chapter(chapter, items, prefix, width):
    batches = [(f"{prefix}__verify_{o:04d}", items[o:o + 8]) for o in range(0, len(items), 8)]
    results, failures = C.run_parallel([request(chapter, ch, rid, REVIEW) for rid, ch in batches], width, label=prefix)
    decisions, fallback = [], {"batches_split": 0, "removed_unverified": 0}
    for rid, chunk in batches:
        if rid in results:
            decisions += results[rid]["payload"]["decisions"]
            continue
        fallback["batches_split"] += 1
        singles = [(f"{rid}_s{i}", [c]) for i, c in enumerate(chunk)]
        res, fail = C.run_parallel([request(chapter, c, r, REVIEW) for r, c in singles], width, label=rid)
        for r, c in singles:
            if r in res:
                decisions += res[r]["payload"]["decisions"]
            else:
                fallback["removed_unverified"] += 1
                decisions.append({"id": c[0]["id"], "actual_referents": ["unresolved"], "supported": False,
                                  "assigned_to_correct_character": False, "appropriately_scoped": False,
                                  "useful": False, "verdict": "remove", "reason": "verification_failed"})
    V.validate({"decisions": decisions}, items)
    return decisions, fallback


def verify_book(book, width):
    bounds = C.load_boundaries()
    expected = [RAW / "ledgers" / f"{book}__chapter_{i:04d}.json" for i in range(max(bounds[book]))]
    assert all(p.exists() for p in expected), f"missing raw ledgers: {book}"
    chapters = {c["chapter_id"]: c for c in C.load_chapters()[book]}

    def one(p):
        ledger = json.loads(p.read_text())
        target = REVIEW / "ledgers" / p.name
        if target.exists():
            v = json.loads(target.read_text())["source_verification"]
            assert v["input_ledger_sha256"] == C.csha(ledger) and v["model"] == C.MODEL
            return
        decisions, fallback = verify_chapter(chapters[ledger["chapter_id"]], V.candidates(ledger), ledger["chapter_id"], 4)
        result = V.filter_ledger(ledger, decisions)
        result["source_verification"].update(fallback=fallback, harness=HARNESS)
        C.write_json(target, result)
        print("VERIFIED", ledger["chapter_id"], result["source_verification"]["kept"], "/",
              result["source_verification"]["candidates"], fallback, flush=True)

    with ThreadPoolExecutor(max_workers=width) as pool:
        list(pool.map(one, expected))
    C.write_json(REVIEW / "verified_books" / f"{book}.json", {"book": book, "chapters": len(expected), "harness": HARNESS,
        "source_ledger_sha256": {p.name: C.csha(json.loads(p.read_text())) for p in expected}})


def strong_repair(request):
    """Pinned last-attempt repair, then targeted fixes driven by the validator's own error, until it passes:
    repeated ids -> dedupe; over the word cap -> drop the last item of the longest section/character;
    invalid, duplicate, principal or note-less supporting names -> drop that entry."""
    import re
    base, validate = request["repair"], request["validator"]

    def repair(payload, error):
        info = base(payload, error) or {}
        info["extra"] = []
        for _ in range(200):
            try:
                validate(payload)
                return info
            except (KeyError, TypeError, ValueError) as e:
                msg = str(e)
            groups = list((payload.get("sections") or {}).values()) + [c.get("notes", []) for c in payload.get("characters") or []]
            if "repeated source ID" in msg:
                for items in groups:
                    for it in items:
                        it["source_ids"] = list(dict.fromkeys(it.get("source_ids", [])))
            elif re.search(r"has \d+ words; maximum", msg):
                longest = max(groups, key=len)
                if not longest:
                    return info
                longest.pop()
            elif "characters" in payload and re.search(r"name|principal|has no notes", msg):
                bad = next((c for c in payload["characters"] if repr(c.get("character")) in msg or f"{c.get('character')} " in msg), None)
                seen, keep = set(), []
                for c in payload["characters"]:
                    k = str(c.get("character", "")).casefold()
                    if c is not bad and k not in seen and c.get("notes"):
                        seen.add(k)
                        keep.append(c)
                if len(keep) == len(payload["characters"]):
                    return info
                payload["characters"] = keep
            else:
                return info
            info["extra"].append(msg[:120])
        return info
    return repair


def synthesize_book(book, width):
    import synthesize as S
    chapters, bounds, labels = C.load_chapters(), C.load_boundaries(), C.load_labels()
    for i in range(max(bounds[book])):
        ledger = json.loads((REVIEW / "ledgers" / f"{book}__chapter_{i:04d}.json").read_text())
        assert ledger.get("source_verification", {}).get("model") == C.MODEL
    streams, plan = S.plan_book(book, chapters, bounds, labels, REVIEW / "synth")
    requests = {r["request_id"]: r for stream in streams for r in stream}
    owners = {rid: boundary for boundary, items in plan.items() for rid, ids in items.values()}
    principals = chapters[book][0]["main_character_labels"]

    def done(rid, results):
        if S.write_boundary(book, owners[rid], plan[owners[rid]], results, principals, labels[book]):
            print("SHEET", book, owners[rid], flush=True)

    independent = [[requests[rid]] for items in plan.values() for rid, ids in items.values()]
    results, failures = S.run_streams(independent, width, label=f"synth {book}", on_done=done)
    if failures:                                     # second round: same prompt, fresh attempts, stronger repair
        retry = [[{**requests[rid], "repair": strong_repair(requests[rid])}] for rid in failures]
        more, failures = S.run_streams(retry, width, label=f"synth-retry {book}")
        results.update(more)
        for rid in more:
            done(rid, results)
    missing = [b for b in bounds[book] if not (REVIEW / "sheets" / f"{book}__{b:04d}.json").exists()]
    C.write_json(REVIEW / "completed_books" / f"{book}.json", {"book": book, "boundaries": len(bounds[book]),
                 "missing_boundaries": missing, "failed_requests": sorted(failures), "harness": HARNESS,
                 "canonical_cohort_changed": False})
    if missing:
        raise RuntimeError(f"{book}: {len(missing)} boundaries without sheets: {missing}")


def finish(book, width):
    try:
        verify_book(book, width)
        synthesize_book(book, width)
        print("COMPLETE", book, flush=True)
        return True
    except Exception as e:
        print("BOOK_FAILED", book, repr(e)[:1000], flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--width", type=int, default=8, help="concurrent chapters / synth requests per book")
    args = ap.parse_args()
    assert C.server_up(), "gpt-oss server not reachable"
    with ThreadPoolExecutor(len(args.books)) as pool:
        ok = list(pool.map(lambda b: finish(b, args.width), args.books))
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
