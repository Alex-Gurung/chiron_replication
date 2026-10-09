"""What does each character sheet add inside the NCP task's own prompt? The NCP writer ruler with only
example["character_sheets"] swapped.

Follows ncp_eval/v70_score_writer.py (the diversity project's canonical NCP likelihood scorer, q4v2 / q4v3 lineage) step
for step; its helpers are imported from the frozen code copy (/home/toolkit/ncp_q4_v2_20260908/code), not rewritten:
  messages  v16_writing_messages(example): system = the planner role; user = "# Story information" (summary of already
            written chapters, character sheets, previous 2 chapters, this chapter so far) + "# The section to plan"
            (header, synopsis, passage summary, length) + "Write the passage itself ..."
  target    example["next_chapter"] as a partial assistant turn after "<answer>\\n" (tokenized_fake_answer_inputs)
  score     mean natural-log probability of the target's tokens (score_summary chapter_mean_logprob, nats per token)
  engine    in-process vLLM, Qwen/Qwen3-4B-Instruct-2507, max_model_len 57344, temperature 0, max_tokens 1,
            prompt_logprobs=1, prefix cache not read; a section whose sequence reaches the window in any condition is
            skipped for all of them.
Sections: /home/toolkit/ncp_q4_v2_20260908/cohorts/{test,val,train}_examples.jsonl (test = the canonical 1,457), those
whose three principals have every sheet variant: 3,703 in 29 books, 664 of them test sections (dark, mercy, witch).
Conditions (only character_sheets changes; key order and the "Supporting cast" entry stay as shipped):
  v2       the example as shipped: the canonical no-plan control (reproduces the stored control_mean_logprob)
  none     character_sheets=None, the block is dropped;  cast: only "Supporting cast" is kept
  <sheet>  the three principals' sheets replaced by that representation (SHEETS)

  python3 chiron/ncp_ruler_sheets.py --prepare           data/ncp_ruler_sheets.jsonl: the sheets per (book, chapter)
  PYTHONPATH=/home/toolkit/ncp_q4_v2_20260908/code:/home/toolkit/diversity/diversity \\
    /home/toolkit/eaiexp/env3/bin/python chiron/ncp_ruler_sheets.py [--shard K --nshards N] [--dry]
Output: outputs/ncp_ruler/<model>/scores.s<K>of<N>.jsonl {example_id, split, book, scores: {condition: mean logprob},
target_tokens, prompt_tokens}. Resumable. --dry tokenizes the first sections and prints lengths (no GPU).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, OUT, append_jsonl, load_reps, read_jsonl  # noqa: E402

COHORTS = Path("/home/toolkit/ncp_q4_v2_20260908/cohorts")
SHEETS = ["legacy", "summary", "sheet_csum_flat", "charmem", "sheet_legsum_x_500_flat"]
CAST = "Supporting cast"
INPUTS = DATA / "ncp_ruler_sheets.jsonl"


def prepare():
    n = 0
    with open(INPUTS, "w") as f:
        for split in ("test", "val", "train"):
            reps = load_reps(split)
            seen = set()
            for line in open(COHORTS / f"{split}_examples.jsonl"):
                r = json.loads(line)
                k = (r["story_id"], r["chapter_index"])
                if k in seen:
                    continue
                seen.add(k)
                names = [x for x in r["character_sheets"] if x != CAST]
                sheets = {s: {x: reps.get((s, *k, x)) for x in names} for s in SHEETS}
                if all(v is not None for d in sheets.values() for v in d.values()):
                    f.write(json.dumps({"book": k[0], "chapter_index": k[1], "sheets": sheets}, ensure_ascii=False) + "\n")
                    n += 1
    print(n, "chapters with every sheet variant ->", INPUTS)


def variants(example, sheets):
    """condition -> the example with only character_sheets changed."""
    shipped = example["character_sheets"]
    out = {"v2": example, "none": {**example, "character_sheets": None},
           "cast": {**example, "character_sheets": {k: v for k, v in shipped.items() if k == CAST}}}
    for s in SHEETS:
        out[s] = {**example, "character_sheets": {k: (v if k == CAST else sheets[s][k]) for k, v in shipped.items()}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-model-len", type=int, default=57344)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.76)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    if args.prepare:
        return prepare()

    from ncp_eval.data import example_id, load_examples
    from ncp_eval.prompts import v16_writing_messages
    from ncp_eval.protocol import MODEL
    from ncp_eval.score import score_summary, selected_logprobs
    from ncp_eval.score_fake_answers import tokenized_fake_answer_inputs
    from ncp_eval.score_plan_presentations import ANSWER_OPEN
    from transformers import AutoTokenizer

    model = args.model or MODEL
    sheets = {(r["book"], r["chapter_index"]): r["sheets"] for r in read_jsonl(INPUTS)}
    examples = sorted(((example_id(r), split, r) for split in ("test", "val", "train") for r in load_examples(COHORTS / f"{split}_examples.jsonl")
                       if (r["story_id"], r["chapter_index"]) in sheets), key=lambda x: x[0])[args.shard::args.nshards]
    root = OUT / "ncp_ruler" / model.split("/")[-1]
    out = root / f"scores.s{args.shard}of{args.nshards}.jsonl"
    done = {r["example_id"] for f in root.glob("scores.*.jsonl") for r in read_jsonl(f)}
    todo = [x for x in examples if x[0] not in done]
    print(len(examples), "sections,", len(todo), "to score", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = None
    if todo and not args.dry:
        from vllm import LLM
        llm = LLM(model=model, tensor_parallel_size=1, max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization)

    for index, (identifier, split, example) in enumerate(todo):
        conditions = {label: {"messages": v16_writing_messages(ex), "prefix": ANSWER_OPEN}
                      for label, ex in variants(example, sheets[(example["story_id"], example["chapter_index"])]).items()}
        inputs = tokenized_fake_answer_inputs(tokenizer, conditions, example["next_chapter"])
        lengths = {l: len(item["prompt_token_ids"]) for l, item in inputs.items()}
        if args.dry:
            item = inputs["v2"]
            print(identifier, "v2 sequence", lengths["v2"], "target tokens", item["target_end"] - item["target_start"], lengths, flush=True)
            if index == 2:
                return
            continue
        rec = {"example_id": identifier, "split": split, "book": example["story_id"]}
        if any(n >= args.max_model_len for n in lengths.values()):       # as v70_score_writer: the section drops from all arms
            append_jsonl(out, {**rec, "skipped": "over_max_model_len"})
            print(f"{index + 1}/{len(todo)} {identifier} SKIPPED over_max_model_len", flush=True)
            continue
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt
        labels = list(conditions)
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, skip_reading_prefix_cache=True)
        prompts = [TokensPrompt(prompt_token_ids=inputs[l]["prompt_token_ids"]) for l in labels]
        outputs = []
        for start in range(0, len(prompts), args.batch_size):
            outputs.extend(llm.generate(prompts[start:start + args.batch_size], params, use_tqdm=False))
        scores = {}
        for label, output in zip(labels, outputs):
            item = inputs[label]
            values = selected_logprobs(output, item["prompt_token_ids"], item["target_start"], item["target_end"])
            scores[label] = score_summary(values, len(values))["chapter_mean_logprob"]
        append_jsonl(out, {**rec, "scores": scores, "target_tokens": inputs["v2"]["target_end"] - inputs["v2"]["target_start"],
                           "prompt_tokens": lengths})
        print(f"{index + 1}/{len(todo)} {identifier}", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
