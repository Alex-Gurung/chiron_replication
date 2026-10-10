"""The NCP writer ruler with the plot-summary slot and the character-sheet slot swapped: what does character information
add to predicting the true next section, and which representation of it adds most?

Scoring is ncp_ruler_sheets.py's (the diversity project's v70_score_writer, helpers imported from its frozen code): the
v16 writing prompt, the real next section as a partial assistant turn after "<answer>\\n", mean log-probability per target
token, in-process vLLM, window 57,344, a section that reaches the window in any condition is skipped for all.
A condition is "<plot>|<characters>", applied to the shipped example and nothing else:
  plot        ship (the dataset's chapter synopses, as shipped) | none (the block is dropped) | a plot-summary condition
              from data/plot_<split>.jsonl (plot_global_1000, plot_q4b_global_1000, plot_l70_global_1000, ...)
  characters  v2 (as shipped) | none (block dropped) | cast (only "Supporting cast") | a sheet condition from the reps
              (the three principals' sheets are replaced; "Supporting cast" and the key order stay as shipped)
"ship|v2" is the canonical no-plan control. Judges: Qwen/Qwen3-4B-Instruct-2507 (canonical; tokenized and cross-checked by
tokenized_fake_answer_inputs) or Qwen/Qwen3.8-27B (a second judge: the same prompt and target, chat template with
thinking off, tokenized by tokenized_fake_answer_from_generation_prefix).

  python3 chiron/ncp_ruler_grid.py --prepare             data/ncp_grid_inputs.jsonl: sheets and plot summaries per chapter
  PYTHONPATH=/home/toolkit/ncp_q4_v2_20260908/code:/home/toolkit/diversity/diversity /home/toolkit/eaiexp/env3/bin/python \\
    chiron/ncp_ruler_grid.py --tag T --conds "ship|v2" "none|none" ... [--every N] [--shard K --nshards M] [--model M] [--dry]
--every N keeps every Nth section of each book (sorted by id). A section needs every requested condition.
Output: outputs/ncp_grid/<model>/<T>.s<K>of<M>.jsonl {example_id, split, book, scores: {condition: mean logprob},
target_tokens, prompt_tokens}. Resumable per tag.
"""
import argparse
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, OUT, append_jsonl, load_reps, read_jsonl  # noqa: E402

COHORTS = Path("/home/toolkit/ncp_q4_v2_20260908/cohorts")
CAST = "Supporting cast"
INPUTS = DATA / "ncp_grid_inputs.jsonl"
SHEETS = ["legacy", "charmem", "summary", "sheet_csum_flat", "sheet_legsum_x_500_flat",                       # Llama CHIRON condensed; gpt-oss
          "sheet_legsum_q4b_x_flat", "summary_q4b", "sheet_csum_q4b_flat", "summary_l70", "sheet_csum_l70_flat"]   # Qwen3-4B; Llama summaries
PLOTS = [f"plot{g}_{kind}_{t}" for g in ("", "_q4b", "_l70") for kind in ("global", "hier") for t in (1000, 2000)]


def prepare():
    n = 0
    tmp = INPUTS.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for split in ("test", "val", "train"):
            reps = load_reps(split)
            plots = {(r["condition"], r["book"], r["boundary"]): r["text"] for r in read_jsonl(DATA / f"plot_{split}.jsonl")}
            seen = set()
            for line in open(COHORTS / f"{split}_examples.jsonl"):
                r = json.loads(line)
                k = (r["story_id"], r["chapter_index"])
                if k in seen:
                    continue
                seen.add(k)
                names = [x for x in r["character_sheets"] if x != CAST]
                sheets = {s: {x: reps.get((s, *k, x)) for x in names} for s in SHEETS}
                sheets = {s: d for s, d in sheets.items() if all(v is not None for v in d.values())}
                if "legacy" in sheets:
                    f.write(json.dumps({"book": k[0], "chapter_index": k[1], "sheets": sheets,
                                        "plots": {p: plots[(p, *k)] for p in PLOTS if (p, *k) in plots}}, ensure_ascii=False) + "\n")
                    n += 1
    os.replace(tmp, INPUTS)
    cov = collections.Counter(c for r in read_jsonl(INPUTS) for c in [*r["sheets"], *r["plots"]])
    print(n, "chapters ->", INPUTS, dict(cov))


def variant(example, spec, inp):
    """The shipped example with the plot and character slots set as the condition says, or None when an input is missing."""
    plot, chars = spec.split("|")
    ex = dict(example)
    if plot == "none":
        ex["prior_plot_summary"] = None
    elif plot != "ship":
        if plot not in inp["plots"]:
            return None
        ex["prior_plot_summary"] = inp["plots"][plot]
    shipped = example["character_sheets"]
    if chars == "none":
        ex["character_sheets"] = None
    elif chars == "cast":
        ex["character_sheets"] = {k: v for k, v in shipped.items() if k == CAST}
    elif chars != "v2":
        if chars not in inp["sheets"]:
            return None
        ex["character_sheets"] = {k: (v if k == CAST else inp["sheets"][chars][k]) for k, v in shipped.items()}
    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--tag", default="grid")
    ap.add_argument("--conds", nargs="+", default=["ship|v2", "none|none"])
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--max-model-len", type=int, default=57344)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.76)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    if args.prepare:
        return prepare()

    from ncp_eval.data import example_id, load_examples
    from ncp_eval.prompts import v16_writing_messages
    from ncp_eval.score import score_summary, selected_logprobs
    from ncp_eval.score_fake_answers import input_ids, tokenized_fake_answer_from_generation_prefix, tokenized_fake_answer_inputs
    from ncp_eval.score_plan_presentations import ANSWER_OPEN
    from transformers import AutoTokenizer

    canonical = args.model == "Qwen/Qwen3-4B-Instruct-2507"
    inputs_by = {(r["book"], r["chapter_index"]): r for r in read_jsonl(INPUTS)}
    per_book = collections.defaultdict(list)
    for split in ("test", "val", "train"):
        for r in load_examples(COHORTS / f"{split}_examples.jsonl"):
            if (r["story_id"], r["chapter_index"]) in inputs_by:
                per_book[r["story_id"]].append((example_id(r), split, r))
    examples = sorted(x for v in per_book.values() for x in sorted(v, key=lambda x: x[0])[::args.every])[args.shard::args.nshards]
    root = OUT / "ncp_grid" / args.model.split("/")[-1]
    out = root / f"{args.tag}.s{args.shard}of{args.nshards}.jsonl"
    done = {r["example_id"] for f in root.glob(f"{args.tag}.s*.jsonl") for r in read_jsonl(f)}
    todo = [x for x in examples if x[0] not in done]
    print(len(examples), "sections,", len(todo), "to score,", len(args.conds), "conditions", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = None
    if todo and not args.dry:
        from vllm import LLM
        llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization)

    def tokenize(conditions, target):
        if canonical:
            return tokenized_fake_answer_inputs(tokenizer, conditions, target)
        return {label: tokenized_fake_answer_from_generation_prefix(                      # second judge: thinking off
                    tokenizer, input_ids(tokenizer.apply_chat_template(c["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False)),
                    c["prefix"], target) for label, c in conditions.items()}

    for index, (identifier, split, example) in enumerate(todo):
        inp = inputs_by[(example["story_id"], example["chapter_index"])]
        exs = {spec: variant(example, spec, inp) for spec in args.conds}
        rec = {"example_id": identifier, "split": split, "book": example["story_id"]}
        if any(v is None for v in exs.values()):
            append_jsonl(out, {**rec, "skipped": "missing_input", "missing": [s for s, v in exs.items() if v is None]})
            continue
        conditions = {spec: {"messages": v16_writing_messages(ex), "prefix": ANSWER_OPEN} for spec, ex in exs.items()}
        inputs = tokenize(conditions, example["next_chapter"])
        lengths = {l: len(item["prompt_token_ids"]) for l, item in inputs.items()}
        if args.dry:
            first = inputs[args.conds[0]]
            print(identifier, "target tokens", first["target_end"] - first["target_start"], lengths, flush=True)
            if index == 2:
                return
            continue
        if any(n >= args.max_model_len for n in lengths.values()):       # as v70_score_writer: the section drops from all arms
            append_jsonl(out, {**rec, "skipped": "over_max_model_len"})
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
        first = inputs[labels[0]]
        append_jsonl(out, {**rec, "scores": scores, "target_tokens": first["target_end"] - first["target_start"], "prompt_tokens": lengths})
        if (index + 1) % 50 == 0:
            print(f"{index + 1}/{len(todo)} {identifier}", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
