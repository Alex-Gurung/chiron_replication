# Findings log

Running record of results and decisions, newest last. Numbers are macro accuracy over the 21 books with at least
10 three-principal sections unless stated; "27B" is Qwen3.8-27B with thinking off.

## 2026-09-29

- Qwen3-4B barely uses any representation (+1 to +5 over names only, 40.4%). Qwen3.8-27B gains +14 to +23 on the
  same passages, positive in 21/21 books for every representation. Mistral-7B needs a forced "[CHAR " prefix
  (it never answers with a bare digit); with it, +8 to +10.
- A plain swap control is uninformative: sheets name their own subject. `swapname:` (names exchanged inside the
  swapped text) drops the 27B to 28.9% and thinking-on to 10%, so the models do read the sheets.
- Passage density matters: 27B v2 goes 39.9% (short spans, ~54 words) -> 56.2% (sections) -> 68.0% (dense
  windows, ~900 words). With thinking on, sections reach 94-97% with any representation.
- Why legacy (full Llama CHIRON-style sheet) and book text win (27B, sections):
  - not recency: removing the chapter right before the passage changes nothing (legacy 63.4 -> 63.4, CHIRON-style
    57.8 -> 57.5, book last 8k 60.4 -> 60.7);
  - not length: legacy cut to its most recent 6k words = 63.3 (full 14k = 63.4); still +5.5 over CHIRON-style at the
    same length;
  - not word overlap: a no-LLM TF-IDF matcher ranks legacy below CHIRON-style;
  - removing legacy's "not mentioned" filler helps: 64.7;
  - legacy mentions the other principals ~31 times per 1,000 words vs 13-20 for the other sheets. Direct test
    (statements naming another principal vs the rest) queued.
- Why the numbers look similar: item identity explains 62% of the variance in correctness, representation 0.3%.
  Representations get different items right: picking the best per passage would reach 86.7% vs 60.3% for the
  best single one; pairwise agreement 77-85%.

## 2026-09-30 (overnight, unattended)

Queued: Qwen3.5-9B-Base (base-model completion scoring), per-passage oracles (clues from the passage; verbatim
prior facts chosen for the passage) on all models, perplexity of the real passage with each representation under
9B base (raw text) and Qwen3-4B / 27B (chat: writing prompt as the user turn), saved reasoning traces, thinking-on
with the entire book, reasoning on short spans and dense windows.

### 01:00-02:00 UTC

- Ops: all 12 pods had died at 23:01 (more than 16 GiB of per-job /tmp compile caches); fixed in serve_and_run, relaunched.
  Jobs needing 2 GPUs starved behind 1-GPU jobs, so they were requeued on 1 GPU (27B at 131k context, gpt-oss at 65k).
  Short-span and dense-window reasoning (49 jobs) parked in state/queue/parked_chiron to let the rest through.
  9B-base perplexity OOMed (full-vocabulary prompt logprobs on 60k-token prompts), so it was requeued with
  --max-num-batched-tokens 2048 and --gpu-mem 0.85.
- Interaction ablation (partial): legacy statements naming another principal alone (~2-3.6k words) = 61.8% vs legacy
  full 63.4% and v2 56.2%; CHIRON-style interaction statements alone (~500 words) = 49.4%. The no-interaction halves
  are pending.
- Perplexity (Qwen3-4B chat, partial). Long sheets lower the loss on the real passage a lot: legacy full -0.39
  nats/token, CHIRON-style full -0.26 (21/21 books). The ~800-word sheets and summaries barely help without story
  context (v2 +0.02, charmem 0.00, summary -0.02) and help a little with it (v2 -0.07, charmem -0.08).

### 02:50 UTC

- Interaction ablation (27B): legacy statements naming another principal (~2-3.6k words) = 61.8%; the rest of legacy
  (~8.6-11k words) = 58.4% (17 books so far); CHIRON-style interaction statements (~500 words) = 49.4%, the rest
  (~5.3k) = 53.0%. Statements about how a character acts with or relates to the other principals are worth the most per
  word, and legacy has 4-7x more of them than CHIRON-style (whose strict entailment filter keeps self-contained claims).
- Oracles built for all 1,087 passages: prior facts median 4 per character (1% empty; 23% of proposals dropped as not
  verbatim); passage clues median 3. The chosen prior facts are often generic traits, so the prior oracle measures what
  the old notes can offer, as judged by gpt-oss.
- Queued combined representations (v2+charmem+summary; legacy without filler + v2; all four).

### 03:50 UTC: oracles

| model | names only | v2 | legacy full | passage oracle | prior-facts oracle | swapped prior oracle |
|---|---|---|---|---|---|---|
| Qwen3-4B | 40.4 | 43.1 | 45.3 | 59.6 | 40.9 | 33.6 |
| Qwen3.5-9B base | 47.6 | 54.5 | 60.1 (partial) | 72.7 | 51.0 | 35.5 |
| Qwen3.8-27B | 40.1 | 56.2 | 63.4 | 59.7 | 44.8 | 31.8 |
| Qwen3.8-27B thinking | 63.7 | 94.0 | 96.7 | 98.4 (partial) | 91.1 (partial) | 9.8 (partial) |

- Without thinking, the passage oracle (clues copied from the passage itself) reaches only 59.7% on the 27B, below
  legacy full. The ceiling of direct one-token answering is the model's matching ability, not information. With
  thinking the same clues give 98%.
- A handful of selected prior facts (median 4) is worse than a whole sheet without thinking (44.8 vs 56.2) but nearly as
  good with thinking (91 vs 94). Direct answering seems to need many weak cues; reasoning can use a few facts.
- Both modes use the facts: the name-swapped prior oracle drops to 31.8% (direct) and 9.8% (thinking).
- Note: 27B jobs requeued on one GPU run at 131k context; prompts longer than that (the longest legacy-full and
  combination prompts late in long books) are logged in .errors.jsonl and skipped.

### 04:00 UTC: reasoning traces (27B thinking on, test+val, partial)

- names only: 60% accuracy, median 5,300 words of reasoning, 17% of traces mention the notes. The model infers from
  the names themselves ("The Leszy" = the male demon) plus pronouns and passage events, so names-only is not zero
  information.
- v2: 89%, median 2,350 words, 47% of traces discuss the notes, and they match specific facts to events, for example
  "If CHAR 2 is Liska, 'Liska's aunt' matches character info: Liska mentions an aunt who lives in Ząbki";
  "Character info: Tonner asks Jessyn to attend a confidential emergency meeting at the lab".
- name-swapped v2: 5.6%, 90% of traces reason from the (misleading) notes.
- Only ~2% of reasoning 4-grams are copied from the notes vs ~13% from the passage: the model paraphrases the notes
  and quotes the passage.

### 05:00 UTC: combinations, 9B base, whole book, perplexity

- Combinations (27B direct): legacy without filler + v2 = 67.0% (best so far; legacy alone 64.7%); all four = 67.1%;
  v2 + charmem + summary = 56.2% (no better than v2 alone). Short sheets are redundant with each other; legacy adds.
- Qwen3.5-9B base (completion scoring): names only 47.6%, v2 54.5%, charmem 56.5%, summary 54.9%, CHIRON-style
  54.1%, legacy full 61.5%, passage oracle 72.7%; name-swapped v2 32.7%. Same ordering as the 27B.
- 27B thinking with the entire book: 96.9% (legacy full 96.7%, v2 94.0%).
- Perplexity of the real passage, loss change per token vs names only (all 21/21 books unless partial):
  - 9B base, no story: v2 -0.091, charmem -0.099, summary -0.103, CHIRON-style full -0.137, legacy full -0.157.
  - 9B base, with the 4k words before the passage: v2 -0.015, charmem -0.019, summary -0.017, CHIRON-style -0.025.
  - 27B chat, no story: v2 -0.086, charmem -0.096, summary -0.102, CHIRON-style -0.136.
  - 27B chat, with story: v2 -0.017, charmem -0.022, CHIRON-style 2k -0.022.
  Character notes cut perplexity by ~9-15% alone and ~1.5-3.5% on top of recent story text; longer notes help more;
  charmem > v2 consistently. The Qwen3-4B chat numbers are erratic (no gain from short sheets without story) and
  should not be leaned on.
