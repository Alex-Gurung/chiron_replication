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

### 06:00 UTC: which sheet sections matter

- Qwen3.5-9B base, one v2/charmem section at a time (~100-160 words each): relationships alone 53.5 / 53.7 (full
  sheet 54.5 / 56.5, names only 47.6), then history and goals (~51-53), physical, dialogue and knowledge weakest
  (~48-51). Everything except relationships (~600 words) = 53.8 / 55.6. The ~120-word relationships section carries
  about as much as the other five sections combined. Consistent with the interaction finding for legacy.
- 27B section results pending (train shard). Reruns: two short-span reasoning shards (2-hour request timeout under
  the old code) and 27B train perplexity with story (over-length prompt) requeued.
- Within one representation, a character's sheet length does not predict its gain over names only (27B; v2 quartiles
  +16.4/+16.6/+12.8/+20.2; charmem +19.3/+17.1/+16.9/+17.6; legacy flat too). Sheets with an empty section are not
  worse (small n). Consistent with "is the needed fact there", not "how much is there".

### 07:00-07:45 UTC: gender, sheet sections on 27B, short-span reasoning

- Sheet sections on the 27B (thinking off): each ~120-word section alone adds only 3 to 6 points over names only
  (relationships 44.9 / 46.4 for v2 / charmem), and the sheet without relationships (54.8 / 56.5) is close to the full
  sheet (56.2 / 58.4). On 9B base relationships alone nearly matched the full sheet, so which part of a sheet matters
  depends on the model; the 27B without thinking seems to add up many weak cues.
- Gender does much of the work. Principals' genders read from the pronouns in their v2 + charmem sheets (28% of
  (passage, character) pairs have a gender unique among the three principals, 56% share it with one other, 17% all
  three the same). 27B thinking, names only: unique 89%, shared 58%, all same 35% (chance 33%): names-only reasoning is
  mostly gender inference from the names. With sheets: unique 97-98%, shared 94-96%, all same 86% (v2) to 94% (legacy
  full). The representations only compete on same-gender characters. Without thinking the 27B barely infers gender from
  names (unique 41%), so part of every sheet's gain there may just be learning the gender. A "Gender: female." only
  representation is queued on all models to measure this.
- Short spans with thinking separate the representations (partial: v2 and swapname_v2 train shards running): names
  only 51.3, v2 74.2 (9 books so far), summary 78.3, charmem 77.8, CHIRON-style 79.6, book 8k 80.6, legacy full 82.1,
  name-swapped v2 19.5. Dense windows with thinking are at ceiling (97-99; names only 70.7).
- Rolling summaries drift onto another principal in two long books: from the middle of refuse, Sol's summary describes
  Rainy ("Little John", the boat Flower); late in when_moon, Kaan's describes Raeve. 1.3% of summaries have the wrong
  gender's pronouns (when_moon 19, refuse 12, husbands 2); v2 0%, charmem 0.1%.
- Ops: the 4-way thinking-on short/window shards were running at ~1,350 records/hour. They were re-split 12-16 ways; the
  eval scripts now treat any earlier output file of a condition as done, whatever its sharding.

### 07:35 UTC: gender only

- "Gender: female." / "Gender: male." as the only information, thinking off (macro, sections): Qwen3-4B 39.3 (names
  only 40.4), Qwen3.5-9B base 46.1 (47.6), Qwen3.8-27B 40.2 (40.1); 27B short spans 34.6 (35.1), dense windows 44.6
  (42.3). Even for characters whose gender is unique among the three principals, the 27B only goes 41 -> 46%. Direct
  answering does not make the pronoun inference ("she" -> the only woman); with thinking, names alone already get 89% of
  those characters. So the direct-scoring gains of the sheets come from their content, while with thinking much of the
  names-only score and of every sheet's score is gender. Thinking-on gender-only is queued behind the reasoning shards.
- Reasoning length (27B thinking, median characters per generation) orders the representations like accuracy and
  still separates them at ceiling. Sections: names only 76k, v2 17.6k, charmem 14.1k, summary 12.7k, CHIRON-style
  12.3k, book 8k 10.4k, legacy full 8.3k, whole book 5.9k, passage oracle 4.3k. Dense windows (all 97-99% accurate):
  v2 12.6k vs legacy full 7.8k.
- 27B chat perplexity complete: without story, legacy full / CHIRON-style -12.7%, summary -9.7%, charmem -9.2%, v2
  -8.2%; with the 4k words before the passage, CHIRON-style -3.4%, legacy full -3.3% (partial, over-length prompts
  skipped), charmem -2.2%, v2 -1.7%.
