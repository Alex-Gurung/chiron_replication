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
- Oracle prior facts by source: 11,859 of the verbatim facts gpt-oss chose come from legacy (last 6k words), 516 from
  charmem (806 words). Weak evidence only: legacy was listed first, is 7x longer, and charmem's compound bullets are
  harder to quote verbatim.

### 09:00 UTC: near-final

- Short spans with thinking, now complete except name-swapped v2 on train: names only 51.3, v2 78.3 (the earlier 74.2
  was 9 books), charmem 77.9, summary 78.3, legacy 78.7, CHIRON-style 79.4, CHIRON-style 2k 79.5, book 8k 80.6, legacy
  full 82.1 (+3.7 over v2, 16/19 books, CI [+1.7, +5.8]); name-swapped v2 19.0 (partial). So short spans do not separate
  the representations much more than sections do; the legacy lead is the one robust difference, as without thinking.
- Prior-facts oracle with thinking, complete: 88.9 (v2 94.0, names only 63.7). Name-swapped version 14.1 (partial).
- 27B sheet sections, complete: relationships only 44.9 / 46.5 (v2 / charmem), other single sections 43.2-45.7,
  everything but relationships 55.0 / 56.3, full sheet 56.2 / 58.4.
- Ops: sec_qwen27_test's vLLM engine hung at 05:04 (0 tokens/s, 13 running, 98 waiting) and sat for almost 4 hours
  without failing; rerun as 7 jobs in 10 minutes. No other job was hung. scripts/final_analysis.sh regenerates every
  analysis file and the report.

### 11:20 UTC: final

- All runs done; every chiron worker pod released (w51/w52 at 10:20, seven more at 11:05, w45/w47 at 11:20).
- Thinking-on gender only (27B, sections, 21 books): 65.7 (names only 63.7, v2 94.0); unique 91.4, shared 55.7, all
  same 35.0. Names already carry gender, so stating it adds 2 points; the remaining ~28 points of a sheet are content.
- Name-swapped controls with thinking, final: prior-facts oracle 15.8, v2 on short spans 19.3 (46 of 4,113
  generations missing: a second vLLM engine hang, at 09:28, under 90% KV use).
- Two hangs out of ~150 27B jobs, both with the engine stuck at 0 tokens/s and requests waiting; the job neither fails
  nor exits. Worth a watchdog on "Avg generation throughput: 0.0" for 10+ minutes in future runs.

## 2026-09-30 afternoon (with the user)

- Length chart: now median prompt tokens with a bar over the middle half of prompts (was the mean). Full legacy is a
  median 0.67x the whole-book prompt and longer for 20% of passages (early in books). The 27B thinking-off whole-book
  condition had silently skipped the 311 passages whose prompts exceed 131k tokens (no error records): rerun at 262k on
  2 GPUs; 58.7% on 776 passages -> 58.4% on all 1,087 (per-character scoring). Combinations completed the same way.
- Scoring artifact: asked one character at a time, Qwen3.8-27B without thinking almost never answers 0 (names only:
  0.1% right when the answer is [CHAR 0]; quote oracle 5%). Per-character argmax therefore caps it near 2/3. Joint
  scoring (best one-to-one mapping of the three answers per passage and block order; the prompt says ids are unique)
  is now the primary thinking-off metric (analyze.py row["joint"]; report). 27B, per character -> joint: names only
  40.1 -> 48.2, v2 56.2 -> 68.8, charmem 58.4 -> 71.9, summary 58.7 -> 69.6, legacy full 63.4 -> 80.3, whole book
  58.4 -> 73.9, legacy+v2 67.1 -> 81.1, passage oracle 59.7 -> 87.9, quote oracle 60.7 -> 91.8. 4B and 9B base change
  little (milder label priors). Thinking-on answers are full mappings already. The old "without thinking the ceiling is
  matching, not information" finding was this artifact.
- Quote oracle (each character's own passage sentences, names left in; all 1,087 passages): 4B 64.8, 9B base 86.4,
  27B 91.8 (joint), 27B thinking 100.
- Hand-written oracles (7 Claude agents, 3 passages from each of the 21 books; data/manual): prior facts (1-4 per
  character, established before the chapter, sourced; median 45 words) and passage clues. On those 63 passages (joint
  for thinking off): 27B prior facts 80.4 (v2 72.5, legacy full 88.0, gpt-oss prior oracle 69.0), passage clues 92.6;
  thinking 99.5 / 100 (v2 95.2); 9B base prior facts 76.4 (v2 64.0, legacy full 70.9); 4B 46.6 (v2 43.8). Swapped with
  names exchanged: 27B 24.2, thinking 7.9. A few dozen words of the right prior facts beat whole sheets.
- Agent notes on the data: note files number chapters 1-based vs the 0-based book file (not a leak); v2 errors (summer
  foster relation reversed; here Anthony is Cleo's son; dark Marysieńka is Liska's cousin); a likely hallucinated
  charmem fact (deep: Lukas as Scarlett's half-sibling).
- Ops: a third vLLM engine hang (27B direct, quote oracle); scripts/hang_watchdog.py now kills and requeues jobs whose
  engine logs nothing for 10 minutes with requests pending.

### 17:00-19:00 UTC: data audit, gpt-oss chapter notes

- Audit clean: no alias leaks in masked passages; [CHAR i] counts match alias mentions; answer ids balanced; no v2 or
  charmem citation beyond the passage's chapter; chapter-so-far text ends right before the passage.
- Issues: (1) gpt-oss claims lose first-person narrators (16/30 books mostly first person): per 300-word snippet the
  model is not told who "I" is, so narrators are marked absent from most of their snippets (sandwich Rocky 7%,
  first_lie Evie 15%, refuse Rainy 15%, funny Daphne 30%; third-person principals median 50%) and other characters'
  claims say "the narrator" (10-17 per 1k words in first-person books; v2/charmem ~0). (2) Relational claims are
  garbled or rated down and dropped. (3) Alias identities missed even when named (Isabelle 61%, Evie 69%, Raeve 71%).
  (4) Thinking-on unreadable answers are mostly the 32k token cap (names only up to 9%), counted wrong.
- The Llama notes are per CHAPTER (one <snippet k> per earlier chapter), not per snippet; gpt-oss claims are per
  300-word snippet. gpt-oss redo of the Llama notes (same 8 questions per chapter, same layout; gen_chapnotes.py,
  median 3.2k words vs Llama 14k): 4B 49.0 (Llama 49.1, claims 45.3), 9B base 62.3 (67.8, 57.3), 27B 75.4 (80.3,
  71.5), 27B thinking 95.6 (96.7, 95.5). Per-chapter answering recovers 3-6 points; the remaining gap to Llama is
  larger in first-person books (27B -7.2 vs -2.4) and coincides with Llama's 4-5x more text.
- Book text by chapters (27B, joint): chapter so far 70.1, last 1 chapter 64.7, 1 chapter + chapter so far 74.8, last
  2 70.3, last 4 73.3, last 8 72.2, whole 73.9. 9B base: chapter so far 65.8 beats any amount of earlier book text.

### 2026-09-30 late: why the gpt-oss claims lose at the same length

- 25% of the claims' words are the character's own name (every claim restates it; other representations 1-6%).
- Cutting the claims to their most recent k words keeps only recent chapters: 500 words cover a median 3 of 20
  chapters, 2,000 words 8 of 20; v2/charmem/summaries cover the whole book at any length.
- Claims are short context-free facts (median 8 words; 18% of 5 words or fewer); relations are often garbled/dropped.
- At about the same length (~6k words) Llama notes from only their most recent chapters score 77.4 vs 71.5 for the
  claims over all chapters; gpt-oss's own chapter notes reach 75.4 at ~3.2k words. So the loss is the snippet/claim
  format, not the model. Narrator fix (claims with headings) does not move accuracy (4B 45.2 vs 45.3, 9B 59.8 vs 57.3).
- Queued: legacy_match (Llama notes cut answer by answer to gpt-oss chapter notes' length; median ~3.1k vs 3.3k words)
  and chapnotes_h_long (gpt-oss asked for thorough answers), to separate model from volume.

### 2026-10-01 02:00- UTC: new gpt-oss sheets (compressed notes), aiming past charmem at ~5k tokens

- Goal: a gpt-oss-120b character sheet that beats every non-oracle representation at its length (charmem 71.9 at
  4.4k mean prompt tokens on the 27B; v2 68.8 at 4.1k), ideally near the Llama notes (80.3 at 59k).
- Design: two stages. Exhaustive per-chapter notes (as the Llama notes are made), then one gpt-oss call per (book,
  boundary, principal) compresses every earlier chapter's notes into a ~1,000-word sheet (chiron/gen_sheet.py). The
  brief is a writer's character bible; it never mentions masking or identification. 1,000 words x 3 characters is about
  4.4k sheet tokens, i.e. ~5k per prompt.
- Round 1 (27B first, then every model for finalists): bible / chrono / dossier layouts from the thorough gpt-oss chapter
  notes (chapnotes_h_long), bible from the short gpt-oss notes (chapnotes), and bible from the Llama notes (a control on
  compression). legacy_gptoss (the Llama extraction prompt run with gpt-oss) is still generating.
- Ops: demoted the with-headings thinking evals (chiron_h, chapnotes_h) to priority 4 (scripts/demote_jobs.py) to free
  GPUs for generation.
- Round 1-2 (27B, thinking off, joint, all 21 books): every "bible" sheet loses to charmem (71.9 at 4.4k tokens):
  from the Llama notes 69.3 (5.7k), from thorough gpt-oss notes 67.0 (5.6k), from short gpt-oss notes 64.6 (5.5k);
  500 words 65.7 (3.1k). With the chapter so far: Llama-notes bible 77.0 vs charmem+pre 79.1. Dropping one section at a
  time barely moves them (no "how others refer to" +1.2 for the short-notes sheet, 0 for the Llama one).
- Why: (1) a fixed ~1,000-word target padded thin notes (13% of short-notes sheets are longer than their notes);
  (2) the bible brief asked for concrete facts "over general personality traits" and spent words on identity, forms of
  address and event lists, while the sheets that work (v2, charmem, Llama's own 500-word summary at 70.4) are mostly
  personality, voice, beliefs and motivations, i.e. what a POV passage's narration and dialogue reveal.
- Round 3: the CHIRON/v2 sections compressed in one call from complete notes (style "chiron", with [Ch. N] citations)
  from gpt-oss Llama-prompt notes, thorough gpt-oss notes and Llama notes; an "inner life" layout; both with --faithful
  (target <= 0.35 x notes, never expand, no invention). Also a 50-word distinctive-name index (non-principal names in
  each character's notes, ranked by count x share vs the other principals) appended to v2/charmem, and recency cuts
  of the notes at 1-3k words as no-generation baselines.
- Ops: 1-GPU gpt-oss servers hold only 2-12 of these 10-60k-token prompts at once (~400 tok/s); sheet jobs now run on
  2 GPUs. Sheet reps live in data/reps_sheets_<split>.jsonl (build_reps --sheets-only, seconds); evals read both files
  with a retry (a 2.7 GB reps rewrite under a running eval gave "Stale file handle"). ~40 older thinking-on eval jobs
  sit in /home/toolkit/eaiexp/state/queue/held (move back to queued/ to resume).
- More 27B results: latest-chapter gpt-oss notes appended to charmem lower it (charmem+last1 69.1); the name index
  hurts (alone 39.4, below names only; charmem+index 69.0; v2+index 65.2); chrono 69.0, dossier 68.6, distinct
  68.7, CHIRON layout from thorough notes 68.6. Recency cuts of the Llama notes: 1k words 65.1, 2k 71.5, 3k ~75.
  legacy_gptoss (gpt-oss, Llama extraction prompt, unfiltered) 75.9 at 70k tokens vs Llama 80.3 at 59k.
- Error analysis 1 (agent; 10 items legacy_full right, charmem wrong): decisive clue = current situation 4, named
  secondary character 2, past event 2, place 1, narrator 1; charmem lacks the clue in 6, has it but loses to a lure in 4.
- Error analysis 2 (agent; 7 items charmem right, new CHIRON-layout sheet wrong): every case has a fact in the wrong
  character's sheet that echoes the passage (actions, quotes, kinship, who-calls-whom like "ma'am"/"fox" reversed),
  inherited from unverified gpt-oss chapter notes; cover stories stated as facts; stale status; quotes and per-chapter
  timelines crowd out odd distinctive details. charmem's quote-verified ledgers filter these.
- So: the notes' attribution quality, not the sheet layout, is the gap. Round 4: compress charmem's own verified ledgers
  (with their claim/belief/flashback/state-change tags) with charmem's synthesis rules plus a Status block; the same
  jointly for the three principals (cast); a review pass; joint cast sheets from thorough notes.
- 05:25: charmem's exact principal synthesis (charmem_synth.py, copied verbatim; gen_charsynth.py) re-run on charmem's
  own reviewed ledgers reproduces the stored prompts byte for byte (messages_sha256 matches the run's synth records), yet
  the regenerated sheets score 68.2 on the 27B (n=1057) against 71.9 for the published charmem sheets. Same code,
  same inputs, same sampling settings (temperature 0.1, reasoning medium): a resample moves accuracy by ~3.7 points, so
  gaps of 2-3 points between single-sample sheet variants are within generation noise. Resampling two variants
  (csyn_ldg_r2, chiron_cnl_r2) to measure it directly.
- Other round-4 results: my approximation of charmem's rules + a Status block 67.9; joint (cast) version 67.8; review
  pass on ledger sheets 69.1 (vs 68.6), on Llama-prompt-notes sheets 68.3 (vs 67.8); gpt-oss Llama summary of the gpt-oss
  Llama-prompt notes (exact Llama summary prompt; gpt-oss writes ~1,800 words) 71.6 at 10k tokens, 80.3 with the chapter
  so far (partial). Entailment filter (gen_entail.py, gpt-oss, batches of 35): keeps 60% of sentences.
- 05:50 BIG: the sheets and recent story text are complementary. charmem plus the last k words of the book before the
  passage's chapter (shown once, before the character blocks; eval condition "charmem&book_last<k>"): k=500 74.9 at
  5.0k tokens (n=756, +4.6 vs charmem, 14/18 books), k=1000 77.3 at 5.5k (n=258 so far), k=2000 82.9 (n=77 so far).
  book_last2000 alone is 67.0. Queued on every model (eval_conds hybrid), plus the same add-on for v2, the Llama summary,
  character summaries, the re-run charmem and the gpt-oss Llama summary, and a plot-summary tail instead of raw text.
- Filtered gpt-oss Llama-pipeline notes (legacy_gptoss_ent): 77.9 at 38.6k tokens (n=751), on the frontier between
  legacy_r6000 (77.4 at 23k) and legacy_full (80.3 at 59k).
- 06:35 Formatting control: build_reps had nested every new sheet's "## Section" headings under the "## Name" block
  ("###"). That costs the 27B 1.5-2.5 points: charmem nested (and with plain hyphens) 69.4 vs 71.9 as published;
  the charmem re-run as written 70.7 vs 69.3 nested. So the resample gap to the published charmem is ~1 point
  (three nested resamples 68.5/69.0/69.3), and every sheet_* number above is understated by ~2. Re-evaluating 15
  variants as written (sheet_<v>_flat).
- Sheet + recent text, complete (27B): charmem + last 500 words 75.4 at 5.0k tokens; + last 1,000 76.1 at 5.7k;
  + last 2,000 76.4 at 7.0k; Llama summary + last 1,000 76.8 at 4.0k; gpt-oss character summaries + last 1,000 76.9
  at 5.3k; v2 + last 1,000 75.2 at 5.4k; gpt-oss Llama summary + last 1,000 76.3 at 11.5k. Recent text alone: 500
  words 60.2, 1,000 65.2. A gpt-oss 300-word recap of the last 2,000 words instead of raw text: 72.4 (+0.5 only).
  Consensus of three charmem samples: 69.3 nested (no gain over single samples).
- Filtered gpt-oss Llama-pipeline notes complete: legacy_gptoss_ent 77.4 at 40.7k tokens (Llama full 80.3 at 59k).
- 07:05 As written (headings not nested), the one-call sheets: CHIRON layout from Llama notes 71.7, chrono 70.4, joint
  70.2, CHIRON from ledgers 69.7; charmem re-run 70.7. Every reasonable ~1,000-word design ends at 70-72 on the 27B;
  no sheet alone clearly beats charmem (71.9).
- Hybrid placement: the recent text after the sheets (right before the passage) is worse than before them (charmem
  72.8 vs 76.1). Trimming charmem to 500 words + 1,000 words of text: 75.0 at 4.2k; Llama summary + 500 words 75.9
  at 3.4k. On Qwen3.5-9B base charmem 61.2 -> 66.7 with the last 1,000 words (Llama full 67.8); Qwen3-4B 45.2 -> 49.1
  (Llama full 49.1).
- Ops: the 62 held thinking jobs are back in the queue (priority 4/9); they fill spare GPUs and will keep running.
- Paired per-book differences vs charmem, 27B, 95% bootstrap over the 21 books: charmem+last 500 words +3.5 [+1.8,
  +5.4] 16/21; +last 1,000 +4.2 [+1.8, +6.6]; gpt-oss summaries + last 1,000 +4.8 [+2.1, +7.6]; Llama summary + last
  1,000 +4.9 [+1.9, +7.8]; v2 + last 1,000 +3.4 [+0.6, +6.0]; charmem re-run -1.2 [-2.9, +0.5]; best one-call sheet
  (CHIRON layout from Llama notes, as written) -0.2 [-2.3, +1.7]; filtered gpt-oss Llama-pipeline notes +5.5 [+2.0, +8.5].
- Report v21 published (same URL) at 07:22 with a "New gpt-oss sheets (Oct 1)" section and the hybrids on the chart.
