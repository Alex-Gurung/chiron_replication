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
- 08:40 gpt-oss redo of the Llama summary step held to ~400 words (gen_legsum --words 500 with rewrite-to-length):
  69.9 at 2.8k tokens alone (Llama's own summary 70.4 at 2.7k), 75.5 at 4.2k with the last 1,000 words, 76.0 at 4.8k
  with 1,500, 81.0 at 6.3k with 1,000 words and the chapter so far. Llama summary + last 1,500 words 77.1 at 4.7k ties the Llama notes cut to 3k words (77.3 at 15k).

### Summary of the Oct 1 campaign (for the 11:00 check-in)

- Asked for: a gpt-oss-120b character sheet, Pareto-optimal against non-oracle representations, ~5k mean prompt
  tokens, +5 over existing sheets on Qwen3.8-27B (thinking off), ideally 80%.
- Sheets alone: no. ~35 one-call designs plus charmem's own synthesis re-run all land at 68-72 at ~1,000 words per
  character (charmem 71.9; best new, as written, 71.7; charmem re-run 70.7). Lures (facts in the wrong character's
  sheet), not missing content, limit them; adding shared content (latest-chapter notes, names, status) makes it worse.
- Sheet + the last 500-1,500 words of the story, shown once before the sheets: +3.5 to +5 on the 27B (all intervals
  over books exclude 0), +5 on 9B base, +4 on 4B, +1.6 with thinking. 75-77% at 4-6k tokens; matches the 15k-token
  Llama cut and beats every other non-oracle representation under 15k. With the chapter so far: 81.7% at 7.8k.
- Not reached: 80% at ~5k without the chapter so far (best ~77).
- gpt-oss Llama notes (archive extraction prompt + filler removal + gpt-oss entailment filter): 77.4 at 41k vs Llama
  80.3 at 59k; gpt-oss Llama summary ~400 words 69.9 vs Llama 70.4.

### 2026-10-01 midday: plot + character combinations, NCP's Story Information, best-of-kind chart

- NCP Story Information (ncp_cohorts_v2): prior_plot_summary = SuperSummary synopses of every earlier chapter (median
  3.6k words), story_text = the two preceding chapters, character sheets = v2; no whole-book overview in this cohort;
  next_chapter_synopsis describes the passage's own chapter (oracle). data/ncp_story.jsonl (build_ncp_story.py); eval
  conditions ncp_plot[@last<k>], ncp_story, ncp_storynext, ncp_next, combinable as <sheet>&<...>.
- 27B (thinking off): sheets only (v2) 68.8 at 4.1k; synopses only 71.2 at 5.6k; last 2 chapters only 70.3 at 8.2k;
  synopses + last 2 chapters 74.8 at 13.3k; v2 + synopses 76.7 at 9.1k; v2 + last 2 chapters 77.6 at 11.7k; all of it
  79.5 at 16.8k (charmem instead of v2 79.8); + the next-chapter synopsis (oracle) 80.5; that one-liner alone 51.6.
  Every part helps; sheets + last 1,000 words (76.1-76.8 at 4-6k) gets most of it.
- Plot + characters: charmem + synopses 77.6 at 9.4k, + last 1,000 words of the synopses 75.8 at 5.6k; gpt-oss plot
  summary (1,000 words) + charmem 74.4 at 6.2k, + Llama summary 74.3 at 4.6k, + v2 75.1. Recent book text beats plot
  summaries as the add-on at equal length.
- Smaller models: 9B base Story Information 65.0 (charmem 61.2, charmem + last 1,000 words 66.7); 4B 44.3, below
  charmem alone (45.2): long context hurts the 4B; charmem + last 1,000 words is best there (49.1).
- Report: "Best of each kind" chart (Pareto line per kind of method) and a Story Information table group.
- Story Information from chosen parts (eval condition <sheet>&si-<plot>-<recent>, 27B thinking off): with synopses +
  last 2 chapters, sheets v2 79.5 / charmem 79.8 / gpt-oss summaries 79.1 / gpt-oss Llama-style summary 79.3 / Llama
  summary 80.3 (= full Llama notes at 15.4k vs 59k tokens) / none 74.8. Plot part with v2 + last 2 chapters: SuperSummary
  synopses 79.5, gpt-oss ~1k-word plot 78.2, gpt-oss ~4k-word plot 76.4, none 77.6. Recent part with synopses: last
  1,000 words instead of 2 chapters costs ~1 point, saves ~6k tokens (charmem 78.9 at 10.7k, Llama summary 79.1 at
  9.1k). Fully generated (gpt-oss sheet + gpt-oss plot ~1k + last 1,000 words): charmem 76.4 at 7.6k, gpt-oss
  Llama-style summary 75.6 at 6.1k: the gpt-oss plot summary adds ~0 over sheet + recent text; the synopses add 2-3.
  With the chapter so far: v2 + last 2 chapters 83.6 at 13.8k; full Story Information 83.3-83.6; the Llama-style
  hybrids 81-82 at 6-7k.
- Story Information grid on the smaller models: 9B base 63-67 (best Llama summary + synopses + last 2 chapters 67.1 at
  15.4k, the same as Llama summary + last 1,000 words alone, 67.0 at 4.0k); 4B 43-45, all below charmem + last 1,000
  words (49.1): long context and plot summaries hurt the 4B.
- Richer notes in the package (27B thinking off): Llama notes cut to 3k words + synopses + last 2 chapters 82.9 at
  27.7k (best non-oracle without the chapter so far; Llama full alone 80.3 at 59k); Llama last 3,000 words + same 82.2;
  gpt-oss chapter notes + same 82.1 at 28.5k; best one-call sheet + same 79.7. With only the last 1,000 words: Llama
  3k cut 81.7 at 16.3k (synopses on top: 81.2), gpt-oss chapter notes 79.1 at 17.2k, best one-call sheet 78.2 at
  7.7k, Llama last 3,000 words 77.2. Full Llama notes / filtered gpt-oss notes in the package still running.
  Queued: chapter-so-far and thinking (262k) for the Llama-3k-cut and gpt-oss-chapter-notes combinations.
- 20:50 Full Llama notes + synopses + last 2 chapters 83.8 at 71.8k (best without the chapter so far); filtered gpt-oss
  Llama-style notes + same 83.3 at 53k (n=1045); filtered gpt-oss notes + last 1,000 words 80.3 at 42k, + synopses
  81.4. With the chapter so far: Llama 3k cut + Story Information 86.9 at 29.9k, + last 1,000 words 86.1 at 18.5k;
  gpt-oss chapter notes + same 86.4 / 85.7 (Llama full + chapter so far: 85.7 at 61k).
- Ops: 2-GPU jobs were starving (freed GPUs return one at a time and 1-GPU jobs take them). scripts/free_pairs.py holds
  queued 1-GPU jobs, stops running 1-GPU thinking jobs two per worker, lets the 2-GPU jobs claim the pairs, then
  requeues/releases. Long thinking jobs raised to priority -2 so a freed pair goes to them.
- 21:50 final long-context 27B (thinking off): full Llama notes + last 1,000 words 83.6 at 60k; + synopses + last 2
  chapters 83.8 at 72k; filtered gpt-oss notes + Story Information 82.3 at 53k (the partial 83.3 settled down), + last
  1,000 words 80.3 at 42k. The Llama notes cut to 3k words in the package (82.9 at 28k) gets within a point at 40% of
  the tokens.
- 22:50 thinking-on results for the new conditions (27B): everything with recent text or synopses sits at 95.6-97.0
  (charmem alone 94.5); the dataset's synopses alone 96.9 at 5.7k; Llama summary + last 1,000 words 96.2 at 4.1k;
  charmem + Story Information 97.0 at 17k; Llama 3k cut / gpt-oss chapter notes + Story Information 96.8-96.9. With
  thinking, the plot synopses alone are as good as any package: the reasoning model needs the plot, not the sheets.
- 00:20 Oct 2: all first-priority thinking runs done. Thinking on, every package with recent text or synopses is at
  95.6-97.0 (rich notes + Story Information 96.8-96.9); with the chapter so far everything is 97.6-98.1 (charmem + the
  chapter so far alone 97.7). Thinking saturates: the representation barely matters once the plot is there.
  Remaining: second-tier thinking fill-ins (short cuts with the chapter so far) and one Qwen3-4B long job.
- 00:50 rich notes on the smaller models: 9B base best = full Llama notes + last 1,000 words 70.5 (Llama full alone
  67.8); Llama last-3k + last 1,000 words 68.2 at 13k; filtered gpt-oss notes + last 1,000 words 68.0; any package with
  the synopses + last 2 chapters drops to 63-66. 4B best = one-call sheet + last 1,000 words 50.5 at 7.7k (Llama full
  alone 49.1); Story Information packages 43-44. The small models want character notes + a little recent text; long
  plot/chapter context hurts them, while the 27B gains from it.
- 15:40 Oct 2: why gpt-oss "done the Llama way" (77.4) trails the Llama notes (80.3). Side by side (witch, Fern, ch 10):
  Llama writes traits and quotes ("supernatural vomit from beyond the stars"); gpt-oss lists events, objects and style
  labels ("uses exclamation marks"), and my batch-of-35 filter dropped correct specifics (Pine-Sol, Turnabout) while
  keeping wrong-chapter "She ..." sentences. Words from chapters the character is not in: Llama 7.7%, gpt-oss 15.8%
  (10.8% after my filter), ~60% of them pronoun-led. The redo also skipped the archive's simplification step and used a
  different filter rubric, so it was not the same process.
  Now: chiron/gen_legacy_exact.py redoes everything after extraction exactly as the archive (via the diversity repo's
  gpt-oss replay): spaCy en_core_web_md sentences (fixes the "Dr." split), one simplification call per sentence, one
  entailment call per statement with the archive's role/rubric, low reasoning, keep only 5s, archive layout
  (legacy_gptoss_x); then the archive summary guided to ~500 words (sheet_legsum_x_500_flat). Fixes only: filler
  sentences dropped before simplification, 1,024-token allowance (the replay's 256 + stop "\n" ends inside reasoning).
  Early: 63% of statements rated 5, 26% rated 1; ~12% of compound sentences split. Also queued: notes without the
  chapters the character is absent from (legacy_full_pres, legacy_gptoss_ent_pres; legacy_full_rr = layout control).
  spaCy lives in .pylib (uv pip install --target; appended to sys.path by the script only).
- 17:40 exact redo results (27B thinking off, 21 books, paired): legacy_gptoss_x 75.9 at 41.5k vs Llama full 80.3 at
  59k (-4.4, 3/21 books better) and vs my batch filter 77.4 (-1.5). +pre 84.0 vs 85.7; + last 1,000 words 80.2 vs 83.6;
  + Story Information 81.7 vs 83.8. Summary (~400 words) 68.8 vs Llama 70.4 (-1.6), and -0.6 to -2.1 in every package.
  9B base: notes -2.1, summary -4.9; 4B: notes -3.1, summary -2.2. Copying the archive's later steps exactly does not
  close the gap; it is slightly worse than my filter.
  Wrong-chapter notes are not the cause: dropping the chapters a character is absent from gives Llama +0.5 over the
  layout control (12/21 books) and gpt-oss +0.9 (15/21).
  Kept statements look alike (8.3 vs 8.4 words; quotes 3.5% gpt-oss vs 1.9% Llama; 1.36M vs 1.40M words kept). The
  difference: Llama names the character in 47% of kept statements, gpt-oss in 28% (68% of gpt-oss statements start
  with a pronoun vs 50%). The gpt-oss judge passes 80% of Llama's statements and 65% of its own.
- 18:30 judge control: the Llama notes re-rated by the same exact gpt-oss entailment step (legacy_full_xf) 80.2 at
  46.8k vs Llama full 80.3 at 59k (8/21 books; +pre 85.7 vs 85.7), and +4.3 over legacy_gptoss_x (18/21 books). The
  gpt-oss judge keeps 80% of Llama's statements and costs nothing; the gap is in gpt-oss's extracted answers, not the
  later steps. (Side result: the gpt-oss re-rating trims the Llama notes by 21% at no cost.)
  Ruled out: chapter misalignment (Llama and gpt-oss notes both match their own chapter, 0.88 / 0.78 name overlap vs
  0.43 / 0.37 for neighbours, every book); outside knowledge (93-94% of names in kept statements occur in the chapter,
  most of the rest are the asked name used before the text uses it); naming the subject (She/He -> name: +0.3 alone,
  -0.9 with the last 1,000 words); wrong-chapter notes (+0.5 / +0.9). A TF-IDF slot-matching proxy cannot tell the
  note sets apart (0.405-0.410 slot accuracy, chance 0.33): the difference is not in shared words.
  Bug fixed: the Llama loader split book ids at the first underscore and missed 4 books (rerun).
  Running: the same notes one section at a time (legacy_full_xf_sec<i> vs legacy_gptoss_x_sec<i>).
- 19:10 per section (both through the same gpt-oss filter and layout; 27B thinking off, 21 books): Llama vs gpt-oss
  answers: how the character speaks 68.2 vs 60.4 (+7.9, 20/21 books), personality and appearance 74.2 vs 68.9 (+5.2,
  19/21), plot and motivation 76.0 vs 71.4 (+4.6, 18/21), knowledge 75.2 vs 72.6 (+2.6, 18/21). Llama's speech notes
  describe attitude ("She is direct and forceful"); gpt-oss's list surface features of lines ("uses the command
  Look", 27% with a quote vs 13%). Simple specificity counts (names, numbers, quotes) do not separate the two.
  Report v36. All chiron workers stopped at 19:10 (the supervisor's idle exit never fires: it waits for the shared
  queue to drain, and other projects' parked jobs stay queued).
- 01:00 Oct 3: gpt-oss extraction prompt variants on the personality/appearance and speech questions (same exact filter,
  27B thinking off, per section, 21 books). Speech (Llama 68.2, gpt-oss 60.4): questions worded as a writer's brief
  (manner, attitude, how they talk to whom; not features of single lines) 63.6 (+3.2, 16/21 books); low reasoning 62.0;
  both 62.1. Personality/appearance (Llama 74.2, gpt-oss 68.9): low 69.1, brief 67.4, both 68.8: no gain, although the
  brief answers read like Llama's ("exuberant, impulsive, rebellious toward authority"). Rewording recovers about a
  third of the speech gap and none of the personality gap; the remaining difference is not the question wording.
  Workers stopped.
- 02:00 the reworded speech answers in the full gpt-oss notes (legacy_gptoss_xs): 76.4 vs 75.9 alone (12/21 books),
  79.8 vs 80.2 with the last 1,000 words; their ~500-word summary 68.2 vs 68.8, and a second sample of the original
  summary 67.9 (with the last 1,000 words 75.0 / 74.7 / 74.8). The +3.2 on the speech section alone does not survive in
  the full notes: within noise. Summary resampling moves ~1 point. Workers stopped. Report v37.
- 13:00 Oct 3 style vs content for the ~500-word summaries (2 x 2, every cell rewritten by gpt-oss: Llama or gpt-oss
  summary x prose or bullets; 27B thinking off). Alone: Llama content prose 67.6 / bullets 67.0, gpt-oss content prose
  66.3 / bullets 67.8 (originals: Llama 70.4, gpt-oss 68.8: rewriting itself costs 1-3 points). With the last 1,000
  words: Llama content 75.5 prose / 75.6 bullets, gpt-oss 73.8 / 75.1 (a partial 78.3 for Llama bullets on 20 books
  did not hold). Style: Llama content -0.6 / +0.1 prose vs bullets, gpt-oss content -1.5 / -1.3 (bullets better).
  Content: in prose Llama +1.3 / +1.6 (14/21, 13/21 books), in bullets -0.8 / +0.5. Everything within ~1.5 points,
  near summary resampling noise (0.9): at summary length neither style nor content separates the two much; the large
  gap is in the full notes, where the format is identical.
  Ops: one restyle eval (test split) hung ~03:00-12:55 with two workers held; no hang watchdog was running. Requeued on
  one 2-GPU worker with the watchdog on; always run hang_watchdog with eval batches.
- 16:10 Oct 6: the CHIRON paper's Character-Summary baseline, done the paper's way (chiron/gen_csum.py: the whole story
  so far in the generation module's prompt with its summarize_story question, greedy, then the entailment filter;
  gpt-oss for Mistral 7B, last 88k words when longer). ~160 words (129 filtered). 27B thinking off: 62.5 at 1.2k tokens
  (filtered 62.7), vs our rolling ~700-word summary 69.6 at 4.0k and the same cut to 250 words 60.6 (+1.9, 16/21
  books); + last 1,000 words 73.1 (filtered 72.3) vs 76.7. 9B base 54.4 / 54.2 (rolling 58.9), 4B 39.4 / 39.8 (40.9).
  The filter changes nothing. Against this baseline every sheet wins (CHIRON condensed 70.4 at 2.7k, full 80.3), like
  the paper's gap (44.9 vs 47.6-58.5); our rolling summary is a much stronger baseline because it is 4x longer.
  Run on one 8-GPU worker after the first ~20 min (user: fewer GPUs). Report v39.
- 23:40 Oct 9: perplexity inside the full Story Information, swapping only the character sheets.
  First attempts (chiron/ppl_eval.py ncp_storynext / ncp_full, my own prompt on our 1,087 passages) were stopped: with
  the notes placed after the story text every sheet raised the 4B's perplexity (+1 to +8%), a layout artefact; and vLLM
  does not read the prefix cache for prompt-logprob requests, so every row costs its whole prompt.
  Final: the diversity project's own NCP scorer (chiron/ncp_ruler_sheets.py, modelled on ncp_eval/v70_score_writer.py
  with its helpers imported from the frozen q4v2 code: v16_writing_messages, target after "<answer>\n", mean logprob per
  token, frozen Qwen3-4B, 57,344 window) with only example["character_sheets"] changed (principals swapped,
  "Supporting cast" kept). 3,703 sections / 29 books have every variant (664 test sections in dark, mercy, witch; god
  has no sheets); none skipped. The unchanged condition reproduces the stored controls (3,573 sections, mean |diff|
  5.5e-05 nats, max 8.4e-03).
  Over no sheets (nats/token, B%, books better): Llama summary +0.0248 / +2.43 / 29 of 29; charmem +0.0160 / +1.56 /
  29; v2 +0.0149 / +1.45 / 29; rolling summary +0.0132 / +1.28 / 26; gpt-oss CHIRON-style summary +0.0116 / +1.12 /
  25; the CHIRON paper's character summary +0.0076 / +0.74 / 26; supporting cast only +0.0057 / +0.56 / 25.
  Over shipped v2: Llama summary +0.0100 (29/29 books; test cohort +0.0093, 3/3), charmem +0.0012 (18/29), rolling
  summary -0.0016 (11/29), gpt-oss CHIRON-style -0.0032 (6/29), paper's summary -0.0072 (1/29).
  Llama summary vs rolling summary +0.0116 (28/29 books), vs the paper's summary +0.0172 (29/29), vs gpt-oss
  CHIRON-style +0.0132 (29/29). One 8-GPU worker, ~45 min; stopped. Report v40.
  Ops: the pod was preempted and restarted once (22:12); hang_watchdog false-killed three restarted jobs because
  vllm.log is appended across restarts (fixed: only lines after the last "Application startup complete").
- 12:10 Oct 10: same-model grid started (user: model-fixed experiments; claims = (1) character information is needed
  for character identification and for book writing, (2) CHIRON sheets carry it efficiently and effectively, more than
  plot summaries and naive character summaries). Two 8-GPU nodes. Generators: l70 = Llama-3.3-70B (downloaded, 132 GB;
  CHIRON notes from the archive), gptoss (have), q4b = Qwen3-4B-Instruct. CHIRON_GEN switches every gen script
  (common.GEN / gen()); q4b uses the archive's own decoding (300 tokens, stop at newline; simplification 256 tokens;
  entailment 16 tokens; summary min(0.8 x input, 2048) tokens). Per generator: CHIRON notes + condensed, rolling
  summary, the paper's whole-story summary (+ filter), plot summaries (hier, global; 500/1000/2000 words).
  Evals: character identification (27B, 9B base, 4B, + Llama-3.3-70B judge) and the NCP writer ruler
  (chiron/ncp_ruler_grid.py: plot slot x character slot; judges Qwen3-4B canonical + Qwen3.8-27B thinking off).
  Report: floating section bar (v41).
- 15:25 Oct 10: NCP grid wave 1 complete on the canonical 4B judge (chiron/ncp_ruler_grid.py, every 2nd section per
  book: 1,861 sections, 29 books; references repeat bit-exactly). Gain over a prompt with neither block (nats/token):
  plot summary only (shipped synopses) +0.0237 (23/29 books); v2 sheets only -0.0018 (12/29); both +0.0387 (28/29).
  Inside the full prompt (over ship|none): Llama CHIRON condensed +0.0255 (29/29), charmem +0.0162 (29/29), v2 +0.0150
  (29/29), rolling summary +0.0132, gpt-oss CHIRON-style +0.0118, the paper's summary +0.0078. As the only summary
  block (over none|none): Llama CHIRON +0.0059 (17/29), charmem +0.0009, v2 -0.0018, the paper's summary -0.0036,
  rolling summary -0.0081, gpt-oss CHIRON-style -0.0137; a gpt-oss plot summary in the plot slot -0.0182 (3/29).
  So on this judge sheets add only next to the dataset's synopses; alone they are flat. Second judge (Qwen3.8-27B,
  thinking off) running on 12 core conditions (needs max_num_seqs 32, max_num_batched_tokens 2048, memory 0.85).
  Generation: q4b CHIRON notes + condensed done (30,864 answers, 1,314 sheets); q4b's paper-style summary loops under
  greedy decoding in 78% of cases (kept, plus a filtered de-duplicated version); l70 summaries done without the
  per-sentence filter (it rereads ~100k tokens per sentence). Ops: pausing in-process scorers left orphan engines
  (scripts/stop_jobs.py now sweeps them); replacing a job's shell script mid-run gives "Stale file handle" at exit.
  Report v42: sections "Is character information needed?" and "Same-model grid".
- 19:00 Oct 10: both chiron pods keep being preempted (EAI stateInfo: "account occupancy 15.9 > 8.6"; the NCP campaign
  holds ~65 GPUs running + 44 queued). w98 was out 15:38-18:22, w97 15:38-16:08 and again from 18:22; both queued at
  18:57. scripts/lane_balance.py moves queued jobs to the lane of whichever pod runs; scripts/grid_pipeline.sh queues
  each generator's evals when its generation ends. No pods added (would push the account further over share).
  Llama-fixed character identification (27B / 9B base / 4B, 21 books): CHIRON condensed 70.4 / 63.1 / 42.2 at 2.7k;
  rolling summary 72.8 / 63.7 / 41.4 at 2.6k (+2.4 on the 27B, 15/21 books); paper-style summary 65.5 / 60.1 / 39.9 at
  1.2k; plot summary ~1,000 words 67.2-68.7 / 63.7-64.1 / 41.5-43.0; CHIRON + plot 75.4 / 63.9 / 43.5; rolling + plot
  76.5 / 64.9 / 45.0; CHIRON notes in full 80.3 / 67.8 / 49.1 at 59k. With the generator fixed, the condensed CHIRON
  sheet does not beat a rolling summary; only the full notes stand out.
  Second NCP judge (Qwen3.8-27B, 1,213 sections so far, 29 books), gain over neither block: sheets only v2 +0.0130
  (29/29), Llama CHIRON +0.0139, gpt-oss rolling +0.0167; plot only +0.0230 (shipped), +0.0187 (gpt-oss 1,000 words);
  both +0.0261. In the full prompt over no sheets: rolling +0.0060, gpt-oss CHIRON-style +0.0050, Llama CHIRON +0.0041,
  charmem +0.0039, v2 +0.0031, paper's summary +0.0016. The 4B's "sheets alone are flat" and "gpt-oss plot summary
  hurts" do not show on the 27B. Report v43.
  q4b: 500-word CHIRON condensed done (legsum_q4b_x_500); top-up of csum/plot nearly done (csum 1,277/1,314).
