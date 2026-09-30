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
