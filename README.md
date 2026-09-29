# chiron_replication

Replicates the masked-character-prediction evaluation from CHIRON
(Gurung & Lapata, EMNLP Findings 2024, arXiv:2406.10190) on the 30-book NCP
dataset from `diversity`, to compare generations of character sheets. The
premise: a more useful character sheet lets a model put masked names back in.

## Task

- **Main set (3-way):** an NCP section (~300 words; the sections of a chapter
  partition it exactly) that names all three of the book's principals.
- **Two-principal set (2-way):** sections naming exactly two principals; the
  model chooses between those two. Kept separate from the main set.
- **Eligibility:** every candidate principal is named in at least two chapters
  before the section's chapter.
- **Masking:** every name mention of each principal becomes `[CHAR i]`, from
  hand-checked alias tables in `aliases/` (full, first and last names,
  nicknames, earlier names, secret identities, titles used as names;
  all-caps forms are added automatically). Possessives stay outside the mask.
  Mask ids are seeded per item. Chapter headings are stripped. Sections are
  dropped if they contain an ambiguous string (e.g. a surname two principals
  share), another character's name that contains a principal alias, or any
  leftover capitalised principal-name token.
- **Pronoun variant (main set only):** gpt-oss resolves every third-person
  pronoun; those referring to a principal become `[CHAR i]` / `[CHAR i]'s`.
  A missed pronoun still leaks gender, so read this variant with care.
- **Model input:** the masked section plus one representation per character,
  labelled with the name the text uses before this chapter. Nothing else.
- **Scoring:** next-token probabilities of the mask digits (temperature 0),
  every order of the character blocks. Reported: per-character accuracy
  (paper-style argmax), best one-to-one assignment accuracy, and all-correct
  rate, per book.

## Representations (all built from chapters before the section's chapter)

| Condition | Source |
|---|---|
| `noinfo` | names only |
| `v2` | `character_sheets` in `ncp_cohorts_v2` |
| `legacy`, `legacy_full` | Llama-3.3-70B CHIRON-style sheets (compressed / full), HF `agurung/new_ncp_data_creation` |
| `chiron`, `chiron_<category>`, `chiron_r<k>` | gpt-oss CHIRON-style sheet: the 8 CHIRON questions per ~300-word snippet and character, claims rated on the paper's 1-5 scale, rating-5 claims grouped by category and TF-IDF deduplicated at 0.9. `_r<k>` keeps the most recent claims up to k words. |
| `summary` | gpt-oss rolling character summary (update per chapter, condensed to ~700 words) |
| `charmem` | the Sep 8 gpt-oss charmem rebuild, finished by `chiron/charmem_finish.py` |
| `book`, `book_last<k>` | the novel so far, or its last k words |
| `swap:<cond>` | each character gets another principal's representation |
| `<cond>@<k>` | representation truncated to its first k words |

## Models

- Predictor: `Qwen/Qwen3-4B-Instruct-2507` (262k context); `mistralai/Mistral-7B-Instruct-v0.2` as a sanity check.
- Every generated representation: `openai/gpt-oss-120b`, reasoning effort medium.
  Short-prompt work runs one server per GPU (measured faster per GPU than
  two-GPU servers); charmem synthesis uses two GPUs per server for 131k prompts.

## Pipeline

```
python3 chiron/prep.py                                   # snippets + display names
.venv/bin/python chiron/ner.py --books ...               # PERSON candidates for alias tables
python3 chiron/build_items.py --split S [--two]          # masked items
python3 scripts/queue_jobs.py chiron|summary|charmem     # gpt-oss generation (eaiexp lane "chiron")
python3 scripts/queue_jobs.py pronouns S...              # pronoun variant
.venv/bin/python chiron/build_reps.py --split S          # representations for the split's items
python3 scripts/queue_jobs.py final S...                 # every eval condition
python3 chiron/score.py --items items_S[_two|_pron]      # tables
```

Workers: `bash scripts/runner_chiron_worker.sh N` launches an 8-GPU eaiexp
worker that serves only the `chiron` lane.

## Checks

- Memorization (`chiron/memo_probe.py`): temperature-0 continuations from book
  prefixes; flag a book if a continuation reproduces >5% of the reference's
  13-grams or a verbatim span of 20+ words (the RAM "unslop" probe).
- Name matching: chapter text stores accented letters decomposed (NFD) while
  labels are composed (NFC); everything is normalised to NFC.
