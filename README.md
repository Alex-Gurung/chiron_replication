# chiron_replication

Replicates the masked-character-prediction evaluation from CHIRON
(Gurung & Lapata, EMNLP Findings 2024, arXiv:2406.10190) on the 30-book NCP
dataset from `diversity`, to compare generations of character sheets. The
premise: a more useful character sheet lets a model put masked names back in.

## Task

- **Item:** one NCP section (~300 words; the sections of a chapter partition it
  exactly) whose text names all three of the book's principal characters.
- **Eligibility:** a principal counts only if it appears in at least two
  chapters before the section's chapter.
- **Masking:** every name mention of each principal (full, first, last,
  nickname, title-as-name, possessive) becomes `[CHAR i]`, with ids shuffled
  per item. Chapter headings are stripped. The paper masks names only and
  leaves pronouns; a pronoun variant is still to be decided.
- **Model input:** the masked section plus one representation per character,
  labelled with the name the text uses before this chapter. Nothing else from
  the book.
- **Metrics:** per-character accuracy and all-three-correct accuracy, reported
  per book.

## Representations

| Condition | Source |
|---|---|
| No-Info | names only |
| Character-Summary | gpt-oss-120b summary of the character from the book up to the chapter |
| Entire-book | raw text up to the chapter, plus last-k-word truncations |
| CHIRON-style sheet | the 8 CHIRON questions per section, answered by gpt-oss-120b, grouped by category and deduplicated |
| v2 sheet | `character_sheets` in `ncp_cohorts_v2` |

Planned ablations: input length vs accuracy, and per-category sheets (the
paper's "Agreed" setting).

## Models

- Predictor: `Qwen/Qwen3-4B-Instruct-2507`; `mistralai/Mistral-7B-Instruct-v0.2` as a sanity check.
- Every generated representation: `openai/gpt-oss-120b`, reasoning effort medium.

## Data

- Sections and v2 sheets: `/home/toolkit/ncp_cohorts_v2/{train,val,test}_examples.jsonl`
  (`story_id`, `chapter_index`, `chunk_index`, `next_chapter`, `main_character_labels`, `character_sheets`).
- Raw chapters: `/home/toolkit/ncp_charmem_v3/chapters.jsonl` (rebuilt from v2).
- Legacy sheets (Llama-3.3-70B): HF dataset `agurung/new_ncp_data_creation`,
  `{split}_long_story_storycharchap_to_csheet.pkl` (full) and
  `{split}_long_story_character_sheet_summaryllama70B_0.5max.pkl` (compressed).
  Key `{book}_{character}_{i}` covers chapters `0..i-1`.
- Books: all 30, since nothing is trained. Development starts on the test
  split (dark, god, mercy, witch).

## Checks

- Memorization: temperature-0 continuations from book prefixes; flag a book if
  a continuation reproduces >5% of the reference's 13-grams or any verbatim
  span of 20+ words (the RAM "unslop" probe). No-Info accuracy per book is a
  second signal.
- Name matching: chapter text stores accented letters decomposed (NFD) while
  labels are composed (NFC); normalise both. Nicknames (e.g. doors: "Izzy")
  and labels that include later-revealed names (dark: "The Leszy (Eliasz
  Kowal)") need a hand-checked alias table.
