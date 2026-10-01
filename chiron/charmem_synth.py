"""charmem's principal-sheet synthesis (Sep 8 gpt-oss pipeline), copied verbatim, so it can run on other notes.

Source: /home/toolkit/eaiexp/runners/charmem_gptoss_medium_20260908/{common,synthesize}.py. The block between the
COPIED markers is unchanged (constants, source_entries, principal_messages, validators, rendering); gen_charsynth.py
feeds it entries built from either charmem's own ledgers (a replication check) or other notes.
"""
import collections
import json
import re

# ---- COPIED from common.py ----
SECTIONS = (
    "physicality_and_personality", "dialogue_and_voice", "history_and_circumstances",
    "knowledge_and_beliefs", "goals_and_motivations", "relationships",
)
SECTION_PURPOSES = {
    "physicality_and_personality": "appearance, body, health, mannerisms, behavior, temperament, and emotional tendencies",
    "dialogue_and_voice": "speech, writing, verbal habits, register, and voice",
    "history_and_circumstances": "objective biography, experiences, roles, social position, and current circumstances",
    "knowledge_and_beliefs": "knowledge, learning, beliefs, misunderstandings, skills, and limits on what the character knows",
    "goals_and_motivations": "active or recurring aims, motivations, decisions, constraints, and meaningful changes or completions",
    "relationships": "specific loyalties, conflicts, dependencies, power differences, and changes in relationships, with clear ownership of actions and feelings",
}

# ---- COPIED from synthesize.py ----
MAX_TOKENS = 24576  # GPT-OSS high reasoning and final JSON share the completion budget.
REQUESTED_PRINCIPAL_WORDS, MAX_PRINCIPAL_WORDS = 900, 1000
REQUESTED_SUPPORTING_WORDS, MAX_SUPPORTING_WORDS = 350, 550
ID_FIX = re.compile(r"^ch(\d{1,3}):")
LEAK = re.compile(r"\(?\s*(?:source_ids?|sources?|see|per|from|in|by)?\s*:?\s*\bch\d{1,3}:[a-z_]*:?\d{0,3}\b[^)\n]*\)?", re.I)
LEAK_TOKEN = re.compile(r"\bch\d{1,3}:[a-z_]+(?::\d{1,3})?\b|\bch\d{1,3}:\d{1,3}\b", re.I)


def normalize_ids(ids):
    """Unambiguous repairs of near-miss source IDs: whitespace, `ch10:` -> `ch010:`."""
    out = []
    for x in ids:
        if isinstance(x, str):
            x = x.strip()
            m = ID_FIX.match(x)
            if m and len(m.group(1)) < 3:
                x = f"ch{int(m.group(1)):03d}:" + x[m.end():]
        out.append(x)
    return out
CITE = re.compile(r"\[Chs?\. ([^\]]+)\]")


def source_entries(ledgers, character=None):
    """One compact JSON object per note, in chapter order. Keys: id (source_id), ch (book chapter
    label), sec, who (supporting cast only), claim, ctx, tf (time frame; omitted when
    chapter_present), when (time anchor), ep (epistemic; omitted when narrated_fact), by
    (claimed_by), sc (state change; omitted when none)."""
    lines, ids = [], {}
    for lg in ledgers:
        c = lg["chapter_index"]; lab = lg["chapter_label"]
        if character is None:
            notes = [("supporting_cast", n["character"], n["note"], n) for n in lg["other_character_information"]]
        else:
            rec = next(r for r in lg["principal_characters"] if r["character"] == character)
            sid = f"ch{c:03d}:presence:000"
            ids[sid] = c
            frame = lg["chapter_frame"] if "segments" not in lg["chapter_frame"] else lg["chapter_frame"]["segments"][0]
            lines.append(json.dumps({"id": sid, "ch": lab, "presence": rec["presence"], "narration": frame.get("narration", ""),
                                     "pov": frame.get("pov_character", ""), "setting": frame.get("time_setting", "")}, ensure_ascii=False))
            notes = [(s, character, n["claim"], n) for s in SECTIONS for n in rec["sections"][s]]
        counter = collections.Counter()
        for section, owner, statement, n in notes:
            counter[section] += 1
            sid = f"ch{c:03d}:{section}:{counter[section]:03d}"
            ids[sid] = c
            e = {"id": sid, "ch": lab, "sec": section}
            if character is None:
                e["who"] = owner
            e["claim"] = statement
            if n["context"].strip():
                e["ctx"] = n["context"]
            if n["time_frame"] != "chapter_present":
                e["tf"] = n["time_frame"]
            e["when"] = n["time_anchor"]
            if n["epistemic"] != "narrated_fact":
                e["ep"] = n["epistemic"]
            if n["claimed_by"].strip():
                e["by"] = n["claimed_by"]
            if n["state_change"] != "none":
                e["sc"] = n["state_change"]
            lines.append(json.dumps(e, ensure_ascii=False))
    return "\n".join(lines), ids


def section_list():
    return "\n".join(f"- `{s}`: {SECTION_PURPOSES[s]}" for s in SECTIONS)


def principal_messages(character, entries, last_label, next_label, book):
    schema = ",\n".join(f'    "{s}": [{{"statement": "one self-contained character note", "source_ids": ["one or more supplied source IDs"]}}]' for s in SECTIONS)
    return [
        {"role": "system", "content": "Create accurate, comprehensive character notes for a novelist. Return only the requested JSON object."},
        {"role": "user", "content": (
            f"# Source-reviewed entries for **{character}** (one JSON object per line, in chapter order)\n\n"
            "Keys: `id` source ID; `ch` the book's own chapter label; `sec` section; `claim` the reviewed note; `ctx` "
            "necessary context; `tf` time frame (absent = the chapter's present; else flashback_or_backstory, "
            "dream_or_vision, hypothetical_or_future, unclear); `when` time anchor; `ep` epistemic status (absent = "
            "narrated fact; else character_claim, character_belief, deception_or_irony, uncertain); `by` who claims or "
            "believes it; `sc` state change (death, departure, arrival_or_return, absence, injury_or_incapacity, "
            "status_change). A `presence` line per chapter says whether the character was present, narrating, or absent.\n\n"
            f"{entries}\n\n"
            f"# Task\n\nBuild the cumulative character sheet for **{character}** from the entries above, which cover every "
            f"chapter of `{book}` through Chapter {last_label}. The novelist will use it to write Chapter {next_label}. "
            f"Write the sheet as of the END of Chapter {last_label}. The sheet will be shown beside a separate plot memory, "
            "so focus on information distinctive to this character rather than retelling events.\n\n"
            "Consolidate repeated evidence into concrete, self-contained notes. Preserve changes over time, meaningful "
            "exceptions, and uncertainty. This is synthesis, not source-entry coverage: do not write one note per chapter "
            "or per entry; prefer fewer, richer notes that combine genuinely related evidence, but do not erase "
            "distinctive character detail. Summarize stable voice patterns instead of cataloguing quotations. Keep only "
            "durable or currently useful knowledge. Put only active, recurring, or character-defining aims under goals; a "
            "completed errand is not a goal. Put only genuine interpersonal dynamics under relationships. Omit routine "
            "completed actions, isolated physical motions, and generic plot recap.\n\n"
            "# Accuracy rules (each entry carries tags; they are binding)\n\n"
            "1. Epistemic status is part of the fact. An entry tagged `character_claim`, `character_belief` or "
            "`deception_or_irony` must be written as who says, believes, lies about or jokes about it (`Storey tells Jess "
            "that ...`, `Jess believes ...`, `Her cover story is that ...; in fact ...`), never as an established fact. "
            "Only `narrated_fact` entries may be stated flatly. Keep `uncertain` entries hedged.\n"
            "2. Time frame is part of the fact. `flashback_or_backstory` entries are past events and must say so with "
            "their anchor (`Before the story, ...`, `As a child, ...`); `dream_or_vision` entries are dreams; "
            "`hypothetical_or_future` entries are plans, fears or imaginings, never accomplished facts. Never turn a "
            "memory or dream into a present-tense situation.\n"
            "3. State changes are hard updates. Death, departure, arrival, absence, injury, and status or relationship "
            "changes (tag `state_change`) define the character's situation as of the LATEST chapter that establishes "
            "them: state the current situation with its chapter (`Dead as of Ch. 7, shot by ...`; `Left the group in Ch. "
            "4 and has not returned`), and put roles, bonds or circumstances that ended into the past tense. When "
            "entries conflict, later chapters win and the change is described as a change, not as two facts.\n"
            "4. Attribution is fixed by the entry's `character`; never import another character's trait, speech, "
            "knowledge or relationship. Name the other party explicitly in relationship notes.\n"
            "5. Never use words like `now` or `currently` unless the latest entries support them; a temporary situation "
            "from an earlier chapter is historical and must be labelled with its chapter.\n\n"
            "Every output note must cite one or more exact supplied `source_id` values, copied in full (never shorten "
            "`ch003:physicality_and_personality:004` to `ch003:004`). Do not invent source IDs, quotations, or facts "
            "absent from the entries. A section may be empty. Across all statements, use no more than "
            f"{REQUESTED_PRINCIPAL_WORDS} words.\n\n"
            f"# Sections\n\n{section_list()}\n\n"
            "# Output format\n\nReturn exactly:\n\n"
            "{\n"
            f'  "character": {json.dumps(character, ensure_ascii=False)},\n'
            '  "sections": {\n'
            f"{schema}\n"
            "  }\n"
            "}"
        )},
    ]



def _check_items(items, valid_ids, where):
    words = 0
    for it in items:
        if not isinstance(it, dict) or set(it) != {"statement", "source_ids"}:
            raise ValueError(f"{where}: note keys must be statement, source_ids")
        if not isinstance(it["statement"], str) or not it["statement"].strip():
            raise ValueError(f"{where}: empty statement")
        if LEAK_TOKEN.search(it["statement"]):
            raise ValueError(f"{where}: the statement text contains a source ID ({LEAK_TOKEN.search(it['statement']).group(0)}); source IDs belong ONLY in source_ids, never in the prose")
        ids = it["source_ids"]
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
            raise ValueError(f"{where}: source_ids must be a nonempty list of strings")
        ids = it["source_ids"] = normalize_ids(ids)
        unknown = sorted(set(ids) - set(valid_ids))
        if unknown:
            raise ValueError(f"{where}: unknown source IDs {unknown[:8]}; copy IDs exactly from the supplied entries")
        if len(set(ids)) != len(ids):
            raise ValueError(f"{where}: repeated source ID")
        words += len(it["statement"].split())
    return words


def validate_principal(payload, character, valid_ids):
    if not isinstance(payload, dict) or set(payload) != {"character", "sections"}:
        raise ValueError("keys must be character, sections")
    if payload["character"] != character:
        raise ValueError("character changed")
    if not isinstance(payload["sections"], dict) or set(payload["sections"]) != set(SECTIONS):
        raise ValueError(f"sections must be exactly {SECTIONS}")
    words = 0
    for s, items in payload["sections"].items():
        if not isinstance(items, list):
            raise ValueError(f"section {s} must be a list")
        words += _check_items(items, valid_ids, s)
    if words > MAX_PRINCIPAL_WORDS:
        raise ValueError(f"sheet has {words} words; maximum is {MAX_PRINCIPAL_WORDS}")



def citation(source_ids, id_to_chapter, labels_for_book):
    chs = sorted({id_to_chapter[s] for s in source_ids})
    labs = [labels_for_book[c] for c in chs]
    return ("[Ch. " if len(labs) == 1 else "[Chs. ") + ", ".join(labs) + "]"


def render_principal(payload, id_to_chapter, labels_for_book):
    lines = []
    for s in SECTIONS:
        lines.extend((f"## {s.replace('_', ' ').title()}", ""))
        items = payload["sections"][s]
        if items:
            lines.extend(f"- {it['statement'].strip()} {citation(it['source_ids'], id_to_chapter, labels_for_book)}" for it in items)
        else:
            lines.append("_No established information._")
        lines.append("")
    return "\n".join(lines).strip()


def make_repair(valid_ids):
    """Last-attempt repair: drop source IDs that do not exist; drop statements left without any."""
    def strip_leaks(it):
        t = it.get("statement", "")
        if LEAK_TOKEN.search(t):
            t = re.sub(r"\s*\((?:[^()]*\bch\d{1,3}:[^()]*)\)", "", t)      # parenthetical id groups
            t = LEAK_TOKEN.sub("", t)
            t = re.sub(r"\s{2,}", " ", t).replace(" ,", ",").replace(" .", ".").strip()
            it["statement"] = t
    def repair(payload, error):
        dropped_ids, dropped_stmts = 0, 0
        for items in (payload.get("sections") or {}).values():
            for it in items:
                if isinstance(it, dict): strip_leaks(it)
        for ch in (payload.get("characters") or []):
            for it in ch.get("notes", []):
                if isinstance(it, dict): strip_leaks(it)
        if "sections" in payload:
            for s, items in payload["sections"].items():
                keep = []
                for it in items:
                    ids = [x for x in normalize_ids(it.get("source_ids", [])) if x in valid_ids]
                    dropped_ids += len(it.get("source_ids", [])) - len(ids)
                    if ids:
                        it["source_ids"] = ids; keep.append(it)
                    else:
                        dropped_stmts += 1
                payload["sections"][s] = keep
        elif "characters" in payload:
            chars = []
            for ch in payload["characters"]:
                keep = []
                for it in ch.get("notes", []):
                    ids = [x for x in normalize_ids(it.get("source_ids", [])) if x in valid_ids]
                    dropped_ids += len(it.get("source_ids", [])) - len(ids)
                    if ids:
                        it["source_ids"] = ids; keep.append(it)
                    else:
                        dropped_stmts += 1
                if keep:
                    ch["notes"] = keep; chars.append(ch)
            payload["characters"] = chars
        return {"dropped_ids": dropped_ids, "dropped_statements": dropped_stmts, "error": error[:300]}
    return repair


# ---- end COPIED ----
