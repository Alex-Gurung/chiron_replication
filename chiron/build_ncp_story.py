"""The NCP dataset's own Story Information parts, one record per (book, chapter): data/ncp_story.jsonl.

  python3 chiron/build_ncp_story.py
From ncp_cohorts_v2 (any split; the fields depend only on the chapter): plot = prior_plot_summary (SuperSummary
synopses of every earlier chapter, in order), raw = story_text (the two preceding chapters), next =
next_chapter_synopsis (SuperSummary's synopsis of the chapter being written: it describes the passage's own chapter,
so it is an oracle here). The cohort has no whole-book overview (high_level_plot_summary is empty).
"""
import json

from common import COHORTS, DATA

seen = {}
for s in ("test", "val", "train"):
    for line in open(COHORTS / f"{s}_examples.jsonl"):
        r = json.loads(line)
        k = (r["story_id"], r["chapter_index"])
        if k not in seen:
            seen[k] = {"book": k[0], "chapter_index": k[1], "plot": r.get("prior_plot_summary") or "",
                       "raw": r.get("story_text") or "", "next": r.get("next_chapter_synopsis") or ""}
with open(DATA / "ncp_story.jsonl", "w") as f:
    for r in seen.values():
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(seen), "chapters")
