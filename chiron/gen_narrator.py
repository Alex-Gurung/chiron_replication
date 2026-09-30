"""Who narrates each chapter in the first person, with gpt-oss-120b, for the chapter-heading/narrator hints
(common.chapter_context). One call per chapter before the last boundary the passages need.

  python3 chiron/gen_narrator.py --books B... [--workers 64]
The model sees the chapter heading, the chapter text and the principals' names, and returns whether the chapter is
told in the first person and by whom (a principal's name as listed, another name, or "" when the text never says).
Output: outputs/narrator/<book>.jsonl. Resumable.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import llm
from common import DATA, NARRATOR, append_jsonl, clean_text, load_chapters, read_jsonl


def messages(ch, names):
    head = " ".join((ch.get("chapter_header") or "").split())
    return [{"role": "system", "content": "You are a careful reader of novels."},
            {"role": "user", "content": (
                f"Chapter heading: {head or '(none)'}\n\nChapter text:\n{clean_text(ch['chapter_text_normalized'])}\n\n"
                f"Main characters of the novel: {', '.join(names)}.\n\n"
                "Is this chapter narrated in the first person (the narrator says \"I\")? If so, who is the narrator? Use "
                "the chapter heading and the text (for example, other characters addressing the narrator by name). If the "
                "narrator is one of the main characters, give that name exactly as listed; otherwise give the name used in "
                "the text; give an empty string if the chapter never makes it clear.\n\n"
                'Return only JSON: {"first_person": true or false, "narrator": "..."}')}]


def check(p):
    if set(p) != {"first_person", "narrator"} or type(p["first_person"]) is not bool or not isinstance(p["narrator"], str):
        raise ValueError('expected {"first_person": bool, "narrator": string}')
    return {"first_person": p["first_person"], "narrator": p["narrator"].strip() if p["first_person"] else ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--books", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    assert llm.server_up(), "gpt-oss server not reachable"
    principals = json.load(open(DATA / "principals.json"))
    chapters = load_chapters()
    last = {}
    for s in ("test", "val", "train"):
        for it in read_jsonl(DATA / f"items_{s}.jsonl"):
            last[it["book"]] = max(last.get(it["book"], 0), it["chapter_index"])
    NARRATOR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for b in args.books:
        done = {r["chapter_index"] for r in read_jsonl(NARRATOR / f"{b}.jsonl")}
        jobs += [(b, ch) for ch in chapters[b] if ch["chapter_index"] < last.get(b, 0) and ch["chapter_index"] not in done]
    print(len(jobs), "chapters to do", flush=True)

    def one(job):
        b, ch = job
        names = [principals[b]["gen_names"][l][ch["chapter_index"]] for l in principals[b]["labels"]]
        try:
            p, meta = llm.ask(messages(ch, names), check, max_tokens=6000)
        except ValueError as e:
            print("FAILED", b, ch["chapter_index"], str(e)[:200], flush=True)
            return
        append_jsonl(NARRATOR / f"{b}.jsonl", {"book": b, "chapter_index": ch["chapter_index"], **p, **meta})
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(one, jobs))


if __name__ == "__main__":
    main()
