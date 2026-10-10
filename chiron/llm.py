"""OpenAI-compatible chat client for the local gpt-oss server, with validation + retry.

API_BASE / MODEL come from the env set by scripts/serve_and_run.py. `ask` returns
(payload, meta) or raises after ATTEMPTS; retries re-prompt with the validation error and
raise the temperature so a deterministic failure does not repeat.
"""
import json
import os
import re
import time
from urllib import error as urlerror, request as urlrequest

API_BASE = os.environ.get("CHIRON_API_BASE", "http://127.0.0.1:8000/v1")
MODEL = os.environ.get("CHIRON_MODEL", "openai/gpt-oss-120b")
REASONING_EFFORT = "medium"
ATTEMPTS = 3


def chat(messages, max_tokens, temperature, timeout=3600, top_p=1.0, effort=None, stop=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p}
    if "gpt-oss" in MODEL:
        body["reasoning_effort"] = effort or REASONING_EFFORT
    if stop:
        body["stop"] = stop
    body = json.dumps(body).encode()
    req = urlrequest.Request(API_BASE + "/chat/completions", data=body,
                             headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    last = None
    for k in range(6):
        try:
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except (urlerror.URLError, ConnectionError, TimeoutError, OSError) as e:
            last = e
            time.sleep(min(60, 5 * (k + 1)))
    raise RuntimeError(f"chat request failed: {last}")


def parse_json(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i < 0 or j <= i:
            raise ValueError("no JSON object in response")
        return json.loads(t[i:j + 1])


def ask(messages, validate, max_tokens=8000, temperature=0.1, as_json=True):
    """validate(payload) returns the cleaned payload or raises ValueError."""
    errors, msgs, temp, usage = [], messages, temperature, {"prompt_tokens": 0, "completion_tokens": 0}
    for n in range(1, ATTEMPTS + 1):
        resp = chat(msgs, max_tokens, temp)
        choice = resp["choices"][0]
        for k in usage:
            usage[k] += (resp.get("usage") or {}).get(k) or 0
        text = (choice["message"].get("content") or "").strip()
        try:
            if choice.get("finish_reason") != "stop":
                raise ValueError(f"finish_reason={choice.get('finish_reason')}")
            if not text:
                raise ValueError("empty response")
            payload = validate(parse_json(text) if as_json else text)
            return payload, {"attempts": n, **usage}
        except (ValueError, KeyError, TypeError) as e:
            errors.append(f"attempt {n}: {str(e)[:300]}")
            temp = max(temp, 0.6)
            msgs = [*messages, {"role": "assistant", "content": text[:20000]},
                    {"role": "user", "content": f"Your previous response failed validation: {str(e)[:800]}. "
                                                "Return a corrected, complete response in exactly the requested format."}]
    raise ValueError(" | ".join(errors))


def server_up():
    try:
        with urlrequest.urlopen(API_BASE + "/models", timeout=10) as r:
            return r.status == 200 and any(m.get("id") == MODEL for m in json.load(r).get("data", []))
    except Exception:
        return False
