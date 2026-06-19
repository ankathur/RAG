"""Robust structured-output helper (see SPEC.md §6).

OpenAI-compatible endpoints differ in how (and whether) they support constrained
JSON output. This module tries, in order, the strongest mechanism a given
``mode`` allows and degrades gracefully:

    json_schema -> json_object -> prompt-instructed JSON  (+ one repair retry)

The result is always parsed into a plain ``dict``; schema *validation* beyond
"is a JSON object" is left to the caller (which coerces via pydantic).
"""

from __future__ import annotations

import json
from typing import Any

from rag.llm.base import StructuredOutputError

# Attempt order per configured mode.
_ATTEMPTS: dict[str, list[str]] = {
    "auto": ["json_schema", "json_object", "prompt"],
    "json_schema": ["json_schema", "json_object", "prompt"],
    "json_object": ["json_object", "prompt"],
    "prompt": ["prompt"],
}


def request_structured(
    *,
    client: Any,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    mode: str = "auto",
    system: str | None = None,
    schema_name: str = "result",
) -> dict[str, Any]:
    """Call the chat-completions endpoint and return a parsed JSON object."""
    attempts = _ATTEMPTS.get(mode, _ATTEMPTS["auto"])
    last_text = ""
    last_error: Exception | None = None

    for kind in attempts:
        messages = _messages(prompt, schema, system, kind)
        response_format = _response_format(kind, schema, schema_name)
        try:
            text = _chat(client, model, messages, response_format)
        except Exception as exc:  # unsupported response_format, etc. -> try next
            last_error = exc
            continue
        last_text = text
        data = _parse_json(text)
        if data is not None:
            return data

    # One repair retry: hand the model its own broken output and ask for valid JSON.
    if last_text:
        repaired = _repair(client, model, last_text, schema)
        if repaired is not None:
            return repaired

    raise StructuredOutputError(
        f"Could not obtain valid JSON from model {model!r}. "
        f"Last output: {last_text[:500]!r}"
    ) from last_error


def _messages(
    prompt: str, schema: dict[str, Any], system: str | None, kind: str
) -> list[dict[str, str]]:
    sys = system or "You are a precise assistant that returns only what is requested."
    user = prompt
    if kind in ("json_object", "prompt"):
        # Endpoints that lack json_schema still need the schema described inline.
        user = (
            f"{prompt}\n\nReturn ONLY a JSON object that conforms to this JSON Schema. "
            f"Do not include any prose or markdown fences.\n\nJSON Schema:\n"
            f"{json.dumps(schema)}"
        )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _response_format(
    kind: str, schema: dict[str, Any], schema_name: str
) -> dict[str, Any] | None:
    if kind == "json_schema":
        return {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "schema": schema, "strict": False},
        }
    if kind == "json_object":
        return {"type": "json_object"}
    return None


def _chat(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, Any] | None,
) -> str:
    kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
    if response_format is not None:
        kwargs["response_format"] = response_format
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def _repair(
    client: Any, model: str, broken: str, schema: dict[str, Any]
) -> dict[str, Any] | None:
    messages = [
        {
            "role": "system",
            "content": "You fix malformed JSON. Return only valid JSON, no commentary.",
        },
        {
            "role": "user",
            "content": (
                "The following text should be a JSON object matching this schema:\n"
                f"{json.dumps(schema)}\n\nText:\n{broken}\n\n"
                "Return the corrected JSON object only."
            ),
        },
    ]
    try:
        text = _chat(client, model, messages, {"type": "json_object"})
    except Exception:
        try:
            text = _chat(client, model, messages, None)
        except Exception:
            return None
    return _parse_json(text)


def _parse_json(text: str) -> dict[str, Any] | None:
    """Tolerantly extract the first JSON object from model output."""
    if not text:
        return None
    cleaned = text.strip()
    # Strip markdown code fences if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1] if "```" in cleaned[3:] else cleaned[3:]
        if cleaned.lstrip().startswith("json"):
            cleaned = cleaned.lstrip()[4:]
    cleaned = cleaned.strip()

    # Fast path.
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # Fallback: scan for the first balanced {...} block.
    start = cleaned.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(cleaned[start : i + 1])
                    return obj if isinstance(obj, dict) else None
                except json.JSONDecodeError:
                    return None
    return None
