"""Cheap checks on a local answer before we trust it. A failure escalates to premium."""

import json
import re
from typing import Any

_REFUSAL = re.compile(
    r"^\s*(?:i'?m sorry,? but|i (?:can(?:not|'t)|am unable to|'m unable to) "
    r"(?:help|assist|answer|provide)|as an ai(?: language model)?)",
    re.IGNORECASE,
)


def validate_local_answer(response: dict[str, Any], payload: dict[str, Any]) -> str | None:
    """Return why the answer is unusable, or None if it looks fine."""
    choices = response.get("choices") or []
    if not choices:
        return "no_choices"
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        return "truncated"
    content = ((choice.get("message") or {}).get("content") or "").strip()
    if not content:
        return "empty"
    if _REFUSAL.search(content.replace("’", "'")):
        return "refusal"
    if (payload.get("response_format") or {}).get("type") == "json_object":
        try:
            json.loads(content)
        except ValueError:
            return "invalid_json"
    return None
