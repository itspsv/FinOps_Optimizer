"""Decide whether a request is cacheable and derive its cache keys."""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from finops.cache.guards import fingerprint

# Params that make output non-reusable or tied to side effects.
_TOOL_PARAMS = ("tools", "functions", "tool_choice", "function_call")


@dataclass(frozen=True)
class CacheKeys:
    request_hash: str  # exact match: model + full payload
    context_hash: str  # semantic match scope: model + params + earlier messages
    prompt_text: str  # last user message, which gets embedded
    guard_fingerprint: str


def skip_reason(payload: dict[str, Any], max_temperature: float) -> str | None:
    """Return why this request must not use the cache, or None if it can."""
    if any(payload.get(p) for p in _TOOL_PARAMS):
        return "tools"
    if payload.get("n", 1) != 1:
        return "multiple_choices"
    temperature = payload.get("temperature")
    if temperature is not None and temperature > max_temperature:
        return "temperature"
    last = payload["messages"][-1]
    if last.get("role") != "user":
        return "last_message_not_user"
    if _text_of(last.get("content")) is None:
        return "non_text_content"
    return None


def build_keys(payload: dict[str, Any], upstream_model: str) -> CacheKeys:
    """Assumes skip_reason(payload) returned None."""
    messages = payload["messages"]
    params = {k: v for k, v in payload.items() if k != "messages"}
    prompt_text = _text_of(messages[-1].get("content")) or ""
    return CacheKeys(
        request_hash=_hash({"model": upstream_model, "payload": payload}),
        context_hash=_hash({"model": upstream_model, "params": params, "history": messages[:-1]}),
        prompt_text=prompt_text,
        guard_fingerprint=fingerprint(prompt_text),
    )


def is_storable(response: dict[str, Any]) -> bool:
    """Only cache complete, plain-text answers."""
    choices = response.get("choices") or []
    if len(choices) != 1:
        return False
    choice = choices[0]
    message = choice.get("message") or {}
    return (
        choice.get("finish_reason") == "stop"
        and not message.get("tool_calls")
        and bool((message.get("content") or "").strip())
    )


def _text_of(content: Any) -> str | None:
    """Plain text of a message, or None if it has non-text parts (images, audio)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(p.get("type") == "text" for p in content):
        return "\n".join(p.get("text", "") for p in content)
    return None


def _hash(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
