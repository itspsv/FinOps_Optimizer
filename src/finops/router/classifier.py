"""Decide whether a request is simple enough for the local model.

Rule-based on purpose: it runs in microseconds and costs nothing, and every
decision comes with readable reasons. Two layers:

1. Hard limits: things the local model can't do well (tools, images, long text).
2. A complexity score from keyword signals; at or above the threshold -> premium.
"""

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from finops.core.config import RoutingSettings


class Tier(StrEnum):
    LOCAL = "local"
    PREMIUM = "premium"


@dataclass(frozen=True)
class RouteDecision:
    tier: Tier
    reason: str
    score: int = 0
    signals: list[str] = field(default_factory=list)
    prompt_tokens_est: int = 0


def _words(*words: str) -> re.Pattern[str]:
    """Whole-word match; list suffixes explicitly, e.g. "comput(?:e|ation)"."""
    return re.compile(r"\b(?:" + "|".join(words) + r")\b", re.IGNORECASE)


# (name, weight, pattern). Positive weights push towards premium.
_SIGNALS: list[tuple[str, int, re.Pattern[str]]] = [
    ("code_block", 2, re.compile(r"```|^\s{4,}\S", re.MULTILINE)),
    (
        "code",
        2,
        _words(
            "code",
            "coding",
            "debug(?:ging)?",
            "refactor",
            "implement(?:ation)?",
            "algorithms?",
            "regex",
            "sql",
            "python",
            "javascript",
            "typescript",
            "java",
            "functions?",
            "stack ?trace",
            "traceback",
            "exception",
            "compiler?",
            "api endpoints?",
        ),
    ),
    ("code_syntax", 2, re.compile(r"\b(?:def|class|import|return|SELECT|const)\b.*[(){}=:;]")),
    # No bare "-" so dates like 2024-10-26 don't count as math.
    ("math_expression", 2, re.compile(r"\d\s*[+*/^=]\s*\d|\d\s+-\s+\d|[∫∑√≤≥]")),
    (
        "math",
        2,
        _words(
            "prove",
            "proof",
            "derivatives?",
            "integrals?",
            "equations?",
            "solve for",
            "probability",
            "calculat(?:e|ion)",
            "comput(?:e|ation)",
            "matri(?:x|ces)",
            "theorem",
            "statistics?",
        ),
    ),
    (
        "reasoning",
        1,
        _words(
            "step by step",
            "analy[sz]e",
            "analysis",
            "compare",
            "comparison",
            "trade-?offs?",
            "design",
            "architecture",
            "strategy",
            "evaluate",
            "pros and cons",
            "explain why",
            "in detail",
            "essay",
            "optimi[sz]e",
            "plan for",
            "critique",
        ),
    ),
    (
        "simple_task",
        -1,
        _words(
            "classify",
            "sentiment",
            "categori[sz]e",
            "label",
            "translate",
            "summari[sz]e",
            "summary",
            "rephrase",
            "rewrite",
            "paraphrase",
            "extract",
            "grammar",
            "spelling",
            "yes or no",
            "one word",
            "tl;?dr",
            "keywords",
            "title for",
        ),
    ),
]

_MAX_HITS_PER_SIGNAL = 2
_LONG_PROMPT_TOKENS = 300
_LONG_CONVERSATION_TURNS = 6


def classify(
    payload: dict[str, Any], settings: RoutingSettings, override: str | None = None
) -> RouteDecision:
    messages = payload["messages"]
    all_text = "\n".join(_text(m.get("content")) for m in messages)
    tokens = estimate_tokens(all_text)

    if override in (Tier.LOCAL, Tier.PREMIUM):
        return RouteDecision(Tier(override), "client_override", prompt_tokens_est=tokens)

    hard = _hard_limit(payload, tokens, settings)
    if hard:
        return RouteDecision(Tier.PREMIUM, hard, prompt_tokens_est=tokens)

    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), messages[-1])
    text = _text(last_user.get("content"))
    score, signals = 0, []
    for name, weight, pattern in _SIGNALS:
        hits = len({m.lower() for m in pattern.findall(text)})
        if hits:
            # Several distinct complexity words count more, up to a cap; the
            # simple-task discount applies once.
            score += weight * (min(hits, _MAX_HITS_PER_SIGNAL) if weight > 0 else 1)
            signals.append(name)
    if estimate_tokens(text) > _LONG_PROMPT_TOKENS:
        score += 1
        signals.append("long_prompt")
    if len(messages) > _LONG_CONVERSATION_TURNS:
        score += 1
        signals.append("long_conversation")

    tier = Tier.PREMIUM if score >= settings.complexity_threshold else Tier.LOCAL
    reason = ",".join(s for s in signals if s != "simple_task") or "simple"
    if tier is Tier.LOCAL and "simple_task" in signals:
        reason = "simple_task"
    return RouteDecision(tier, reason, score, signals, tokens)


def estimate_tokens(text: str) -> int:
    """~4 characters per token for English; close enough for routing limits."""
    return (len(text) + 3) // 4


def _hard_limit(payload: dict[str, Any], tokens: int, settings: RoutingSettings) -> str | None:
    if any(payload.get(k) for k in ("tools", "functions")):
        return "tools"
    if any(_has_non_text(m.get("content")) for m in payload["messages"]):
        return "multimodal"
    if (payload.get("response_format") or {}).get("type") == "json_schema":
        return "structured_output"
    if tokens > settings.local_max_prompt_tokens:
        return "prompt_too_long"
    requested = payload.get("max_completion_tokens") or payload.get("max_tokens")
    if requested and requested > settings.local_max_completion_tokens:
        return "long_output_requested"
    return None


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
    return ""


def _has_non_text(content: Any) -> bool:
    return isinstance(content, list) and any(p.get("type") != "text" for p in content)
