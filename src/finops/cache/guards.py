"""Cheap lexical checks that stop embeddings from merging prompts with different meanings.

Embeddings score "Q3 2024 revenue" vs "Q3 2025 revenue" or "how do I enable X" vs
"how do I disable X" as near-identical. Two prompts may only share a cache entry if
their fingerprints match exactly, so these cases always miss.
"""

import re

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

_NEGATIONS = frozenset(
    {"not", "no", "never", "none", "nothing", "without", "cannot", "nor", "neither"}
)

# Words whose opposite often embeds almost identically. Each maps to a canonical
# label, so "turn on" style synonyms aren't needed for the common cases.
_POLARITY = {
    "enable": "enable",
    "enabled": "enable",
    "disable": "disable",
    "disabled": "disable",
    "add": "add",
    "remove": "remove",
    "delete": "remove",
    "increase": "increase",
    "decrease": "decrease",
    "install": "install",
    "uninstall": "uninstall",
    "start": "start",
    "stop": "stop",
    "open": "open",
    "close": "close",
    "buy": "buy",
    "sell": "sell",
    "before": "before",
    "after": "after",
    "max": "max",
    "maximum": "max",
    "min": "min",
    "minimum": "min",
    "best": "best",
    "worst": "worst",
    "first": "first",
    "last": "last",
    "true": "true",
    "false": "false",
}


def fingerprint(text: str) -> str:
    lowered = text.lower().replace("’", "'")  # curly apostrophe, as in "don’t"
    numbers = sorted({n.replace(",", "") for n in _NUMBER.findall(lowered)})
    words = _WORD.findall(lowered)
    negated = any(w in _NEGATIONS or w.endswith("n't") for w in words)
    polarity = sorted({_POLARITY[w] for w in words if w in _POLARITY})
    return f"num={','.join(numbers)}|neg={int(negated)}|pol={','.join(polarity)}"
