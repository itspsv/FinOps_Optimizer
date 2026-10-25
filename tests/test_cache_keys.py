import pytest

from finops.cache.guards import fingerprint
from finops.cache.keys import build_keys, is_storable, skip_reason


def _payload(text: str, **params) -> dict:
    return {"messages": [{"role": "user", "content": text}], **params}


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("What was Q3 2024 revenue?", "What was Q3 2025 revenue?"),
        ("How do I enable dark mode?", "How do I disable dark mode?"),
        ("Is this dish spicy?", "Is this dish not spicy?"),
        ("Why doesn't it compile?", "Why does it compile?"),
        ("Should I buy Tesla stock?", "Should I sell Tesla stock?"),
    ],
)
def test_fingerprint_separates_lookalike_prompts(a: str, b: str):
    assert fingerprint(a) != fingerprint(b)


def test_fingerprint_ignores_wording_changes():
    assert fingerprint("How do I reset my password?") == fingerprint(
        "Steps to change my password, please"
    )
    assert fingerprint("Revenue was 1,200 in 2024") == fingerprint("2024 revenue: 1200")


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (_payload("hi", tools=[{"type": "function"}]), "tools"),
        (_payload("hi", n=3), "multiple_choices"),
        (_payload("hi", temperature=1.5), "temperature"),
        ({"messages": [{"role": "assistant", "content": "hi"}]}, "last_message_not_user"),
        (
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},
            "non_text_content",
        ),
    ],
)
def test_skip_reasons(payload: dict, reason: str):
    assert skip_reason(payload, max_temperature=1.0) == reason


def test_cacheable_request_has_no_skip_reason():
    assert skip_reason(_payload("hi", temperature=0.2), max_temperature=1.0) is None
    text_parts = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    assert skip_reason(text_parts, max_temperature=1.0) is None


def test_context_hash_scopes_by_history_params_and_model():
    base = build_keys(_payload("Summarize it"), "model-a")

    other_history = {
        "messages": [
            {"role": "user", "content": "Here is doc B"},
            {"role": "user", "content": "Summarize it"},
        ]
    }
    assert build_keys(other_history, "model-a").context_hash != base.context_hash
    assert build_keys(_payload("Summarize it", max_tokens=5), "model-a").context_hash != (
        base.context_hash
    )
    assert build_keys(_payload("Summarize it"), "model-b").context_hash != base.context_hash
    # Rewording the last message keeps the context but changes the exact key.
    reworded = build_keys(_payload("Give me a summary"), "model-a")
    assert reworded.context_hash == base.context_hash
    assert reworded.request_hash != base.request_hash


def test_is_storable():
    def response(**choice):
        message = {"role": "assistant", "content": "ok", **choice.pop("message", {})}
        return {"choices": [{"finish_reason": "stop", "message": message, **choice}]}

    assert is_storable(response())
    assert not is_storable(response(finish_reason="length"))
    assert not is_storable(response(message={"content": "  "}))
    assert not is_storable(response(message={"tool_calls": [{"id": "1"}]}))
    assert not is_storable({"choices": []})
