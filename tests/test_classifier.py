import pytest

from finops.core.config import RoutingSettings
from finops.router import Tier, classify
from finops.router.classifier import has_supplied_material

SETTINGS = RoutingSettings(local_policy="permissive")  # v1 rules
V2 = RoutingSettings(local_policy="grounded")


def _ask(text: str, **params) -> dict:
    return {"messages": [{"role": "user", "content": text}], **params}


@pytest.mark.parametrize(
    "text",
    [
        "Classify the sentiment as positive or negative: I love this product.",
        "Translate 'good morning' into Spanish.",
        "Summarize this in one sentence: The meeting moved to Friday.",
        "What is the capital of France?",
        "Fix the grammar: me and him goes to school.",
        "Give me a title for a blog post about houseplants.",
        "Is a computer faster than a calculator? Yes or no.",
        "What happened on 2024-10-26?",
    ],
)
def test_simple_prompts_go_local(text: str):
    assert classify(_ask(text), SETTINGS).tier is Tier.LOCAL


@pytest.mark.parametrize(
    ("text", "signal"),
    [
        ("Write a Python function that merges two sorted lists.", "code"),
        ("Why does this fail?\n```\nx = [1, 2\n```", "code_block"),
        ("def add(a, b): return a + b  -- what's wrong here?", "code_syntax"),
        ("What is 37 * 49 + 12?", "math_expression"),
        ("Prove that the square root of 2 is irrational.", "math"),
        ("Calculate the probability of rolling two sixes.", "math"),
    ],
)
def test_code_and_math_go_premium(text: str, signal: str):
    decision = classify(_ask(text), SETTINGS)
    assert decision.tier is Tier.PREMIUM
    assert signal in decision.signals


def test_reasoning_alone_stays_local_but_adds_up():
    one_signal = classify(_ask("Compare cats and dogs as pets."), SETTINGS)
    two_signals = classify(
        _ask("Compare these two architecture options and analyze the trade-offs."), SETTINGS
    )
    assert one_signal.tier is Tier.LOCAL
    assert two_signals.tier is Tier.PREMIUM


def test_simple_task_offsets_one_complexity_signal():
    decision = classify(_ask("Summarize this design document in three bullets."), SETTINGS)
    assert decision.tier is Tier.LOCAL
    assert decision.reason == "simple_task"


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (_ask("hi", tools=[{"type": "function"}]), "tools"),
        (
            {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {}}]}]},
            "multimodal",
        ),
        (_ask("hi", response_format={"type": "json_schema"}), "structured_output"),
        (_ask("word " * 2000), "prompt_too_long"),
        (_ask("hi", max_tokens=4000), "long_output_requested"),
    ],
)
def test_hard_limits_force_premium(payload: dict, reason: str):
    decision = classify(payload, SETTINGS)
    assert decision.tier is Tier.PREMIUM
    assert decision.reason == reason


def test_long_conversation_counts_as_complexity():
    history = [{"role": "user", "content": "hello"}] * 7
    decision = classify({"messages": history}, SETTINGS)
    assert "long_conversation" in decision.signals


def test_client_override_wins():
    code = _ask("Write a Python function.")
    assert classify(code, SETTINGS, override="local").tier is Tier.LOCAL
    assert classify(_ask("hi"), SETTINGS, override="premium").reason == "client_override"


# --- v2 "grounded" policy -------------------------------------------------------------

PASSAGE = (
    "Virgin Australia commenced services on 31 August 2000 as Virgin Blue, with two "
    "aircraft on a single route. It suddenly found itself a major airline in Australia's "
    "domestic market after the collapse of Ansett Australia in September 2001."
)


@pytest.mark.parametrize(
    "text",
    [
        "Classify the sentiment as positive or negative: I love this product.",
        "Translate 'good morning' into Spanish.",
        f"When did Virgin Australia start operating?\n\n{PASSAGE}",
        f"Extract the year the airline was founded.\n\n{PASSAGE}",
        "What is the capital of Japan? Answer in one word.",
    ],
)
def test_v2_keeps_grounded_and_transform_tasks_local(text: str):
    decision = classify(_ask(text), V2)
    assert decision.tier is Tier.LOCAL, decision


@pytest.mark.parametrize(
    ("text", "signal"),
    [
        ("Is a PhD in chemistry worth pursuing?", "open_knowledge"),
        ("What exactly is the West Bank?", "open_knowledge"),
        ("Brainstorm ten names for a coffee shop.", "open_knowledge"),
        (
            "Janet's ducks lay 16 eggs per day. She eats three and sells the rest for $2 "
            "each. How much does she make every day?",
            "math_word_problem",
        ),
    ],
)
def test_v2_sends_open_questions_and_word_problems_premium(text: str, signal: str):
    decision = classify(_ask(text), V2)
    assert decision.tier is Tier.PREMIUM
    assert signal in decision.signals


def test_short_trailer_after_blank_line_is_not_a_passage():
    assert not has_supplied_material("What is 2 apples plus 3?\n\nAnswer with a number.")
    assert has_supplied_material(f"Question?\n\n{PASSAGE}")


def test_v1_policy_is_unchanged_by_v2_rules():
    assert classify(_ask("Is a PhD in chemistry worth pursuing?"), SETTINGS).tier is Tier.LOCAL
