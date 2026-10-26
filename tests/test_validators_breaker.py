import pytest

from finops.providers.breaker import CircuitBreaker
from finops.router.validators import validate_local_answer


def _answer(content: str | None, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {"finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}
        ]
    }


@pytest.mark.parametrize(
    ("response", "payload", "problem"),
    [
        (_answer("Paris"), {}, None),
        ({"choices": []}, {}, "no_choices"),
        (_answer("Once upon a", finish_reason="length"), {}, "truncated"),
        (_answer("   "), {}, "empty"),
        (_answer("I'm sorry, but I can't help with that."), {}, "refusal"),
        (_answer("As an AI language model, I don't know."), {}, "refusal"),
        (_answer("I can’t assist with this request."), {}, "refusal"),
        (_answer("not json"), {"response_format": {"type": "json_object"}}, "invalid_json"),
        (_answer('{"ok": true}'), {"response_format": {"type": "json_object"}}, None),
    ],
)
def test_validate_local_answer(response: dict, payload: dict, problem: str | None):
    assert validate_local_answer(response, payload) == problem


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_breaker_opens_after_threshold_and_recovers():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=3, cooldown_s=30, clock=clock)

    for _ in range(3):
        assert breaker.allow()
        breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow()

    clock.now = 31
    assert breaker.state == "half_open"
    assert breaker.allow()  # one trial call
    assert not breaker.allow()  # no second trial while the first is in flight
    breaker.record_success()
    assert breaker.state == "closed"


def test_failed_trial_reopens_breaker():
    clock = FakeClock()
    breaker = CircuitBreaker(failure_threshold=1, cooldown_s=30, clock=clock)
    breaker.record_failure()
    clock.now = 31
    assert breaker.allow()

    breaker.record_failure()

    assert breaker.state == "open"


def test_success_resets_failure_count():
    breaker = CircuitBreaker(failure_threshold=2, cooldown_s=30)
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == "closed"
