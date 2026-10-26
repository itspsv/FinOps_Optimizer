"""End-to-end routing through the proxy, with Groq and Ollama both mocked."""

import json

import httpx
import respx
from fastapi.testclient import TestClient

from tests.fakes import LOCAL_URL, PREMIUM_URL, InMemoryCacheStore

SIMPLE = "Classify the sentiment as positive or negative: I love this product."
COMPLEX = "Write a Python function that merges two sorted lists."


def _completion(content: str, model: str, finish_reason: str = "stop") -> dict:
    return {
        "id": "chatcmpl-x",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
    }


LOCAL_OK = _completion("positive", "test-local-model")
PREMIUM_OK = _completion("Positive.", "test-premium-model")


def _ask(client: TestClient, text: str, headers: dict | None = None, **extra) -> httpx.Response:
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": text}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=headers or {})


def _mock(upstream: respx.MockRouter, local=None, premium=None):
    local_route = upstream.post(LOCAL_URL).mock(
        **(
            {"side_effect": local}
            if isinstance(local, Exception)
            else {"return_value": local or httpx.Response(200, json=LOCAL_OK)}
        )
    )
    premium_route = upstream.post(PREMIUM_URL).mock(
        return_value=premium or httpx.Response(200, json=PREMIUM_OK)
    )
    return local_route, premium_route


def test_simple_prompt_is_answered_locally(routed_client: TestClient, upstream: respx.MockRouter):
    local, premium = _mock(upstream)

    response = _ask(routed_client, SIMPLE)

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "positive"
    assert response.headers["x-finops-route"] == "local"
    assert response.headers["x-finops-provider"] == "ollama"
    assert response.headers["x-finops-route-reason"] == "simple_task"
    assert "x-finops-escalated" not in response.headers
    assert local.call_count == 1 and premium.call_count == 0
    sent = json.loads(local.calls.last.request.content)
    assert sent["model"] == "test-local-model"
    assert sent["max_tokens"] == 512  # capped for the slow local model
    assert sent["messages"][0]["role"] == "system"  # local-only conciseness hint
    assert sent["messages"][1:] == [{"role": "user", "content": SIMPLE}]


def test_complex_prompt_goes_to_premium(routed_client: TestClient, upstream: respx.MockRouter):
    local, premium = _mock(upstream)

    response = _ask(routed_client, COMPLEX)

    assert response.headers["x-finops-route"] == "premium"
    assert response.headers["x-finops-route-reason"] == "code"
    assert local.call_count == 0 and premium.call_count == 1


def test_bad_local_answer_escalates(routed_client: TestClient, upstream: respx.MockRouter):
    truncated = _completion("posi", "test-local-model", finish_reason="length")
    local, premium = _mock(upstream, local=httpx.Response(200, json=truncated))

    response = _ask(routed_client, SIMPLE)

    assert response.json()["choices"][0]["message"]["content"] == "Positive."
    assert response.headers["x-finops-route"] == "premium"
    assert response.headers["x-finops-escalated"] == "local_truncated"
    assert local.call_count == 1 and premium.call_count == 1
    # The escalated call gets the client's original payload, without local changes.
    escalated = json.loads(premium.calls.last.request.content)
    assert "max_tokens" not in escalated
    assert escalated["messages"] == [{"role": "user", "content": SIMPLE}]


def test_local_down_falls_back_then_breaker_skips_it(
    routed_client: TestClient, upstream: respx.MockRouter
):
    local, premium = _mock(upstream, local=httpx.ConnectError("connection refused"))

    reasons = [_ask(routed_client, SIMPLE).headers["x-finops-escalated"] for _ in range(4)]

    assert reasons == ["local_error_502"] * 3 + ["local_unavailable"]
    assert local.call_count == 3  # breaker opened after 3 failures
    assert premium.call_count == 4
    assert routed_client.get("/health").json()["local_breaker"] == "open"


def test_premium_failure_after_escalation_returns_error(
    routed_client: TestClient, upstream: respx.MockRouter
):
    error = {"error": {"message": "Rate limit reached"}}
    _mock(
        upstream,
        local=httpx.ConnectError("down"),
        premium=httpx.Response(429, json=error),
    )

    response = _ask(routed_client, SIMPLE)

    assert response.status_code == 429
    assert response.json() == error


def test_override_header(routed_client: TestClient, upstream: respx.MockRouter):
    local, premium = _mock(upstream)

    response = _ask(routed_client, SIMPLE, headers={"x-finops-route-override": "premium"})

    assert response.headers["x-finops-route"] == "premium"
    assert response.headers["x-finops-route-reason"] == "client_override"
    assert local.call_count == 0


def test_streamed_simple_prompt_is_replayed_from_local(
    routed_client: TestClient, upstream: respx.MockRouter
):
    local, _ = _mock(upstream)

    response = _ask(routed_client, SIMPLE, stream=True)

    assert response.headers["x-finops-route"] == "local"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"content": "positive"' in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    assert json.loads(local.calls.last.request.content)["stream"] is False


def test_streamed_complex_prompt_passes_through(
    routed_client: TestClient, upstream: respx.MockRouter
):
    sse = b'data: {"choices":[{"delta":{"content":"def"}}]}\n\ndata: [DONE]\n\n'
    _mock(
        upstream,
        premium=httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"}),
    )

    response = _ask(routed_client, COMPLEX, stream=True)

    assert response.headers["x-finops-route"] == "premium"
    assert response.content == sse


def test_local_answers_are_cached(
    routed_cached_client: TestClient, upstream: respx.MockRouter, cache_store: InMemoryCacheStore
):
    local, _ = _mock(upstream)

    first = _ask(routed_cached_client, SIMPLE)
    second = _ask(routed_cached_client, SIMPLE)

    assert first.headers["x-finops-route"] == "local"
    assert second.headers["x-finops-route"] == "cache"
    assert local.call_count == 1
    assert cache_store.rows[0].entry.upstream_model == "test-local-model"
