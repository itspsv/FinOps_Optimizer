import httpx
import respx
from fastapi.testclient import TestClient

from tests.fakes import InMemoryCacheStore

COMPLETION = {
    "id": "chatcmpl-upstream",
    "object": "chat.completion",
    "created": 1,
    "model": "test-premium-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Go to Settings > Security."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
}


def _ask(client: TestClient, text: str, **extra) -> httpx.Response:
    headers = extra.pop("headers", {})
    body = {"model": "gpt-4o", "messages": [{"role": "user", "content": text}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=headers)


def test_repeat_question_is_served_from_cache(
    cached_client: TestClient, upstream: respx.MockRouter, cache_store: InMemoryCacheStore
):
    route = upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=COMPLETION)
    )

    first = _ask(cached_client, "How do I reset my password?")
    second = _ask(cached_client, "How do I reset my password?")
    paraphrase = _ask(cached_client, "how do I reset my password")

    assert route.call_count == 1
    assert first.headers["x-finops-cache"] == "miss"
    assert first.headers["x-finops-route"] == "premium"
    assert second.headers["x-finops-cache"] == "hit_exact"
    assert second.headers["x-finops-route"] == "cache"
    assert paraphrase.headers["x-finops-cache"] == "hit_semantic"
    assert float(paraphrase.headers["x-finops-cache-similarity"]) >= 0.8
    content = second.json()["choices"][0]["message"]["content"]
    assert content == "Go to Settings > Security."
    assert second.json()["id"] != COMPLETION["id"]
    assert cache_store.rows[0].hits == 2


def test_cache_hit_can_be_streamed(cached_client: TestClient, upstream: respx.MockRouter):
    upstream.post("/chat/completions").mock(return_value=httpx.Response(200, json=COMPLETION))
    _ask(cached_client, "How do I reset my password?")

    response = _ask(cached_client, "How do I reset my password?", stream=True)

    assert response.headers["x-finops-route"] == "cache"
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "Go to Settings > Security." in response.text
    assert response.text.endswith("data: [DONE]\n\n")


def test_bypass_header_forces_upstream(cached_client: TestClient, upstream: respx.MockRouter):
    route = upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=COMPLETION)
    )
    _ask(cached_client, "How do I reset my password?")

    response = _ask(
        cached_client,
        "How do I reset my password?",
        headers={"x-finops-cache-control": "bypass"},
    )

    assert response.headers["x-finops-cache"] == "bypass"
    assert route.call_count == 2


def test_upstream_errors_are_not_cached(
    cached_client: TestClient, upstream: respx.MockRouter, cache_store: InMemoryCacheStore
):
    upstream.post("/chat/completions").mock(
        return_value=httpx.Response(429, json={"error": {"message": "slow down"}})
    )

    _ask(cached_client, "How do I reset my password?")

    assert cache_store.rows == []


def test_cache_outage_still_answers(
    cached_client: TestClient, upstream: respx.MockRouter, cache_store: InMemoryCacheStore
):
    upstream.post("/chat/completions").mock(return_value=httpx.Response(200, json=COMPLETION))
    cache_store.fail = True

    response = _ask(cached_client, "How do I reset my password?")

    assert response.status_code == 200
    assert response.headers["x-finops-cache"] == "error"
    assert response.headers["x-finops-route"] == "premium"
