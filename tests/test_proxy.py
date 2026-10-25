import json

import httpx
import respx
from fastapi.testclient import TestClient

COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "model": "test-premium-model",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
}

REQUEST = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.2,
}


def test_health(client: TestClient):
    assert client.get("/health").json() == {"status": "ok", "cache": "disabled"}


def test_forwards_to_premium_with_configured_model(client: TestClient, upstream: respx.MockRouter):
    route = upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, json=COMPLETION)
    )

    response = client.post("/v1/chat/completions", json=REQUEST)

    assert response.status_code == 200
    assert response.json() == COMPLETION
    assert response.headers["x-finops-route"] == "premium"
    assert response.headers["x-finops-provider"] == "groq"
    assert response.headers["x-finops-request-id"].startswith("req_")
    assert float(response.headers["x-finops-latency-ms"]) >= 0

    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer test-key"
    body = json.loads(sent.content)
    assert body["model"] == "test-premium-model"
    assert body["stream"] is False
    assert body["temperature"] == 0.2
    assert body["messages"] == REQUEST["messages"]


def test_passes_through_upstream_errors(client: TestClient, upstream: respx.MockRouter):
    error = {"error": {"message": "Rate limit reached", "type": "rate_limit_error"}}
    upstream.post("/chat/completions").mock(return_value=httpx.Response(429, json=error))

    response = client.post("/v1/chat/completions", json=REQUEST)

    assert response.status_code == 429
    assert response.json() == error


def test_timeout_returns_504(client: TestClient, upstream: respx.MockRouter):
    upstream.post("/chat/completions").mock(side_effect=httpx.ReadTimeout("slow"))

    response = client.post("/v1/chat/completions", json=REQUEST)

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "upstream_timeout"


def test_unreachable_returns_502(client: TestClient, upstream: respx.MockRouter):
    upstream.post("/chat/completions").mock(side_effect=httpx.ConnectError("refused"))

    response = client.post("/v1/chat/completions", json=REQUEST)

    assert response.status_code == 502
    assert response.json()["error"]["type"] == "upstream_unreachable"


def test_streams_sse_bytes(client: TestClient, upstream: respx.MockRouter):
    sse = (
        b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"!"}}]}\n\n'
        b"data: [DONE]\n\n"
    )
    route = upstream.post("/chat/completions").mock(
        return_value=httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    )

    response = client.post("/v1/chat/completions", json={**REQUEST, "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == sse
    assert json.loads(route.calls.last.request.content)["stream"] is True


def test_stream_upstream_error_keeps_status(client: TestClient, upstream: respx.MockRouter):
    error = {"error": {"message": "Invalid API key", "type": "invalid_request_error"}}
    upstream.post("/chat/completions").mock(return_value=httpx.Response(401, json=error))

    response = client.post("/v1/chat/completions", json={**REQUEST, "stream": True})

    assert response.status_code == 401
    assert response.json() == error


def test_invalid_body_returns_openai_style_400(client: TestClient):
    response = client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": []})

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
