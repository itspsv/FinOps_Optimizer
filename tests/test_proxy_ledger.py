"""Every request path writes exactly one priced ledger row."""

import httpx
import respx
from fastapi.testclient import TestClient

from tests.fakes import LOCAL_URL, PREMIUM_URL, InMemoryLedgerStore
from tests.test_routing import COMPLEX, LOCAL_OK, PREMIUM_OK, SIMPLE, _ask, _completion


def _mock_ok(upstream: respx.MockRouter) -> None:
    upstream.post(LOCAL_URL).mock(return_value=httpx.Response(200, json=LOCAL_OK))
    upstream.post(PREMIUM_URL).mock(return_value=httpx.Response(200, json=PREMIUM_OK))


def test_local_then_cache_rows(
    ledger_client: TestClient, upstream: respx.MockRouter, ledger_store: InMemoryLedgerStore
):
    _mock_ok(upstream)

    _ask(ledger_client, SIMPLE)
    _ask(ledger_client, SIMPLE)

    local, cached = (r.event for r in ledger_store.rows)
    assert (local.route, local.upstream_model, local.decision_reason) == (
        "local",
        "test-local-model",
        "simple_task",
    )
    assert local.local_latency_ms is not None
    assert (cached.route, cached.cache_status) == ("cache", "hit_exact")
    assert cached.prompt_tokens == 20  # tokens the cache saved
    assert all(r.saved_usd > 0 and r.cost_usd == 0 for r in ledger_store.rows)


def test_premium_and_escalation_rows(
    ledger_client: TestClient, upstream: respx.MockRouter, ledger_store: InMemoryLedgerStore
):
    truncated = _completion("posi", "test-local-model", finish_reason="length")
    upstream.post(LOCAL_URL).mock(return_value=httpx.Response(200, json=truncated))
    upstream.post(PREMIUM_URL).mock(return_value=httpx.Response(200, json=PREMIUM_OK))

    _ask(ledger_client, COMPLEX)
    _ask(ledger_client, SIMPLE)

    direct, escalated = (r for r in ledger_store.rows)
    assert direct.event.route == "premium" and direct.event.escalation_reason is None
    assert escalated.event.route == "premium"
    assert escalated.event.decision_tier == "local"
    assert escalated.event.escalation_reason == "local_truncated"
    assert direct.saved_usd == 0  # priced as the baseline model: no saving
    assert direct.billed_usd > 0  # but we know what Groq actually charged


def test_error_row(
    ledger_client: TestClient, upstream: respx.MockRouter, ledger_store: InMemoryLedgerStore
):
    upstream.post(PREMIUM_URL).mock(return_value=httpx.Response(429, json={"error": {}}))

    _ask(ledger_client, COMPLEX)

    (row,) = ledger_store.rows
    assert row.event.status_code == 429
    assert row.baseline_cost_usd == row.cost_usd == 0


def test_streamed_premium_uses_reported_usage(
    ledger_client: TestClient, upstream: respx.MockRouter, ledger_store: InMemoryLedgerStore
):
    sse = (
        b'data: {"model":"test-premium-model","choices":[{"delta":{"content":"def f"}}]}\n\n'
        b'data: {"choices":[],"x_groq":{"usage":{"prompt_tokens":40,"completion_tokens":9}}}\n\n'
        b"data: [DONE]\n\n"
    )
    upstream.post(PREMIUM_URL).mock(
        return_value=httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    )

    response = _ask(ledger_client, COMPLEX, stream=True)

    assert response.content == sse  # the tap never alters the stream
    (row,) = ledger_store.rows
    assert (row.event.prompt_tokens, row.event.completion_tokens) == (40, 9)
    assert row.event.tokens_estimated is False
    assert row.event.upstream_model == "test-premium-model"


def test_streamed_premium_without_usage_is_estimated(
    ledger_client: TestClient, upstream: respx.MockRouter, ledger_store: InMemoryLedgerStore
):
    sse = b'data: {"choices":[{"delta":{"content":"12345678"}}]}\n\ndata: [DONE]\n\n'
    upstream.post(PREMIUM_URL).mock(
        return_value=httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
    )

    _ask(ledger_client, COMPLEX, stream=True)

    (row,) = ledger_store.rows
    assert row.event.tokens_estimated is True
    assert row.event.completion_tokens == 2  # 8 characters ~ 2 tokens
    assert row.event.prompt_tokens > 0


def test_summary_endpoint(
    ledger_client: TestClient, upstream: respx.MockRouter, ledger_store: InMemoryLedgerStore
):
    _mock_ok(upstream)
    _ask(ledger_client, SIMPLE)  # local
    _ask(ledger_client, SIMPLE)  # cache
    _ask(ledger_client, COMPLEX)  # premium

    summary = ledger_client.get("/finops/summary").json()

    assert summary["requests"] == 3
    assert summary["deflection_rate"] == 2 / 3
    assert summary["routes"]["cache"]["requests"] == 1
    assert summary["cost"]["saved_usd"] > 0
    assert summary["pricing"]["baseline_model"] == "baseline"


def test_summary_without_ledger_is_503(client: TestClient):
    assert client.get("/finops/summary").status_code == 503
