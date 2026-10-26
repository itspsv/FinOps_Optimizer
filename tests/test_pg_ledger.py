"""PgLedgerStore against the finops_test database (skipped if Postgres is unreachable)."""

from collections.abc import AsyncIterator

import psycopg
import pytest

from finops.db.pool import open_pool
from finops.db.schema import apply_schema
from finops.ledger import Ledger, PriceTable, RequestEvent
from finops.ledger.store import PgLedgerStore
from tests.test_pgvector_store import TEST_DB, _ensure_database


@pytest.fixture
async def pg_ledger(prices: PriceTable) -> AsyncIterator[Ledger]:
    await _ensure_database()
    async with await psycopg.AsyncConnection.connect(f"dbname={TEST_DB}") as conn:
        await conn.execute("DROP TABLE IF EXISTS request_log")
        await apply_schema(conn, embedding_dim=4)
    pool = await open_pool(f"dbname={TEST_DB}")
    try:
        yield Ledger(PgLedgerStore(pool), prices)
    finally:
        await pool.close()


def _event(i: int, route: str, latency_ms: float, namespace: str = "a", status: int = 200):
    return RequestEvent(
        request_id=f"req_{i}",
        namespace=namespace,
        requested_model="gpt-4o",
        stream=False,
        status_code=status,
        route=route,
        latency_ms=latency_ms,
        upstream_model={"premium": "test-premium-model", "local": "test-local-model"}.get(route),
        prompt_tokens=1000,
        completion_tokens=100,
        escalation_reason="local_truncated" if i == 4 else None,
    )


async def test_summary_from_postgres(pg_ledger: Ledger):
    events = [
        _event(1, "cache", 5),
        _event(2, "cache", 15),
        _event(3, "local", 500),
        _event(4, "premium", 800),
        _event(5, "premium", 0, status=502),
        _event(6, "cache", 5, namespace="b"),
    ]
    for event in events:
        await pg_ledger.record(event)
    await pg_ledger.record(events[0])  # duplicate request_id is ignored

    summary = await pg_ledger.summary(namespace="a")

    assert summary["requests"] == 4
    assert summary["errors"] == 1
    assert summary["deflection_rate"] == pytest.approx(0.75)
    assert summary["escalations"] == 1
    assert summary["routes"]["cache"]["p50_ms"] == pytest.approx(10.0)
    # Each served request: 1000 x $2/M + 100 x $10/M = $0.003 baseline.
    assert summary["cost"]["baseline_usd"] == pytest.approx(0.012)
    assert summary["cost"]["saved_usd"] == pytest.approx(0.009)
    assert summary["cost"]["savings_pct"] == pytest.approx(0.75)
    assert summary["pricing"]["version"] == "test"
    assert (await pg_ledger.summary())["requests"] == 5  # all namespaces
    assert (await pg_ledger.summary(since_minutes=60))["requests"] == 5
