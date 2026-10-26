from dataclasses import asdict
from typing import Any, Protocol

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from finops.ledger.models import LedgerRow

ROUTES = ("cache", "local", "premium")


class LedgerStore(Protocol):
    async def insert(self, row: LedgerRow) -> None: ...

    async def summary(self, namespace: str | None, since_minutes: int | None) -> dict[str, Any]: ...


_COLUMNS = [
    "request_id", "namespace", "requested_model", "stream", "status_code", "route",
    "cache_status", "cache_similarity", "decision_tier", "decision_reason",
    "complexity_score", "escalation_reason", "provider", "upstream_model", "prompt_tokens",
    "completion_tokens", "tokens_estimated", "latency_ms", "local_latency_ms",
    "premium_latency_ms", "pricing_version", "baseline_model", "baseline_cost_usd", "cost_usd",
    "saved_usd", "billed_usd",
]  # fmt: skip

_INSERT = (
    f"INSERT INTO request_log ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join(f'%({c})s' for c in _COLUMNS)}) "
    "ON CONFLICT (request_id) DO NOTHING"
)

_FILTER = """
    (%(ns)s::text IS NULL OR namespace = %(ns)s)
    AND (%(mins)s::int IS NULL OR created_at >= now() - make_interval(mins => %(mins)s::int))
"""

_BY_ROUTE = f"""
    SELECT route,
           count(*)                                                    AS requests,
           count(*) FILTER (WHERE escalation_reason IS NOT NULL)       AS escalations,
           coalesce(sum(prompt_tokens), 0)                             AS prompt_tokens,
           coalesce(sum(completion_tokens), 0)                         AS completion_tokens,
           avg(latency_ms)                                             AS avg_ms,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms)     AS p50_ms,
           percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)    AS p95_ms,
           coalesce(sum(baseline_cost_usd), 0)                         AS baseline_usd,
           coalesce(sum(cost_usd), 0)                                  AS cost_usd,
           coalesce(sum(saved_usd), 0)                                 AS saved_usd,
           coalesce(sum(billed_usd), 0)                                AS billed_usd
    FROM request_log
    WHERE status_code < 400 AND {_FILTER}
    GROUP BY route
"""

_ERRORS = f"SELECT count(*) AS errors FROM request_log WHERE status_code >= 400 AND {_FILTER}"


class PgLedgerStore:
    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    async def insert(self, row: LedgerRow) -> None:
        params = {**asdict(row.event), **asdict(row)}
        params.pop("event")
        async with self._pool.connection() as conn:
            await conn.execute(_INSERT, params)

    async def summary(self, namespace: str | None, since_minutes: int | None) -> dict[str, Any]:
        params = {"ns": namespace, "mins": since_minutes}
        async with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(_BY_ROUTE, params)
            by_route = {r["route"]: r for r in await cur.fetchall()}
            await cur.execute(_ERRORS, params)
            errors = (await cur.fetchone())["errors"]
        return build_summary(by_route, errors)


def build_summary(by_route: dict[str, dict[str, Any]], errors: int) -> dict[str, Any]:
    """Headline FinOps metrics from per-route aggregates."""

    def num(route: str, key: str) -> float:
        value = by_route.get(route, {}).get(key)
        return float(value) if value is not None else 0.0

    def tokens(route: str) -> float:
        return num(route, "prompt_tokens") + num(route, "completion_tokens")

    total = sum(num(r, "requests") for r in ROUTES)
    deflected = num("cache", "requests") + num("local", "requests")
    total_tokens = sum(tokens(r) for r in ROUTES)
    baseline = sum(num(r, "baseline_usd") for r in ROUTES)
    saved = sum(num(r, "saved_usd") for r in ROUTES)

    def ms(route: str, key: str) -> float | None:
        value = by_route.get(route, {}).get(key)
        return round(float(value), 1) if value is not None else None

    return {
        "requests": int(total),
        "errors": int(errors),
        "deflection_rate": deflected / total if total else 0.0,
        "token_deflection_rate": (tokens("cache") + tokens("local")) / total_tokens
        if total_tokens
        else 0.0,
        "escalations": int(sum(num(r, "escalations") for r in ROUTES)),
        "cost": {
            "baseline_usd": round(baseline, 6),
            "actual_usd": round(sum(num(r, "cost_usd") for r in ROUTES), 6),
            "saved_usd": round(saved, 6),
            "savings_pct": saved / baseline if baseline else 0.0,
            "billed_usd": round(sum(num(r, "billed_usd") for r in ROUTES), 6),
        },
        "routes": {
            r: {
                "requests": int(num(r, "requests")),
                "tokens": int(tokens(r)),
                "avg_ms": ms(r, "avg_ms"),
                "p50_ms": ms(r, "p50_ms"),
                "p95_ms": ms(r, "p95_ms"),
            }
            for r in ROUTES
        },
    }
