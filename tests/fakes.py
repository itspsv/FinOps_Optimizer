"""In-memory stand-ins so cache logic can be tested without Postgres or a model download."""

import hashlib
import math
import re
from dataclasses import dataclass, field

from finops.cache.keys import CacheKeys
from finops.cache.store import CacheMatch, NewEntry
from finops.ledger.models import LedgerRow
from finops.ledger.store import build_summary

DIM = 64

UPSTREAM = "https://upstream.test/v1"
LOCAL_UPSTREAM = "http://local.test/v1"
PREMIUM_URL = f"{UPSTREAM}/chat/completions"
LOCAL_URL = f"{LOCAL_UPSTREAM}/chat/completions"


class FakeEmbedder:
    """Bag-of-words hashing: prompts sharing most words get high cosine similarity."""

    model_name = "fake-embedder"

    def __init__(self, overrides: dict[str, list[float]] | None = None):
        self.overrides = overrides or {}
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        if text in self.overrides:
            return self.overrides[text]
        vector = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        return vector


@dataclass
class _Row:
    id: int
    entry: NewEntry
    hits: int = 0


@dataclass
class InMemoryCacheStore:
    rows: list[_Row] = field(default_factory=list)
    fail: bool = False

    async def find_exact(self, namespace: str, request_hash: str) -> CacheMatch | None:
        self._maybe_fail()
        for row in self.rows:
            if row.entry.namespace == namespace and row.entry.keys.request_hash == request_hash:
                return CacheMatch(row.id, row.entry.response, 1.0)
        return None

    async def find_nearest(
        self, namespace: str, keys: CacheKeys, embedding: list[float], embedding_model: str
    ) -> CacheMatch | None:
        self._maybe_fail()
        candidates = [
            row
            for row in self.rows
            if row.entry.namespace == namespace
            and row.entry.keys.context_hash == keys.context_hash
            and row.entry.keys.guard_fingerprint == keys.guard_fingerprint
            and row.entry.embedding_model == embedding_model
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda r: _cosine(r.entry.embedding, embedding))
        return CacheMatch(best.id, best.entry.response, _cosine(best.entry.embedding, embedding))

    async def put(self, entry: NewEntry) -> None:
        self._maybe_fail()
        self.rows = [
            r
            for r in self.rows
            if not (
                r.entry.namespace == entry.namespace
                and r.entry.keys.request_hash == entry.keys.request_hash
            )
        ]
        self.rows.append(_Row(len(self.rows) + 1, entry))

    async def record_hit(self, entry_id: int) -> None:
        for row in self.rows:
            if row.id == entry_id:
                row.hits += 1

    def _maybe_fail(self) -> None:
        if self.fail:
            raise ConnectionError("database down")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@dataclass
class InMemoryLedgerStore:
    """Mirrors PgLedgerStore's aggregation in Python."""

    rows: list[LedgerRow] = field(default_factory=list)

    async def insert(self, row: LedgerRow) -> None:
        self.rows.append(row)

    async def summary(self, namespace: str | None, since_minutes: int | None) -> dict:
        rows = [r for r in self.rows if namespace is None or r.event.namespace == namespace]
        ok = [r for r in rows if r.event.status_code < 400]
        by_route = {}
        for route in {r.event.route for r in ok}:
            group = [r for r in ok if r.event.route == route]
            latencies = sorted(r.event.latency_ms for r in group)
            by_route[route] = {
                "requests": len(group),
                "escalations": sum(r.event.escalation_reason is not None for r in group),
                "prompt_tokens": sum(r.event.prompt_tokens for r in group),
                "completion_tokens": sum(r.event.completion_tokens for r in group),
                "avg_ms": sum(latencies) / len(latencies),
                "p50_ms": _percentile(latencies, 0.5),
                "p95_ms": _percentile(latencies, 0.95),
                "baseline_usd": sum(r.baseline_cost_usd for r in group),
                "cost_usd": sum(r.cost_usd for r in group),
                "saved_usd": sum(r.saved_usd for r in group),
                "billed_usd": sum(r.billed_usd or 0 for r in group),
            }
        return build_summary(by_route, len(rows) - len(ok))


def _percentile(values: list[float], q: float) -> float:
    """Linear interpolation, like Postgres percentile_cont."""
    pos = (len(values) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return values[low] + (values[high] - values[low]) * (pos - low)
