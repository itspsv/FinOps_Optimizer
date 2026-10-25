"""Two-tier cache: exact hash first (no embedding cost), then semantic search.

The cache fails open: if the database or embedder breaks, requests still go
upstream and the error is logged, never shown to the client.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from finops.cache.embeddings import Embedder
from finops.cache.keys import CacheKeys, build_keys, is_storable, skip_reason
from finops.cache.store import CacheMatch, CacheStore, NewEntry
from finops.core.config import CacheSettings

logger = logging.getLogger(__name__)


class CacheStatus(StrEnum):
    HIT_EXACT = "hit_exact"
    HIT_SEMANTIC = "hit_semantic"
    MISS = "miss"
    SKIP = "skip"  # request not cacheable (tools, high temperature, ...)
    BYPASS = "bypass"  # client asked to skip the cache
    ERROR = "error"  # cache unavailable; failed open


@dataclass
class CacheLookup:
    status: CacheStatus
    namespace: str
    upstream_model: str
    match: CacheMatch | None = None
    keys: CacheKeys | None = None
    embedding: list[float] | None = None
    nearest_similarity: float | None = None  # best candidate, even on a miss
    skip_reason: str | None = None

    @property
    def is_hit(self) -> bool:
        return self.status in (CacheStatus.HIT_EXACT, CacheStatus.HIT_SEMANTIC)


class SemanticCache:
    def __init__(self, store: CacheStore, embedder: Embedder, settings: CacheSettings):
        self._store = store
        self._embedder = embedder
        self._settings = settings

    async def lookup(
        self, payload: dict[str, Any], upstream_model: str, namespace: str, bypass: bool = False
    ) -> CacheLookup:
        result = CacheLookup(CacheStatus.MISS, namespace, upstream_model)
        if bypass:
            result.status = CacheStatus.BYPASS
            return result
        if reason := skip_reason(payload, self._settings.max_temperature):
            result.status, result.skip_reason = CacheStatus.SKIP, reason
            return result

        try:
            result.keys = build_keys(payload, upstream_model)
            if match := await self._store.find_exact(namespace, result.keys.request_hash):
                result.status, result.match = CacheStatus.HIT_EXACT, match
                return result

            result.embedding = await self._embedder.embed(result.keys.prompt_text)
            nearest = await self._store.find_nearest(
                namespace, result.keys, result.embedding, self._embedder.model_name
            )
        except Exception:
            logger.exception("cache lookup failed; continuing without cache")
            result.status = CacheStatus.ERROR
            return result

        if nearest is not None:
            result.nearest_similarity = nearest.similarity
            if nearest.similarity >= self._settings.similarity_threshold:
                result.status, result.match = CacheStatus.HIT_SEMANTIC, nearest
        return result

    async def store(self, lookup: CacheLookup, response: dict[str, Any]) -> None:
        """Save an upstream answer after a miss. Meant to run after the response is sent."""
        if lookup.status != CacheStatus.MISS or lookup.keys is None or lookup.embedding is None:
            return
        if not is_storable(response):
            return
        try:
            await self._store.put(
                NewEntry(
                    namespace=lookup.namespace,
                    keys=lookup.keys,
                    embedding=lookup.embedding,
                    embedding_model=self._embedder.model_name,
                    upstream_model=lookup.upstream_model,
                    response=response,
                    ttl_s=self._settings.ttl_s,
                )
            )
        except Exception:
            logger.exception("cache store failed")

    async def record_hit(self, lookup: CacheLookup) -> None:
        if lookup.match is None:
            return
        try:
            await self._store.record_hit(lookup.match.entry_id)
        except Exception:
            logger.exception("cache hit bookkeeping failed")
