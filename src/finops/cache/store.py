from dataclasses import dataclass
from typing import Any, Protocol

from pgvector import Vector
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from finops.cache.keys import CacheKeys


@dataclass(frozen=True)
class CacheMatch:
    entry_id: int
    response: dict[str, Any]
    similarity: float  # 1.0 for exact matches


@dataclass(frozen=True)
class NewEntry:
    namespace: str
    keys: CacheKeys
    embedding: list[float]
    embedding_model: str
    upstream_model: str
    response: dict[str, Any]
    ttl_s: int


class CacheStore(Protocol):
    async def find_exact(self, namespace: str, request_hash: str) -> CacheMatch | None: ...

    async def find_nearest(
        self, namespace: str, keys: CacheKeys, embedding: list[float], embedding_model: str
    ) -> CacheMatch | None:
        """Closest live entry in the same context with the same guard fingerprint.

        Returns it regardless of similarity so callers can log near misses.
        """
        ...

    async def put(self, entry: NewEntry) -> None: ...

    async def record_hit(self, entry_id: int) -> None: ...


class PgVectorCacheStore:
    def __init__(self, pool: AsyncConnectionPool):
        self._pool = pool

    async def find_exact(self, namespace: str, request_hash: str) -> CacheMatch | None:
        async with self._pool.connection() as conn:
            row = await (
                await conn.execute(
                    """
                    SELECT id, response FROM cache_entries
                    WHERE namespace = %s AND request_hash = %s AND expires_at > now()
                    """,
                    (namespace, request_hash),
                )
            ).fetchone()
        return CacheMatch(row[0], row[1], 1.0) if row else None

    async def find_nearest(
        self, namespace: str, keys: CacheKeys, embedding: list[float], embedding_model: str
    ) -> CacheMatch | None:
        vector = Vector(embedding)
        async with self._pool.connection() as conn, conn.transaction():
            # HNSW returns its top candidates *before* the WHERE filter runs, so a
            # selective filter could come back empty. Iterative scans keep searching.
            await conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            row = await (
                await conn.execute(
                    """
                    SELECT id, response, 1 - (embedding <=> %(v)s) AS similarity
                    FROM cache_entries
                    WHERE namespace = %(ns)s
                      AND context_hash = %(ctx)s
                      AND embedding_model = %(em)s
                      AND guard_fingerprint = %(fp)s
                      AND expires_at > now()
                    ORDER BY embedding <=> %(v)s
                    LIMIT 1
                    """,
                    {
                        "v": vector,
                        "ns": namespace,
                        "ctx": keys.context_hash,
                        "em": embedding_model,
                        "fp": keys.guard_fingerprint,
                    },
                )
            ).fetchone()
        return CacheMatch(row[0], row[1], float(row[2])) if row else None

    async def put(self, entry: NewEntry) -> None:
        usage = entry.response.get("usage") or {}
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO cache_entries (
                    namespace, request_hash, context_hash, prompt_text, guard_fingerprint,
                    embedding, embedding_model, upstream_model, response,
                    prompt_tokens, completion_tokens, expires_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    now() + make_interval(secs => %s)
                )
                ON CONFLICT (namespace, request_hash) DO UPDATE SET
                    response = EXCLUDED.response,
                    embedding = EXCLUDED.embedding,
                    embedding_model = EXCLUDED.embedding_model,
                    prompt_tokens = EXCLUDED.prompt_tokens,
                    completion_tokens = EXCLUDED.completion_tokens,
                    created_at = now(),
                    expires_at = EXCLUDED.expires_at
                """,
                (
                    entry.namespace,
                    entry.keys.request_hash,
                    entry.keys.context_hash,
                    entry.keys.prompt_text,
                    entry.keys.guard_fingerprint,
                    Vector(entry.embedding),
                    entry.embedding_model,
                    entry.upstream_model,
                    Jsonb(entry.response),
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    entry.ttl_s,
                ),
            )

    async def record_hit(self, entry_id: int) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                "UPDATE cache_entries SET hit_count = hit_count + 1, last_hit_at = now() "
                "WHERE id = %s",
                (entry_id,),
            )
