"""Runs against a real Postgres + pgvector database (finops_test by default).

Skipped automatically when Postgres isn't reachable.
"""

import os
from collections.abc import AsyncIterator

import psycopg
import pytest
from psycopg import sql

from finops.cache.keys import build_keys
from finops.cache.store import NewEntry, PgVectorCacheStore
from finops.db.pool import open_pool
from finops.db.schema import apply_schema

TEST_DB = os.environ.get("FINOPS_TEST_DATABASE", "finops_test")
DIM = 4
ANSWER = {"choices": [{"message": {"content": "A"}}], "usage": {"prompt_tokens": 7}}


async def _ensure_database() -> None:
    try:
        conn = await psycopg.AsyncConnection.connect("dbname=postgres", autocommit=True)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    async with conn:
        exists = await (
            await conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,))
        ).fetchone()
        if not exists:
            await conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_DB)))


@pytest.fixture
async def store() -> AsyncIterator[PgVectorCacheStore]:
    await _ensure_database()
    async with await psycopg.AsyncConnection.connect(f"dbname={TEST_DB}") as conn:
        await conn.execute("DROP TABLE IF EXISTS cache_entries")
        await apply_schema(conn, embedding_dim=DIM)
    pool = await open_pool(f"dbname={TEST_DB}")
    try:
        yield PgVectorCacheStore(pool)
    finally:
        await pool.close()


def _entry(text: str, embedding: list[float], namespace: str = "default", ttl_s: int = 60):
    keys = build_keys({"messages": [{"role": "user", "content": text}]}, "m")
    return NewEntry(namespace, keys, embedding, "emb", "m", ANSWER, ttl_s)


async def test_exact_and_nearest_lookup(store: PgVectorCacheStore):
    entry = _entry("How do I reset my password?", [1, 0, 0, 0])
    await store.put(entry)

    exact = await store.find_exact("default", entry.keys.request_hash)
    near_keys = build_keys({"messages": [{"role": "user", "content": "reset password how"}]}, "m")
    nearest = await store.find_nearest("default", near_keys, [0.9, 0.1, 0, 0], "emb")

    assert exact.response == ANSWER and exact.similarity == 1.0
    assert nearest.entry_id == exact.entry_id
    assert 0.99 < nearest.similarity < 1.0


async def test_filters_namespace_model_and_fingerprint(store: PgVectorCacheStore):
    await store.put(_entry("Revenue in 2024?", [1, 0, 0, 0]))

    same_fp = build_keys({"messages": [{"role": "user", "content": "2024 revenue"}]}, "m")
    other_fp = build_keys({"messages": [{"role": "user", "content": "2025 revenue"}]}, "m")

    assert await store.find_nearest("other", same_fp, [1, 0, 0, 0], "emb") is None
    assert await store.find_nearest("default", same_fp, [1, 0, 0, 0], "other-emb") is None
    assert await store.find_nearest("default", other_fp, [1, 0, 0, 0], "emb") is None
    assert await store.find_nearest("default", same_fp, [1, 0, 0, 0], "emb") is not None


async def test_expired_entries_are_ignored_and_upsert_refreshes(store: PgVectorCacheStore):
    expired = _entry("hello", [0, 1, 0, 0], ttl_s=-1)
    await store.put(expired)
    assert await store.find_exact("default", expired.keys.request_hash) is None

    await store.put(_entry("hello", [0, 1, 0, 0], ttl_s=60))
    assert await store.find_exact("default", expired.keys.request_hash) is not None


async def test_record_hit(store: PgVectorCacheStore):
    entry = _entry("hello", [0, 0, 1, 0])
    await store.put(entry)
    match = await store.find_exact("default", entry.keys.request_hash)

    await store.record_hit(match.entry_id)
    await store.record_hit(match.entry_id)

    async with await psycopg.AsyncConnection.connect(f"dbname={TEST_DB}") as conn:
        row = await (
            await conn.execute(
                "SELECT hit_count FROM cache_entries WHERE id = %s", (match.entry_id,)
            )
        ).fetchone()
    assert row[0] == 2
