from finops.cache import CacheStatus, SemanticCache
from tests.fakes import FakeEmbedder, InMemoryCacheStore

MODEL = "premium-model"
ANSWER = {
    "choices": [
        {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "A"}}
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
}


def _payload(text: str) -> dict:
    return {"messages": [{"role": "user", "content": text}]}


async def _prime(cache: SemanticCache, text: str, namespace: str = "default") -> None:
    lookup = await cache.lookup(_payload(text), MODEL, namespace)
    assert lookup.status == CacheStatus.MISS
    await cache.store(lookup, ANSWER)


async def test_exact_hit_skips_embedding(cache: SemanticCache, embedder: FakeEmbedder):
    await _prime(cache, "How do I reset my password?")
    calls_before = embedder.calls

    lookup = await cache.lookup(_payload("How do I reset my password?"), MODEL, "default")

    assert lookup.status == CacheStatus.HIT_EXACT
    assert lookup.match.response == ANSWER
    assert embedder.calls == calls_before


async def test_semantic_hit_above_threshold(cache: SemanticCache):
    await _prime(cache, "How do I reset my password?")

    lookup = await cache.lookup(_payload("how do I reset my password"), MODEL, "default")

    assert lookup.status == CacheStatus.HIT_SEMANTIC
    assert lookup.nearest_similarity >= 0.8


async def test_below_threshold_is_a_miss_but_reports_similarity(cache: SemanticCache):
    await _prime(cache, "How do I reset my password?")

    lookup = await cache.lookup(_payload("What is the weather in Paris?"), MODEL, "default")

    assert lookup.status == CacheStatus.MISS
    assert lookup.nearest_similarity is not None
    assert lookup.nearest_similarity < 0.8


async def test_guard_blocks_lookalike_with_different_number(cache: SemanticCache):
    await _prime(cache, "What was Q3 2024 revenue?")

    lookup = await cache.lookup(_payload("What was Q3 2025 revenue?"), MODEL, "default")

    assert lookup.status == CacheStatus.MISS
    assert lookup.nearest_similarity is None  # filtered out before similarity is computed


async def test_namespaces_are_isolated(cache: SemanticCache):
    await _prime(cache, "What is our refund policy?", namespace="tenant-a")

    lookup = await cache.lookup(_payload("What is our refund policy?"), MODEL, "tenant-b")

    assert lookup.status == CacheStatus.MISS


async def test_bypass_and_skip(cache: SemanticCache):
    bypassed = await cache.lookup(_payload("hi"), MODEL, "default", bypass=True)
    skipped = await cache.lookup({**_payload("hi"), "temperature": 2.0}, MODEL, "default")

    assert bypassed.status == CacheStatus.BYPASS
    assert skipped.status == CacheStatus.SKIP
    assert skipped.skip_reason == "temperature"


async def test_fails_open_when_store_is_down(cache: SemanticCache, cache_store: InMemoryCacheStore):
    cache_store.fail = True

    lookup = await cache.lookup(_payload("hi"), MODEL, "default")
    await cache.store(lookup, ANSWER)  # must not raise

    assert lookup.status == CacheStatus.ERROR


async def test_does_not_store_truncated_answers(
    cache: SemanticCache, cache_store: InMemoryCacheStore
):
    lookup = await cache.lookup(_payload("Write an essay"), MODEL, "default")
    truncated = {"choices": [{**ANSWER["choices"][0], "finish_reason": "length"}]}

    await cache.store(lookup, truncated)

    assert cache_store.rows == []
