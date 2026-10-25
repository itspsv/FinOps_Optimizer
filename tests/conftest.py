from collections.abc import Iterator

import pytest
import respx
from fastapi.testclient import TestClient

from finops.api.main import create_app
from finops.cache import SemanticCache
from finops.core.config import CacheSettings, ProviderSettings, Settings
from tests.fakes import FakeEmbedder, InMemoryCacheStore

UPSTREAM = "https://upstream.test/v1"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        log_level="DEBUG",
        cache=CacheSettings(enabled=False, similarity_threshold=0.8),
        premium=ProviderSettings(
            name="groq",
            base_url=UPSTREAM,
            model="test-premium-model",
            api_key="test-key",
            timeout_s=5,
        ),
    )


@pytest.fixture
def upstream() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=UPSTREAM, assert_all_called=False) as router:
        yield router


@pytest.fixture
def client(settings: Settings, upstream: respx.MockRouter) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings)) as test_client:
        yield test_client


@pytest.fixture
def cache_store() -> InMemoryCacheStore:
    return InMemoryCacheStore()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def cache(
    settings: Settings, cache_store: InMemoryCacheStore, embedder: FakeEmbedder
) -> SemanticCache:
    return SemanticCache(cache_store, embedder, settings.cache)


@pytest.fixture
def cached_client(
    settings: Settings, upstream: respx.MockRouter, cache: SemanticCache
) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings, cache=cache)) as test_client:
        yield test_client
