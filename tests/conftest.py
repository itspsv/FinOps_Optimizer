from collections.abc import Iterator

import pytest
import respx
from fastapi.testclient import TestClient

from finops.api.main import create_app
from finops.cache import SemanticCache
from finops.core.config import CacheSettings, ProviderSettings, RoutingSettings, Settings
from tests.fakes import LOCAL_UPSTREAM, UPSTREAM, FakeEmbedder, InMemoryCacheStore


@pytest.fixture
def settings() -> Settings:
    """Premium-only by default; routing tests use `routed_settings`."""
    return Settings(
        log_level="DEBUG",
        cache=CacheSettings(enabled=False, similarity_threshold=0.8),
        routing=RoutingSettings(enabled=False),
        premium=ProviderSettings(
            name="groq",
            base_url=UPSTREAM,
            model="test-premium-model",
            api_key="test-key",
            timeout_s=5,
        ),
        local=ProviderSettings(
            name="ollama", base_url=LOCAL_UPSTREAM, model="test-local-model", timeout_s=5
        ),
    )


@pytest.fixture
def routed_settings(settings: Settings) -> Settings:
    return settings.model_copy(update={"routing": RoutingSettings(enabled=True)})


@pytest.fixture
def upstream() -> Iterator[respx.MockRouter]:
    """Mocks both Groq and Ollama; routes use full URLs (PREMIUM_URL, LOCAL_URL)."""
    with respx.mock(assert_all_called=False) as router:
        yield router


@pytest.fixture
def client(settings: Settings, upstream: respx.MockRouter) -> Iterator[TestClient]:
    with TestClient(create_app(settings=settings)) as test_client:
        yield test_client


@pytest.fixture
def routed_client(routed_settings: Settings, upstream: respx.MockRouter) -> Iterator[TestClient]:
    with TestClient(create_app(settings=routed_settings)) as test_client:
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


@pytest.fixture
def routed_cached_client(
    routed_settings: Settings, upstream: respx.MockRouter, cache: SemanticCache
) -> Iterator[TestClient]:
    with TestClient(create_app(settings=routed_settings, cache=cache)) as test_client:
        yield test_client
