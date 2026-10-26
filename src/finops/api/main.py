import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from finops import __version__
from finops.api.routes import router
from finops.cache import SemanticCache
from finops.core.config import Settings, get_settings
from finops.core.logging import configure_logging
from finops.providers import OpenAICompatibleProvider, Provider
from finops.router import RoutingEngine

logger = logging.getLogger(__name__)


def create_app(
    settings: Settings | None = None,
    premium: Provider | None = None,
    local: Provider | None = None,
    cache: SemanticCache | None = None,
) -> FastAPI:
    """Build the proxy app. Tests can inject settings, fake providers or a cache."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or get_settings()
        configure_logging(resolved.log_level)
        async with AsyncExitStack() as stack:
            premium_provider = premium or OpenAICompatibleProvider(resolved.premium)
            stack.push_async_callback(premium_provider.aclose)
            local_provider = local
            if local_provider is None and resolved.routing.enabled and resolved.local:
                local_provider = OpenAICompatibleProvider(resolved.local)
            if local_provider is not None:
                stack.push_async_callback(local_provider.aclose)
            app.state.engine = RoutingEngine(premium_provider, local_provider, resolved.routing)
            app.state.cache = cache
            if cache is None and resolved.cache.enabled:
                app.state.cache = await _build_cache(resolved, stack)
            yield

    app = FastAPI(title="AI FinOps Optimizer", version=__version__, lifespan=lifespan)
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def openai_style_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": "Invalid request body",
                    "type": "invalid_request_error",
                    "details": jsonable_encoder(exc.errors()),
                }
            },
        )

    return app


async def _build_cache(settings: Settings, stack: AsyncExitStack) -> SemanticCache | None:
    """Start the cache, or run without it if Postgres or the embedder is unavailable."""
    from finops.cache.embeddings import FastEmbedEmbedder
    from finops.cache.store import PgVectorCacheStore
    from finops.db.pool import open_pool

    try:
        pool = await open_pool(settings.database_url)
        stack.push_async_callback(pool.close)
        embedder = FastEmbedEmbedder(
            settings.cache.embedding_model,
            cache_dir=str(settings.cache.model_cache_dir.expanduser()),
        )
        await embedder.embed("warm up")  # load the model now, not on the first request
    except Exception:
        logger.exception("semantic cache unavailable; starting without it")
        return None
    logger.info("semantic cache ready (model=%s)", settings.cache.embedding_model)
    return SemanticCache(PgVectorCacheStore(pool), embedder, settings.cache)


app = create_app()
