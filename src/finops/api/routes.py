import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from finops.api.schemas import ChatCompletionRequest
from finops.api.streaming import completion_to_sse
from finops.cache import CacheLookup, SemanticCache
from finops.providers import Provider, ProviderError

logger = logging.getLogger(__name__)
router = APIRouter()

NAMESPACE_HEADER = "x-finops-namespace"
CACHE_CONTROL_HEADER = "x-finops-cache-control"


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    return {
        "status": "ok",
        "cache": "enabled" if request.app.state.cache is not None else "disabled",
    }


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Response:
    provider: Provider = request.app.state.premium
    cache: SemanticCache | None = request.app.state.cache
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    started = time.perf_counter()
    payload = body.upstream_payload()
    headers = {"x-finops-request-id": request_id}

    lookup: CacheLookup | None = None
    if cache is not None:
        lookup = await cache.lookup(
            payload,
            upstream_model=provider.model,
            namespace=request.headers.get(NAMESPACE_HEADER, "default"),
            bypass=request.headers.get(CACHE_CONTROL_HEADER, "").lower() == "bypass",
        )
        headers["x-finops-cache"] = lookup.status.value
        if lookup.nearest_similarity is not None:
            headers["x-finops-cache-similarity"] = f"{lookup.nearest_similarity:.4f}"

    if lookup is not None and lookup.is_hit:
        return _serve_from_cache(cache, lookup, body, request_id, headers, started)

    headers["x-finops-route"] = "premium"
    headers["x-finops-provider"] = provider.name
    try:
        if body.stream:
            # Streamed answers aren't cached yet; that needs reassembling the SSE chunks.
            chunks = await provider.stream(payload)
            logger.info("request_id=%s route=premium stream=true", request_id)
            return StreamingResponse(chunks, media_type="text/event-stream", headers=headers)
        result = await provider.chat(payload)
    except ProviderError as exc:
        logger.warning(
            "request_id=%s route=premium status=%s error=%s",
            request_id,
            exc.status_code,
            exc.body,
        )
        return JSONResponse(exc.body, status_code=exc.status_code, headers=headers)

    latency_ms = _elapsed_ms(started)
    usage = result.get("usage") or {}
    headers["x-finops-latency-ms"] = f"{latency_ms:.1f}"
    logger.info(
        "request_id=%s route=premium cache=%s requested_model=%s upstream_model=%s "
        "prompt_tokens=%s completion_tokens=%s latency_ms=%.1f",
        request_id,
        lookup.status.value if lookup else "disabled",
        body.model,
        provider.model,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        latency_ms,
    )
    # Write to the cache after the response is sent, so a miss pays no extra latency.
    background = BackgroundTask(cache.store, lookup, result) if cache and lookup else None
    return JSONResponse(result, headers=headers, background=background)


def _serve_from_cache(
    cache: SemanticCache,
    lookup: CacheLookup,
    body: ChatCompletionRequest,
    request_id: str,
    headers: dict[str, str],
    started: float,
) -> Response:
    completion: dict[str, Any] = {**lookup.match.response, "id": f"chatcmpl-{request_id}"}
    latency_ms = _elapsed_ms(started)
    headers["x-finops-route"] = "cache"
    headers["x-finops-latency-ms"] = f"{latency_ms:.1f}"
    usage = completion.get("usage") or {}
    logger.info(
        "request_id=%s route=cache cache=%s similarity=%.4f saved_prompt_tokens=%s "
        "saved_completion_tokens=%s latency_ms=%.1f",
        request_id,
        lookup.status.value,
        lookup.match.similarity,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        latency_ms,
    )
    background = BackgroundTask(cache.record_hit, lookup)
    if body.stream:
        return StreamingResponse(
            completion_to_sse(completion),
            media_type="text/event-stream",
            headers=headers,
            background=background,
        )
    return JSONResponse(completion, headers=headers, background=background)


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
