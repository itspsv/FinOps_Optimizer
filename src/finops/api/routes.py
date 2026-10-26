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
from finops.providers import ProviderError
from finops.router import RouteDecision, RouteOutcome, RoutingEngine, Tier

logger = logging.getLogger(__name__)
router = APIRouter()

NAMESPACE_HEADER = "x-finops-namespace"
CACHE_CONTROL_HEADER = "x-finops-cache-control"
ROUTE_OVERRIDE_HEADER = "x-finops-route-override"  # "local" or "premium"


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    engine: RoutingEngine = request.app.state.engine
    return {
        "status": "ok",
        "cache": "enabled" if request.app.state.cache is not None else "disabled",
        "routing": "enabled" if engine.local is not None else "disabled",
        "local_breaker": engine.breaker.state if engine.local is not None else "n/a",
    }


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Response:
    engine: RoutingEngine = request.app.state.engine
    cache: SemanticCache | None = request.app.state.cache
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    started = time.perf_counter()
    payload = body.upstream_payload()
    headers = {"x-finops-request-id": request_id}

    lookup: CacheLookup | None = None
    if cache is not None:
        lookup = await cache.lookup(
            payload,
            model_scope=engine.cache_scope,
            namespace=request.headers.get(NAMESPACE_HEADER, "default"),
            bypass=request.headers.get(CACHE_CONTROL_HEADER, "").lower() == "bypass",
        )
        headers["x-finops-cache"] = lookup.status.value
        if lookup.nearest_similarity is not None:
            headers["x-finops-cache-similarity"] = f"{lookup.nearest_similarity:.4f}"

    if lookup is not None and lookup.is_hit:
        return _serve_from_cache(cache, lookup, body, request_id, headers, started)

    decision = engine.decide(payload, request.headers.get(ROUTE_OVERRIDE_HEADER, "").lower())
    headers["x-finops-route-reason"] = decision.reason
    headers["x-finops-complexity"] = str(decision.score)

    try:
        if body.stream and decision.tier is Tier.PREMIUM:
            # True passthrough streaming. Streamed premium answers aren't cached yet.
            chunks = await engine.premium.stream(payload)
            headers |= {"x-finops-route": "premium", "x-finops-provider": engine.premium.name}
            _log(request_id, lookup, decision, None, started, stream=True)
            return StreamingResponse(chunks, media_type="text/event-stream", headers=headers)
        # Local answers are checked before use, so they're buffered and replayed if
        # the client asked for a stream. Short local answers take well under a second.
        outcome = await engine.run(payload, decision)
    except ProviderError as exc:
        logger.warning(
            "request_id=%s route=%s status=%s error=%s",
            request_id,
            decision.tier.value,
            exc.status_code,
            exc.body,
        )
        return JSONResponse(exc.body, status_code=exc.status_code, headers=headers)

    headers |= {
        "x-finops-route": outcome.route.value,
        "x-finops-provider": outcome.provider,
        "x-finops-latency-ms": f"{_elapsed_ms(started):.1f}",
    }
    if outcome.escalated:
        headers["x-finops-escalated"] = outcome.escalation_reason
    _log(request_id, lookup, decision, outcome, started)

    # Write to the cache after the response is sent, so a miss pays no extra latency.
    background = BackgroundTask(cache.store, lookup, outcome.response) if cache and lookup else None
    if body.stream:
        return StreamingResponse(
            completion_to_sse(outcome.response),
            media_type="text/event-stream",
            headers=headers,
            background=background,
        )
    return JSONResponse(outcome.response, headers=headers, background=background)


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


def _log(
    request_id: str,
    lookup: CacheLookup | None,
    decision: RouteDecision,
    outcome: RouteOutcome | None,
    started: float,
    stream: bool = False,
) -> None:
    usage = (outcome.response.get("usage") if outcome else None) or {}
    logger.info(
        "request_id=%s route=%s cache=%s decision=%s reason=%s score=%s escalated=%s "
        "stream=%s prompt_tokens=%s completion_tokens=%s latency_ms=%.1f",
        request_id,
        outcome.route.value if outcome else decision.tier.value,
        lookup.status.value if lookup else "disabled",
        decision.tier.value,
        decision.reason,
        decision.score,
        outcome.escalation_reason if outcome else None,
        stream,
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        _elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
