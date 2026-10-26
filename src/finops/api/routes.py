import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTasks

from finops.api.schemas import ChatCompletionRequest
from finops.api.streaming import StreamTap, completion_to_sse
from finops.cache import CacheLookup, SemanticCache
from finops.ledger import Ledger, RequestEvent, usage_tokens
from finops.providers import ProviderError
from finops.router import RouteDecision, RouteOutcome, RoutingEngine, Tier
from finops.router.classifier import estimate_tokens

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
        "ledger": "enabled" if request.app.state.ledger is not None else "disabled",
        "routing": "enabled" if engine.local is not None else "disabled",
        "local_breaker": engine.breaker.state if engine.local is not None else "n/a",
    }


@router.get("/finops/summary")
async def finops_summary(
    request: Request, namespace: str | None = None, since_minutes: int | None = None
) -> dict[str, Any]:
    ledger: Ledger | None = request.app.state.ledger
    if ledger is None:
        raise HTTPException(503, "ledger is disabled or the database is unavailable")
    return await ledger.summary(namespace, since_minutes)


class _Request:
    """Per-request context shared by the response paths below."""

    def __init__(self, request: Request, body: ChatCompletionRequest):
        self.engine: RoutingEngine = request.app.state.engine
        self.cache: SemanticCache | None = request.app.state.cache
        self.ledger: Ledger | None = request.app.state.ledger
        self.body = body
        self.id = f"req_{uuid.uuid4().hex[:16]}"
        self.started = time.perf_counter()
        self.namespace = request.headers.get(NAMESPACE_HEADER, "default")
        self.headers = {"x-finops-request-id": self.id}
        self.tasks = BackgroundTasks()
        self.lookup: CacheLookup | None = None
        self.decision: RouteDecision | None = None

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000

    def record(self, **fields: Any) -> None:
        """Queue a ledger write for after the response is sent."""
        if self.ledger is None:
            return
        lookup, decision = self.lookup, self.decision
        event = RequestEvent(
            request_id=self.id,
            namespace=self.namespace,
            requested_model=self.body.model,
            stream=self.body.stream,
            cache_status=lookup.status.value if lookup else None,
            cache_similarity=lookup.nearest_similarity if lookup else None,
            decision_tier=decision.tier.value if decision else None,
            decision_reason=decision.reason if decision else None,
            complexity_score=decision.score if decision else None,
            **fields,
        )
        self.tasks.add_task(self.ledger.record, event)


@router.post("/v1/chat/completions")
async def chat_completions(body: ChatCompletionRequest, request: Request) -> Response:
    ctx = _Request(request, body)
    payload = body.upstream_payload()

    if ctx.cache is not None:
        ctx.lookup = await ctx.cache.lookup(
            payload,
            model_scope=ctx.engine.cache_scope,
            namespace=ctx.namespace,
            bypass=request.headers.get(CACHE_CONTROL_HEADER, "").lower() == "bypass",
        )
        ctx.headers["x-finops-cache"] = ctx.lookup.status.value
        if ctx.lookup.nearest_similarity is not None:
            ctx.headers["x-finops-cache-similarity"] = f"{ctx.lookup.nearest_similarity:.4f}"
        if ctx.lookup.is_hit:
            return _serve_from_cache(ctx)

    ctx.decision = ctx.engine.decide(
        payload, request.headers.get(ROUTE_OVERRIDE_HEADER, "").lower()
    )
    ctx.headers["x-finops-route-reason"] = ctx.decision.reason
    ctx.headers["x-finops-complexity"] = str(ctx.decision.score)

    try:
        if body.stream and ctx.decision.tier is Tier.PREMIUM:
            return await _stream_premium(ctx, payload)
        # Local answers are checked before use, so they're buffered and replayed if
        # the client asked for a stream. Short local answers take well under a second.
        outcome = await ctx.engine.run(payload, ctx.decision)
    except ProviderError as exc:
        logger.warning(
            "request_id=%s route=%s status=%s error=%s",
            ctx.id,
            ctx.decision.tier.value,
            exc.status_code,
            exc.body,
        )
        ctx.record(
            status_code=exc.status_code,
            route=ctx.decision.tier.value,
            latency_ms=ctx.elapsed_ms(),
        )
        return JSONResponse(
            exc.body, status_code=exc.status_code, headers=ctx.headers, background=ctx.tasks
        )

    return _serve_outcome(ctx, outcome)


def _serve_outcome(ctx: _Request, outcome: RouteOutcome) -> Response:
    latency_ms = ctx.elapsed_ms()
    ctx.headers |= {
        "x-finops-route": outcome.route.value,
        "x-finops-provider": outcome.provider,
        "x-finops-latency-ms": f"{latency_ms:.1f}",
    }
    if outcome.escalated:
        ctx.headers["x-finops-escalated"] = outcome.escalation_reason
    prompt_tokens, completion_tokens = usage_tokens(outcome.response)
    _log(ctx, outcome.route.value, prompt_tokens, completion_tokens, latency_ms, outcome)

    # Cache and ledger writes run after the response is sent: no added latency.
    if ctx.cache and ctx.lookup:
        ctx.tasks.add_task(ctx.cache.store, ctx.lookup, outcome.response)
    ctx.record(
        status_code=200,
        route=outcome.route.value,
        latency_ms=latency_ms,
        escalation_reason=outcome.escalation_reason,
        provider=outcome.provider,
        upstream_model=outcome.response.get("model"),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        local_latency_ms=outcome.local_latency_ms,
        premium_latency_ms=outcome.premium_latency_ms,
    )
    if ctx.body.stream:
        return StreamingResponse(
            completion_to_sse(outcome.response),
            media_type="text/event-stream",
            headers=ctx.headers,
            background=ctx.tasks,
        )
    return JSONResponse(outcome.response, headers=ctx.headers, background=ctx.tasks)


async def _stream_premium(ctx: _Request, payload: dict[str, Any]) -> Response:
    """True passthrough streaming. Streamed premium answers aren't cached yet."""
    premium = ctx.engine.premium
    tap = StreamTap(await premium.stream(payload))
    ctx.headers |= {"x-finops-route": "premium", "x-finops-provider": premium.name}

    async def record_when_done() -> None:
        # Runs after the last chunk is sent, so the tap has seen the whole answer.
        prompt_tokens, completion_tokens = usage_tokens({"usage": tap.usage})
        estimated = tap.usage is None
        if estimated:
            prompt_tokens = ctx.decision.prompt_tokens_est
            completion_tokens = estimate_tokens(tap.text)
        latency_ms = ctx.elapsed_ms()
        _log(ctx, "premium", prompt_tokens, completion_tokens, latency_ms, stream=True)
        ctx.record(
            status_code=200,
            route="premium",
            latency_ms=latency_ms,
            provider=premium.name,
            upstream_model=tap.model or premium.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tokens_estimated=estimated,
            premium_latency_ms=latency_ms,
        )
        await ctx.tasks()

    return StreamingResponse(
        tap, media_type="text/event-stream", headers=ctx.headers, background=record_when_done
    )


def _serve_from_cache(ctx: _Request) -> Response:
    lookup = ctx.lookup
    completion: dict[str, Any] = {**lookup.match.response, "id": f"chatcmpl-{ctx.id}"}
    latency_ms = ctx.elapsed_ms()
    ctx.headers["x-finops-route"] = "cache"
    ctx.headers["x-finops-latency-ms"] = f"{latency_ms:.1f}"
    prompt_tokens, completion_tokens = usage_tokens(completion)
    _log(ctx, "cache", prompt_tokens, completion_tokens, latency_ms)

    ctx.tasks.add_task(ctx.cache.record_hit, lookup)
    ctx.record(
        status_code=200,
        route="cache",
        latency_ms=latency_ms,
        provider="cache",
        upstream_model=completion.get("model"),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    if ctx.body.stream:
        return StreamingResponse(
            completion_to_sse(completion),
            media_type="text/event-stream",
            headers=ctx.headers,
            background=ctx.tasks,
        )
    return JSONResponse(completion, headers=ctx.headers, background=ctx.tasks)


def _log(
    ctx: _Request,
    route: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
    outcome: RouteOutcome | None = None,
    stream: bool = False,
) -> None:
    logger.info(
        "request_id=%s route=%s cache=%s similarity=%s decision=%s reason=%s score=%s "
        "escalated=%s stream=%s prompt_tokens=%s completion_tokens=%s latency_ms=%.1f",
        ctx.id,
        route,
        ctx.lookup.status.value if ctx.lookup else "disabled",
        f"{ctx.lookup.nearest_similarity:.4f}"
        if ctx.lookup and ctx.lookup.nearest_similarity
        else None,
        ctx.decision.tier.value if ctx.decision else None,
        ctx.decision.reason if ctx.decision else None,
        ctx.decision.score if ctx.decision else None,
        outcome.escalation_reason if outcome else None,
        stream,
        prompt_tokens,
        completion_tokens,
        latency_ms,
    )
