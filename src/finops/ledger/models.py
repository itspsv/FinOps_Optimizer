from dataclasses import dataclass


@dataclass(frozen=True)
class RequestEvent:
    """Facts about one proxied request, as observed by the API layer."""

    request_id: str
    namespace: str
    requested_model: str | None
    stream: bool
    status_code: int
    route: str  # cache | local | premium
    latency_ms: float
    cache_status: str | None = None
    cache_similarity: float | None = None
    decision_tier: str | None = None
    decision_reason: str | None = None
    complexity_score: int | None = None
    escalation_reason: str | None = None
    provider: str | None = None
    upstream_model: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_estimated: bool = False
    local_latency_ms: float | None = None
    premium_latency_ms: float | None = None


@dataclass(frozen=True)
class LedgerRow:
    event: RequestEvent
    pricing_version: str
    baseline_model: str
    baseline_cost_usd: float
    cost_usd: float
    saved_usd: float
    billed_usd: float | None
