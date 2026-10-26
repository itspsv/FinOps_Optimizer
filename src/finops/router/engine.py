"""Routing workflow as a LangGraph state machine.

    START ─▶ (decision.tier) ─┬─ local ──▶ answer ok? ── yes ──▶ END
                              │              │ no / local down
                              │              ▼
                              └─ premium ◀───┘ ──▶ END

The classifier runs before the graph (see `decide`) because streaming premium
requests skip the graph and pass straight through.
"""

import time
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from finops.core.config import RoutingSettings
from finops.providers import Provider, ProviderError
from finops.providers.breaker import CircuitBreaker
from finops.router.classifier import RouteDecision, Tier, classify
from finops.router.validators import validate_local_answer


class _State(TypedDict, total=False):
    payload: dict[str, Any]
    decision: RouteDecision
    response: dict[str, Any]
    route: Tier
    provider: str
    escalation_reason: str
    local_latency_ms: float
    premium_latency_ms: float


@dataclass(frozen=True)
class RouteOutcome:
    response: dict[str, Any]
    route: Tier
    provider: str
    decision: RouteDecision
    escalation_reason: str | None = None
    local_latency_ms: float | None = None
    premium_latency_ms: float | None = None

    @property
    def escalated(self) -> bool:
        return self.escalation_reason is not None


class RoutingEngine:
    def __init__(
        self,
        premium: Provider,
        local: Provider | None,
        settings: RoutingSettings,
        breaker: CircuitBreaker | None = None,
    ):
        self.premium = premium
        self.local = local if settings.enabled else None
        self.settings = settings
        self.breaker = breaker or CircuitBreaker(
            settings.breaker_failure_threshold, settings.breaker_cooldown_s
        )
        self._graph = self._build_graph()

    @property
    def cache_scope(self) -> str:
        """Cache key component: answers depend on the routing policy, not one model."""
        if self.local is None:
            return self.premium.model
        return f"router:{self.settings.policy_version}:{self.premium.model}"

    def decide(self, payload: dict[str, Any], override: str | None = None) -> RouteDecision:
        if self.local is None:
            return RouteDecision(Tier.PREMIUM, "routing_disabled")
        return classify(payload, self.settings, override)

    async def run(self, payload: dict[str, Any], decision: RouteDecision) -> RouteOutcome:
        """Execute the decision. Raises ProviderError if premium (the last resort) fails."""
        state = await self._graph.ainvoke({"payload": payload, "decision": decision})
        return RouteOutcome(
            response=state["response"],
            route=state["route"],
            provider=state["provider"],
            decision=decision,
            escalation_reason=state.get("escalation_reason"),
            local_latency_ms=state.get("local_latency_ms"),
            premium_latency_ms=state.get("premium_latency_ms"),
        )

    def _build_graph(self):
        graph = StateGraph(_State)
        graph.add_node("local", self._local_node)
        graph.add_node("premium", self._premium_node)
        graph.add_conditional_edges(
            START,
            lambda s: "local" if s["decision"].tier is Tier.LOCAL else "premium",
            ["local", "premium"],
        )
        graph.add_conditional_edges(
            "local", lambda s: END if "response" in s else "premium", ["premium", END]
        )
        graph.add_edge("premium", END)
        return graph.compile()

    async def _local_node(self, state: _State) -> dict[str, Any]:
        if self.local is None or not self.breaker.allow():
            return {"escalation_reason": "local_unavailable"}

        payload = state["payload"]
        local_payload = {**payload}
        if self.settings.local_system_prompt:
            local_payload["messages"] = [
                {"role": "system", "content": self.settings.local_system_prompt},
                *payload["messages"],
            ]
        if not (payload.get("max_tokens") or payload.get("max_completion_tokens")):
            local_payload["max_tokens"] = self.settings.local_max_completion_tokens

        started = time.perf_counter()
        try:
            response = await self.local.chat(local_payload)
        except ProviderError as exc:
            self.breaker.record_failure()
            return {
                "escalation_reason": f"local_error_{exc.status_code}",
                "local_latency_ms": _elapsed_ms(started),
            }
        self.breaker.record_success()
        latency = _elapsed_ms(started)

        if problem := validate_local_answer(response, payload):
            return {"escalation_reason": f"local_{problem}", "local_latency_ms": latency}
        return {
            "response": response,
            "route": Tier.LOCAL,
            "provider": self.local.name,
            "local_latency_ms": latency,
        }

    async def _premium_node(self, state: _State) -> dict[str, Any]:
        started = time.perf_counter()
        response = await self.premium.chat(state["payload"])
        return {
            "response": response,
            "route": Tier.PREMIUM,
            "provider": self.premium.name,
            "premium_latency_ms": _elapsed_ms(started),
        }


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
