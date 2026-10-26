import pytest

from finops.core.config import PROJECT_ROOT
from finops.ledger import Ledger, PriceTable, RequestEvent
from finops.ledger.store import build_summary


def _event(route: str, status: int = 200, **fields) -> RequestEvent:
    defaults = {
        "request_id": f"req_{route}",
        "namespace": "default",
        "requested_model": "gpt-4o",
        "stream": False,
        "status_code": status,
        "route": route,
        "latency_ms": 10.0,
        "prompt_tokens": 1_000_000,
        "completion_tokens": 100_000,
    }
    return RequestEvent(**{**defaults, **fields})


def test_project_pricing_file_is_valid():
    prices = PriceTable.load(PROJECT_ROOT / "config" / "pricing.yaml")
    assert prices.models[prices.baseline_model].input_per_1m > 0
    assert prices.cost("llama3.2:3b", 1000, 1000) == 0.0


def test_pricing_rejects_unpriced_baseline():
    with pytest.raises(ValueError, match="no entry"):
        PriceTable(version="x", baseline_model="missing", premium_priced_as="missing", models={})


def test_cost_math(prices: PriceTable):
    # 1M input at $2 + 100k output at $10/M = $3.00
    assert prices.cost("baseline", 1_000_000, 100_000) == pytest.approx(3.0)
    assert prices.cost("unknown-model", 10, 10) is None


@pytest.mark.parametrize(
    ("event", "baseline", "cost", "billed"),
    [
        # Premium is priced as the baseline model; Groq's real price is "billed".
        (_event("premium", upstream_model="test-premium-model"), 3.0, 3.0, 0.21),
        (_event("local", upstream_model="test-local-model"), 3.0, 0.0, 0.0),
        (_event("cache", upstream_model="test-premium-model"), 3.0, 0.0, 0.0),
        (_event("premium", status=429), 0.0, 0.0, 0.0),
    ],
)
def test_price_by_route(ledger: Ledger, event, baseline, cost, billed):
    row = ledger.price(event)
    assert row.baseline_cost_usd == pytest.approx(baseline)
    assert row.cost_usd == pytest.approx(cost)
    assert row.saved_usd == pytest.approx(baseline - cost)
    assert row.billed_usd == pytest.approx(billed)
    assert row.pricing_version == "test"


def test_unknown_upstream_model_has_no_billed_cost(ledger: Ledger):
    row = ledger.price(_event("premium", upstream_model="mystery"))
    assert row.cost_usd == pytest.approx(3.0)
    assert row.billed_usd is None


def test_build_summary_headline_metrics():
    by_route = {
        "cache": {"requests": 2, "prompt_tokens": 20, "completion_tokens": 10,
                  "baseline_usd": 2.0, "cost_usd": 0, "saved_usd": 2.0, "billed_usd": 0,
                  "escalations": 0, "avg_ms": 5, "p50_ms": 5, "p95_ms": 6},
        "local": {"requests": 1, "prompt_tokens": 10, "completion_tokens": 5,
                  "baseline_usd": 1.0, "cost_usd": 0, "saved_usd": 1.0, "billed_usd": 0,
                  "escalations": 0, "avg_ms": 500, "p50_ms": 500, "p95_ms": 500},
        "premium": {"requests": 1, "prompt_tokens": 30, "completion_tokens": 15,
                    "baseline_usd": 1.0, "cost_usd": 1.0, "saved_usd": 0, "billed_usd": 0.1,
                    "escalations": 1, "avg_ms": 800, "p50_ms": 800, "p95_ms": 800},
    }  # fmt: skip

    summary = build_summary(by_route, errors=1)

    assert summary["requests"] == 4
    assert summary["errors"] == 1
    assert summary["deflection_rate"] == pytest.approx(0.75)
    assert summary["token_deflection_rate"] == pytest.approx(45 / 90)
    assert summary["escalations"] == 1
    assert summary["cost"]["savings_pct"] == pytest.approx(0.75)
    assert summary["cost"]["billed_usd"] == pytest.approx(0.1)
    assert summary["routes"]["cache"]["p50_ms"] == 5


def test_empty_summary_has_no_division_errors():
    summary = build_summary({}, errors=0)
    assert summary["deflection_rate"] == 0.0
    assert summary["cost"]["savings_pct"] == 0.0
    assert summary["routes"]["local"]["p50_ms"] is None
