"""Turn request events into priced ledger rows.

Cost model (all at list prices from config/pricing.yaml):

    baseline = tokens served x baseline model price   (no optimizer: everything premium)
    cost     = premium route -> tokens x `premium_priced_as` price
               local route   -> tokens x local model price ($0 on our own hardware)
               cache route   -> $0
    saved    = baseline - cost

Token counts are those of the answer actually served. For local answers that is the
local model's own count, which the benchmark report states as an assumption.
"""

import logging
from typing import Any

from finops.ledger.models import LedgerRow, RequestEvent
from finops.ledger.pricing import PriceTable
from finops.ledger.store import LedgerStore

logger = logging.getLogger(__name__)


class Ledger:
    def __init__(self, store: LedgerStore, prices: PriceTable):
        self._store = store
        self.prices = prices

    def price(self, event: RequestEvent) -> LedgerRow:
        p, c = event.prompt_tokens, event.completion_tokens
        if event.status_code >= 400:
            baseline = cost = 0.0
            billed: float | None = 0.0
        else:
            baseline = self.prices.cost(self.prices.baseline_model, p, c) or 0.0
            if event.route == "premium":
                cost = self.prices.cost(self.prices.premium_priced_as, p, c) or 0.0
                billed = self.prices.cost(event.upstream_model, p, c)
            elif event.route == "local":
                cost = self.prices.cost(event.upstream_model, p, c) or 0.0
                billed = 0.0
            else:
                cost = billed = 0.0
        return LedgerRow(
            event=event,
            pricing_version=self.prices.version,
            baseline_model=self.prices.baseline_model,
            baseline_cost_usd=baseline,
            cost_usd=cost,
            saved_usd=baseline - cost,
            billed_usd=billed,
        )

    async def record(self, event: RequestEvent) -> None:
        """Runs after the response is sent; never raises."""
        try:
            await self._store.insert(self.price(event))
        except Exception:
            logger.exception("ledger write failed for %s", event.request_id)

    async def summary(self, namespace: str | None = None, since_minutes: int | None = None):
        stats = await self._store.summary(namespace, since_minutes)
        return {
            **stats,
            "pricing": {
                "version": self.prices.version,
                "baseline_model": self.prices.baseline_model,
                "premium_priced_as": self.prices.premium_priced_as,
            },
        }


def usage_tokens(response: dict[str, Any] | None) -> tuple[int, int]:
    usage = (response or {}).get("usage") or {}
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
