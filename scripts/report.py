"""Print the FinOps summary from the ledger.

    uv run python scripts/report.py [--namespace NS] [--since-minutes N] [--json]
"""

import argparse
import asyncio
import json

from finops.core.config import get_settings
from finops.db.pool import open_pool
from finops.ledger import Ledger, PriceTable
from finops.ledger.store import ROUTES, PgLedgerStore


def _fmt_ms(value: float | None) -> str:
    return f"{value:,.1f}" if value is not None else "-"


def render(summary: dict) -> str:
    cost, pricing = summary["cost"], summary["pricing"]
    lines = [
        "AI FinOps Optimizer: summary",
        "",
        f"  Requests served          {summary['requests']:>10,}   (errors: {summary['errors']})",
        f"  Deflection rate          {summary['deflection_rate']:>10.1%}   kept off premium",
        f"  Token deflection rate    {summary['token_deflection_rate']:>10.1%}",
        f"  Escalations              {summary['escalations']:>10,}   local -> premium",
        "",
        f"  Baseline cost            ${cost['baseline_usd']:>11,.4f}   "
        f"all traffic on {pricing['baseline_model']}",
        f"  Actual cost              ${cost['actual_usd']:>11,.4f}   "
        f"premium priced as {pricing['premium_priced_as']}",
        f"  Saved                    ${cost['saved_usd']:>11,.4f}   ({cost['savings_pct']:.1%})",
        f"  Billed by Groq           ${cost['billed_usd']:>11,.4f}   (real dev spend)",
        "",
        f"  {'route':<10}{'requests':>10}{'tokens':>12}{'avg ms':>11}{'p50 ms':>11}{'p95 ms':>11}",
    ]
    for route in ROUTES:
        r = summary["routes"][route]
        lines.append(
            f"  {route:<10}{r['requests']:>10,}{r['tokens']:>12,}{_fmt_ms(r['avg_ms']):>11}"
            f"{_fmt_ms(r['p50_ms']):>11}{_fmt_ms(r['p95_ms']):>11}"
        )
    lines += ["", f"  Pricing version {pricing['version']}"]
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace")
    parser.add_argument("--since-minutes", type=int)
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    args = parser.parse_args()

    settings = get_settings()
    pool = await open_pool(settings.database_url)
    try:
        ledger = Ledger(PgLedgerStore(pool), PriceTable.load(settings.ledger.pricing_file))
        summary = await ledger.summary(args.namespace, args.since_minutes)
    finally:
        await pool.close()
    print(json.dumps(summary, indent=2) if args.json else render(summary))


if __name__ == "__main__":
    asyncio.run(main())
