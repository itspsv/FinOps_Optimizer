"""Combine ledger, replay, evaluation and calibration results into one report.

make bench-report RUN=<run id>   ->  benchmark/results/<run id>/report.md + summary.json
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from benchmark.calibrate import RESULT_FILE as CALIBRATION_FILE
from benchmark.replay import ReplayResult, load_results, run_dir
from benchmark.workload import WORKLOAD_FILE, WorkItem, load
from finops.core.config import get_settings
from finops.db.pool import open_pool
from finops.ledger import Ledger, PriceTable
from finops.ledger.store import PgLedgerStore


def routing_accuracy(items: dict[str, WorkItem], results: dict[str, ReplayResult]) -> dict:
    """Router decisions vs the expected tier of each source (cache hits excluded)."""
    rows = []
    for r in results.values():
        item = items[r.id]
        if r.status != 200 or r.route == "cache" or item.expected_tier is None:
            continue
        if item.source == "gsm8k":  # local or premium depending on routing policy
            continue
        decided = "local" if r.route == "local" or r.escalated else "premium"
        rows.append((item.expected_tier, decided))
    confusion = {
        f"expected_{e}_got_{d}": sum(1 for x in rows if x == (e, d))
        for e in ("local", "premium")
        for d in ("local", "premium")
    }
    correct = sum(e == d for e, d in rows)
    return {"n": len(rows), "accuracy": correct / len(rows) if rows else None, **confusion}


def cache_behaviour(items: dict[str, WorkItem], results: dict[str, ReplayResult]) -> dict:
    def hit_rate(kind: str) -> dict:
        rows = [r for r in results.values() if items[r.id].kind == kind and r.status == 200]
        hits = sum((r.cache or "").startswith("hit") for r in rows)
        return {"n": len(rows), "hits": hits, "hit_rate": hits / len(rows) if rows else None}

    return {
        "repeats": hit_rate("repeat"),
        "paraphrases": hit_rate("paraphrase"),  # should hit
        "lookalikes": hit_rate("near_miss"),  # should NOT hit
    }


def build_summary(
    run_id: str,
    ledger: dict,
    items: dict[str, WorkItem],
    results: dict[str, ReplayResult],
    evaluation: dict | None,
    calibration: dict | None,
    measured_baseline: dict | None = None,
) -> dict[str, Any]:
    routes = ledger["routes"]
    cache_p50, premium_p50 = routes["cache"]["p50_ms"], routes["premium"]["p50_ms"]
    served = ledger["requests"] or 1
    return {
        "run_id": run_id,
        "workload": {
            "requests": len(results),
            "sources": sorted({items[r].source for r in results}),
        },
        "ledger": ledger,
        "shares": {r: routes[r]["requests"] / served for r in routes},
        "speedup_cache_vs_premium": premium_p50 / cache_p50 if cache_p50 and premium_p50 else None,
        "routing": routing_accuracy(items, results),
        "cache": cache_behaviour(items, results),
        "evaluation": evaluation,
        "measured_baseline": measured_baseline,
        "calibration": calibration
        and {
            "pairs": calibration["pairs"],
            "chosen": calibration["chosen"],
            "at_0.92": next((p for p in calibration["sweep"] if p["threshold"] == 0.92), None),
        },
    }


def _pct(x: float | None) -> str:
    return f"{x:.1%}" if x is not None else "n/a"


def _ms(x: float | None) -> str:
    return f"{x:,.0f} ms" if x is not None else "n/a"


def render(s: dict) -> str:
    led, cost, routes = s["ledger"], s["ledger"]["cost"], s["ledger"]["routes"]
    pricing, ev, cal = led["pricing"], s["evaluation"] or {}, s["calibration"]
    shares, cache = s["shares"], s["cache"]
    lq, sp = ev.get("local_quality") or {}, ev.get("semantic_hit_precision") or {}
    mb = s.get("measured_baseline")
    own = mb["runs"].get(s["run_id"]) if mb and mb["complete"] else None
    measured_row = (
        [
            f"| **API cost saved vs. measured all-premium** | **{_pct(own['savings_pct'])}** "
            f"(${own['saved_usd']:.4f} of ${mb['baseline_usd']:.4f}) |"
        ]
        if own
        else []
    )
    measured_cost = []
    if own:
        measured_cost = [
            "",
            f"**Measured baseline:** all {mb['requests']} requests on `{mb['baseline_model']}` at "
            f"list price, using token counts measured on the premium model: "
            f"${mb['baseline_usd']:.4f}. Unlike the ledger baseline above, this is the same for "
            "any routing policy, so these runs can be compared:",
            "",
            "| Run | Premium requests | Cost | Saved |",
            "|---|---|---|---|",
            *(
                f"| {name} | {r['premium_requests']} | ${r['cost_usd']:.4f} | "
                f"{_pct(r['savings_pct'])} |"
                for name, r in mb["runs"].items()
            ),
        ]
    headline_savings = own["savings_pct"] if own else cost["savings_pct"]

    lines = [
        f"# Benchmark report: {s['run_id']}",
        "",
        f"Replayed **{s['workload']['requests']} requests** built from public datasets "
        f"({', '.join(s['workload']['sources'])}) through the proxy, one at a time.",
        "",
        "## Headline",
        "",
        "| Metric | Result |",
        "|---|---|",
        f"| Requests kept off the premium API | **{_pct(led['deflection_rate'])}** |",
        f"| Tokens kept off the premium API | {_pct(led['token_deflection_rate'])} |",
        f"| API cost saved vs. ledger baseline (served tokens) | **{_pct(cost['savings_pct'])}** "
        f"(${cost['saved_usd']:.4f} of ${cost['baseline_usd']:.4f}) |",
        *measured_row,
        f"| Cache p50 latency vs. premium p50 | **{_ms(routes['cache']['p50_ms'])}** vs. "
        f"{_ms(routes['premium']['p50_ms'])}"
        + (
            f" ({s['speedup_cache_vs_premium']:.0f}x faster)"
            if s["speedup_cache_vs_premium"]
            else ""
        )
        + " |",
        f"| Local answers judged as good as premium | {_pct(lq.get('pass_rate'))} "
        f"(n={lq.get('n', 0)}) |",
        f"| Semantic cache hits judged correct | {_pct(sp.get('pass_rate'))} "
        f"(n={sp.get('n', 0)}) |",
        "",
        "## Where requests went",
        "",
        "| Route | Share | Requests | Tokens | p50 | p95 |",
        "|---|---|---|---|---|---|",
    ]
    for route in ("cache", "local", "premium"):
        r = routes[route]
        lines.append(
            f"| {route} | {_pct(shares[route])} | {r['requests']} | {r['tokens']:,} | "
            f"{_ms(r['p50_ms'])} | {_ms(r['p95_ms'])} |"
        )
    lines += [
        "",
        f"Errors: {led['errors']}. Escalations (local answer rejected, premium answered): "
        f"{led['escalations']}.",
        "",
        "## Cost",
        "",
        f"- **Baseline:** every request on `{pricing['baseline_model']}` at list price: "
        f"${cost['baseline_usd']:.4f}",
        f"- **With the optimizer:** premium priced as `{pricing['premium_priced_as']}`, local "
        f"and cache $0: ${cost['actual_usd']:.4f}",
        f"- **Actually billed by Groq** during the run: ${cost['billed_usd']:.4f}",
        f"- Pricing version `{pricing['version']}`",
        *measured_cost,
        "",
        "## Cache behaviour",
        "",
        "| Traffic | Expected | Hit rate |",
        "|---|---|---|",
        f"| Exact repeats | hit | {_pct(cache['repeats']['hit_rate'])} "
        f"({cache['repeats']['hits']}/{cache['repeats']['n']}) |",
        f"| Human paraphrases (Quora duplicates) | hit | {_pct(cache['paraphrases']['hit_rate'])} "
        f"({cache['paraphrases']['hits']}/{cache['paraphrases']['n']}) |",
        f"| Look-alike questions (Quora non-duplicates) | miss | "
        f"{_pct(cache['lookalikes']['hit_rate'])} "
        f"({cache['lookalikes']['hits']}/{cache['lookalikes']['n']}) |",
    ]
    if cal:
        chosen, old = cal["chosen"], cal["at_0.92"]
        lines += [
            "",
            f"Threshold **{chosen['threshold']}**, calibrated on {cal['pairs']:,} labeled Quora "
            f"pairs: precision {chosen['precision']:.3f}, recall {chosen['recall']:.3f}"
            + (f" (at 0.92: precision {old['precision']:.3f})." if old else "."),
        ]
    rt = s["routing"]
    lines += [
        "",
        "## Routing",
        "",
        f"Router decisions matched the expected tier for **{_pct(rt['accuracy'])}** of "
        f"{rt['n']} labeled requests (simple Dolly categories expected local; HumanEval expected "
        "premium; GSM8K not counted). These labels predate the grounded policy, which sends "
        "open_qa to premium by design.",
        "",
        f"- Expected local, sent premium: {rt['expected_local_got_premium']} "
        "(route reasons are in responses.jsonl)",
        f"- Expected premium, sent local: {rt['expected_premium_got_local']}",
    ]
    if ev.get("gsm8k"):
        lines += ["", "GSM8K exact-match accuracy by route:", ""]
        for route, g in sorted(ev["gsm8k"].items()):
            lines.append(f"- {route}: {g['correct']}/{g['n']} ({_pct(g['accuracy'])})")
    lines += [
        "",
        "## Assumptions and limits",
        "",
        "- Premium calls were served by Groq but priced at the baseline model's list price, to "
        "model a company paying commercial API rates.",
        "- Baseline cost uses the tokens of the answer actually served. Local answers are told "
        "to be brief, so they are shorter than a premium answer to the same prompt. This "
        "understates baseline and savings, and savings percentages from runs with different "
        "routing are not directly comparable.",
        "- The workload's repeat and paraphrase rates are design choices; savings scale with how "
        "repetitive real traffic is.",
        f"- Quality is judged by `{ev.get('judge_model', 'n/a')}`, a sample-based LLM judge, not "
        "human review.",
        '- Local requests get a short "be concise" system prompt; the judge grades correctness '
        "and whether the question was answered, not length.",
        "- Single machine (MacBook Air M1, 8 GB), requests sent sequentially.",
        "",
        "## Resume-ready lines",
        "",
        f"- Cut simulated LLM API spend by **{_pct(headline_savings)}** on a "
        f"{s['workload']['requests']}-request replay of public prompts by serving "
        f"{_pct(shares['cache'])} of traffic from an exact and semantic cache and routing "
        f"{_pct(shares['local'])} to a local model.",
    ]
    if s["speedup_cache_vs_premium"]:
        lines.append(
            f"- Served cache hits at {_ms(routes['cache']['p50_ms'])} p50 vs. "
            f"{_ms(routes['premium']['p50_ms'])} for premium API calls "
            f"(**{s['speedup_cache_vs_premium']:.0f}x faster**)."
        )
    if cal and cal["at_0.92"]:
        lines.append(
            f"- Calibrated the semantic-cache threshold on {cal['pairs']:,} labeled pairs, "
            f"raising hit precision from {cal['at_0.92']['precision']:.0%} to "
            f"{cal['chosen']['precision']:.0%}"
            + (
                f"; an LLM judge rated {_pct(lq['pass_rate'])} of local answers as matching "
                "premium quality."
                if lq.get("pass_rate") is not None
                else "."
            )
        )
    return "\n".join(lines) + "\n"


async def ledger_summary(namespace: str) -> dict:
    settings = get_settings()
    pool = await open_pool(settings.database_url)
    try:
        ledger = Ledger(PgLedgerStore(pool), PriceTable.load(settings.ledger.pricing_file))
        return await ledger.summary(namespace)
    finally:
        await pool.close()


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    out = run_dir(args.run_id)
    results = load_results(out / "responses.jsonl")
    if not results:
        raise SystemExit(f"No responses for run {args.run_id}.")
    items = {i.id: i for i in load(WORKLOAD_FILE)}
    summary = build_summary(
        args.run_id,
        asyncio.run(ledger_summary(f"bench-{args.run_id}")),
        items,
        results,
        _read_json(out / "eval_summary.json"),
        _read_json(CALIBRATION_FILE),
        _read_json(out / "baseline.json"),
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "report.md").write_text(render(summary))
    print(render(summary))
    print(f"Saved {out / 'report.md'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
