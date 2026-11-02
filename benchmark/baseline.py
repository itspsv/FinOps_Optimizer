"""Measure what a run's requests would really cost if all of them went to premium.

The ledger's baseline prices the answer actually served. Local answers are told to be brief,
so that understates the cost of requests that didn't go to premium, and two runs with
different routing get different baselines for identical requests. This sends each of those
prompts to the premium model once, the way a normal client would, and records the tokens used.
Calls go straight to Groq, so they don't appear in any run's ledger. Re-running resumes.

    uv run python -m benchmark.baseline --run-id v3-40 --compare-run full-1
"""

import argparse
import asyncio
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import psycopg

from benchmark.evaluate import DailyLimitReached, TokenBudget
from benchmark.replay import ReplayResult, load_results, run_dir
from benchmark.workload import WORKLOAD_FILE, WorkItem, load
from finops.core.config import get_settings
from finops.ledger import PriceTable
from finops.providers import OpenAICompatibleProvider, Provider, ProviderError

Tokens = tuple[int, int]  # prompt, completion
RESERVED_OUTPUT_TOKENS = 1500  # budget for answers until the API reports real usage


def ledger_rows(database_url: str, run_id: str) -> dict[str, dict[str, Any]]:
    query = (
        "SELECT request_id, route, prompt_tokens, completion_tokens, cost_usd FROM request_log "
        "WHERE namespace = %s AND status_code < 400"
    )
    with psycopg.connect(database_url) as conn:
        rows = conn.execute(query, (f"bench-{run_id}",)).fetchall()
    return {r[0]: {"route": r[1], "tokens": (r[2], r[3]), "cost_usd": float(r[4])} for r in rows}


def premium_tokens_by_prompt(
    results: dict[str, ReplayResult], ledger: dict[str, dict], items: dict[str, WorkItem]
) -> dict[str, Tokens]:
    """Tokens already measured: requests this run really sent to premium."""
    known: dict[str, Tokens] = {}
    for result in results.values():
        row = ledger.get(result.request_id or "")
        if row and row["route"] == "premium":
            known.setdefault(items[result.id].prompt, row["tokens"])
    return known


def prompts_to_measure(ids: Iterable[str], items: dict[str, WorkItem], known: dict) -> list[str]:
    return list(dict.fromkeys(items[i].prompt for i in ids if items[i].prompt not in known))


def compute(
    ids: list[str],
    items: dict[str, WorkItem],
    tokens: dict[str, Tokens],
    prices: PriceTable,
    runs: dict[str, tuple[dict[str, ReplayResult], dict[str, dict]]],
) -> dict[str, Any]:
    """Savings for each run against one baseline measured on the same requests."""
    missing = [i for i in ids if items[i].prompt not in tokens]
    baseline = sum(
        prices.cost(prices.baseline_model, *tokens[items[i].prompt]) or 0.0
        for i in ids
        if i not in missing
    )
    out: dict[str, Any] = {
        "requests": len(ids),
        "unmeasured_requests": len(missing),
        "complete": not missing,
        "baseline_model": prices.baseline_model,
        "pricing_version": prices.version,
        "baseline_usd": round(baseline, 6),
        "runs": {},
    }
    for name, (results, ledger) in runs.items():
        rows = [ledger.get(results[i].request_id or "") for i in ids if i in results]
        rows = [r for r in rows if r]
        cost = sum(r["cost_usd"] for r in rows)
        out["runs"][name] = {
            "requests": len(rows),
            "premium_requests": sum(r["route"] == "premium" for r in rows),
            "cost_usd": round(cost, 6),
            "saved_usd": round(baseline - cost, 6),
            "savings_pct": (baseline - cost) / baseline if baseline else 0.0,
        }
    return out


async def measure(
    prompts: list[str], provider: Provider, budget: TokenBudget, out_file: Path, sleep=time.sleep
) -> dict[str, Tokens]:
    """Send each prompt once with a normal request. Stops early at Groq's daily cap."""
    measured: dict[str, Tokens] = {}
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            row = json.loads(line)
            measured[row["prompt"]] = (row["prompt_tokens"], row["completion_tokens"])
    todo = [p for p in prompts if p not in measured]
    print(f"Measuring {len(todo)} prompts on premium ({len(measured)} already done)", flush=True)
    with out_file.open("a") as f:
        for n, prompt in enumerate(todo, 1):
            payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0}
            usage = None
            for attempt in range(1, 7):
                budget.reserve(len(prompt) // 4 + RESERVED_OUTPUT_TOKENS)
                try:
                    usage = (await provider.chat(payload)).get("usage") or {}
                    break
                except ProviderError as exc:
                    if exc.status_code == 429 and "per day" in json.dumps(exc.body):
                        raise DailyLimitReached(exc.body["error"]["message"]) from exc
                    if exc.status_code not in (429, 500, 502, 503, 504) or attempt == 6:
                        raise
                    print(f"    Groq HTTP {exc.status_code}, retrying", flush=True)
                    sleep(20 * attempt)
            budget.correct_last(usage.get("total_tokens", 0))
            row = (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
            measured[prompt] = row
            f.write(
                json.dumps(
                    {"prompt": prompt, "prompt_tokens": row[0], "completion_tokens": row[1]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)
    return measured


async def amain(run_id: str, compare_runs: list[str], tpm: int) -> None:
    settings = get_settings()
    prices = PriceTable.load(settings.ledger.pricing_file)
    items = {i.id: i for i in load(WORKLOAD_FILE)}
    results = load_results(run_dir(run_id) / "responses.jsonl")
    ledger = ledger_rows(settings.database_url, run_id)
    ids = [r.id for r in results.values() if r.status == 200]

    tokens = premium_tokens_by_prompt(results, ledger, items)
    out = run_dir(run_id)
    calls_file = out / "baseline_calls.jsonl"
    prompts = prompts_to_measure(ids, items, tokens)
    provider = OpenAICompatibleProvider(settings.premium)
    try:
        tokens |= await measure(prompts, provider, TokenBudget(tpm), calls_file)
    except DailyLimitReached as exc:
        print(f"Stopping: Groq daily token cap reached. {exc}", flush=True)
        if calls_file.exists():
            for line in calls_file.read_text().splitlines():
                row = json.loads(line)
                tokens[row["prompt"]] = (row["prompt_tokens"], row["completion_tokens"])
    finally:
        await provider.aclose()

    runs = {run_id: (results, ledger)}
    for other in compare_runs:
        runs[other] = (
            load_results(run_dir(other) / "responses.jsonl"),
            ledger_rows(settings.database_url, other),
        )
    summary = compute(ids, items, tokens, prices, runs)
    (out / "baseline.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nNext: make bench-report RUN={run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--compare-run", action="append", default=[])
    parser.add_argument("--tpm", type=int, default=7000)
    args = parser.parse_args()
    asyncio.run(amain(args.run_id, args.compare_run, args.tpm))


if __name__ == "__main__":
    main()
