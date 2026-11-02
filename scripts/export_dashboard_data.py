"""Save the benchmark numbers the dashboard shows to dashboard/data/results.json.

Benchmark results and the ledger stay on this machine, so the dashboard reads this small
committed file instead. It needs no database, Ollama or Groq.

    uv run python -m scripts.export_dashboard_data --run-id v3-40 --first-run full-1
"""

import argparse
import json
import math
from pathlib import Path

from benchmark import baseline
from benchmark.replay import load_results, run_dir
from benchmark.workload import WORKLOAD_FILE, load
from finops.core.config import PROJECT_ROOT, get_settings
from finops.ledger import PriceTable

OUTPUT = PROJECT_ROOT / "dashboard" / "data" / "results.json"
OLD_MODEL_RESULTS = "local-llama3.2_3b"
NEW_MODEL_RESULTS = "local-qwen3_4b-instruct"


def _judged_local(run_id: str) -> dict[str, str]:
    rows = map(json.loads, (run_dir(run_id) / "eval.jsonl").read_text().splitlines())
    return {
        r["id"]: r["verdict"]
        for r in rows
        if r["check"] == "local_vs_reference" and r["verdict"] in ("pass", "fail")
    }


def _quality(run_id: str, ids: set[str]) -> dict:
    verdicts = [v for i, v in _judged_local(run_id).items() if i in ids]
    passed, n = verdicts.count("pass"), len(verdicts)
    p = passed / n
    return {"passed": passed, "n": n, "rate": p, "margin": 1.96 * math.sqrt(p * (1 - p) / n)}


def _model_comparison(first_run: str) -> dict:
    """Old vs new free model on the same questions, from benchmark/localeval.py."""

    def answers(model_dir: str) -> list[dict]:
        path = run_dir(model_dir) / "answers.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    old, new = answers(OLD_MODEL_RESULTS), answers(NEW_MODEL_RESULTS)

    def math(rows: list[dict]) -> dict:
        done = [r for r in rows if r["set"] == "math" and not r["error"]]
        return {"passed": sum(bool(r["correct"]) for r in done), "n": len(done)}

    # The new model's graded answers, against the old model's grades on those same questions.
    new_verdicts = {
        r["id"]: r["verdict"]
        for r in new
        if r["set"] == "knowledge" and r["verdict"] in ("pass", "fail")
    }
    old_verdicts = _judged_local(first_run)
    same = [i for i in new_verdicts if i in old_verdicts]
    return {
        "math": {"old": math(old), "new": math(new)},
        "knowledge": {
            "old": {"passed": sum(old_verdicts[i] == "pass" for i in same), "n": len(same)},
            "new": {"passed": sum(new_verdicts[i] == "pass" for i in same), "n": len(same)},
        },
    }


def build(run_id: str, first_run: str) -> dict:
    settings = get_settings()
    prices = PriceTable.load(settings.ledger.pricing_file)
    items = {i.id: i for i in load(WORKLOAD_FILE)}
    results = load_results(run_dir(run_id) / "responses.jsonl")
    ledger = baseline.ledger_rows(settings.database_url, run_id)
    first_results = load_results(run_dir(first_run) / "responses.jsonl")
    first_ledger = baseline.ledger_rows(settings.database_url, first_run)
    ids = [r.id for r in results.values() if r.status == 200]

    measured = json.loads((run_dir(run_id) / "baseline.json").read_text())
    if not measured["complete"]:
        raise SystemExit("The measured baseline is incomplete; finish `make bench-baseline` first.")
    summary = json.loads((run_dir(run_id) / "summary.json").read_text())

    tokens = baseline.premium_tokens_by_prompt(results, ledger, items)
    calls = (run_dir(run_id) / "baseline_calls.jsonl").read_text().splitlines()
    for row in map(json.loads, calls):
        tokens[row["prompt"]] = (row["prompt_tokens"], row["completion_tokens"])

    routes = {r: {"requests": 0, "cost_usd": 0.0} for r in ("cache", "local", "premium")}
    for i in ids:
        route = routes[ledger[results[i].request_id]["route"]]
        route["requests"] += 1
        route["cost_usd"] += prices.cost(prices.baseline_model, *tokens[items[i].prompt])
    for name, route in routes.items():
        route["p50_seconds"] = summary["ledger"]["routes"][name]["p50_ms"] / 1000

    first_routes = [first_ledger[first_results[i].request_id]["route"] for i in ids]
    subset = set(ids)
    current, first = measured["runs"][run_id], measured["runs"][first_run]
    cache = summary["cache"]
    return {
        "requests": len(ids),
        "baseline_usd": measured["baseline_usd"],
        "baseline_model": measured["baseline_model"],
        "routes": routes,
        "current": {
            "savings": current["savings_pct"],
            "cost_usd": current["cost_usd"],
            "kept_off_paid": 1 - routes["premium"]["requests"] / len(ids),
            "quality": _quality(run_id, subset),
        },
        "first": {
            "savings": first["savings_pct"],
            "cost_usd": first["cost_usd"],
            "kept_off_paid": 1 - first_routes.count("premium") / len(ids),
            "route_requests": {r: first_routes.count(r) for r in routes},
            "quality": _quality(first_run, subset),
        },
        "cache": {k: {"hits": cache[k]["hits"], "n": cache[k]["n"]} for k in cache},
        "free_models": _model_comparison(first_run),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--first-run", required=True)
    args = parser.parse_args()
    data = build(args.run_id, args.first_run)
    path = Path(OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
