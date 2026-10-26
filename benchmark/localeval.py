"""Compare local models head-to-head on fixed questions, calling Ollama directly.

    uv run python -m benchmark.localeval --model qwen3:4b-instruct            # answer (free)
    uv run python -m benchmark.localeval --model qwen3:4b-instruct --judge    # + LLM judge

Question sets (same for every model):
  math       the workload's 30 GSM8K problems, scored by exact final number (free)
  knowledge  the questions judged in a baseline run (default full-1), so the judge can reuse
             that run's Groq reference answers and only pays for one judge call per answer

Requests mirror what the proxy sends a local model: the same brevity system prompt and
the same max_tokens cap. Results go to benchmark/results/local-<model>/; re-runs resume.
"""

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark.evaluate import (
    LOCAL_TEMPLATE,
    MAX_CHARS,
    DailyLimitReached,
    GroqClient,
    TokenBudget,
    collect_references,
    final_number,
)
from benchmark.replay import RESULTS_DIR
from benchmark.workload import WORKLOAD_FILE, WorkItem, load
from finops.core.config import get_settings
from finops.providers import OpenAICompatibleProvider, ProviderError


@dataclass
class LocalAnswer:
    id: str
    set: str  # math | knowledge
    answer: str | None
    finish_reason: str | None
    completion_tokens: int
    latency_ms: float
    error: str | None = None
    correct: bool | None = None  # math: exact match
    verdict: str | None = None  # knowledge: pass | fail, once judged
    reason: str | None = None


def out_dir(model: str) -> Path:
    return RESULTS_DIR / f"local-{model.replace(':', '_').replace('/', '_')}"


def question_sets(items: dict[str, WorkItem], baseline_run: str) -> list[tuple[str, WorkItem]]:
    math = [("math", i) for i in items.values() if i.source == "gsm8k"]
    judged = RESULTS_DIR / baseline_run / "eval.jsonl"
    knowledge_ids = sorted(
        {
            row["id"]
            for row in map(json.loads, judged.read_text().splitlines())
            if row["check"] == "local_vs_reference" and row["verdict"] in ("pass", "fail")
        }
    )
    return math + [("knowledge", items[i]) for i in knowledge_ids]


def _load(path: Path) -> dict[tuple[str, str], LocalAnswer]:
    if not path.exists():
        return {}
    rows = [LocalAnswer(**json.loads(line)) for line in path.read_text().splitlines() if line]
    return {(r.set, r.id): r for r in rows}


def _save(path: Path, answers: dict[tuple[str, str], LocalAnswer]) -> None:
    path.write_text(
        "".join(json.dumps(asdict(a), ensure_ascii=False) + "\n" for a in answers.values())
    )


async def answer_all(model: str, questions: list[tuple[str, WorkItem]], path: Path) -> dict:
    settings = get_settings()
    local = OpenAICompatibleProvider(settings.local.model_copy(update={"model": model}))
    routing = settings.routing
    answers = _load(path)
    todo = [(s, i) for s, i in questions if (s, i.id) not in answers or answers[(s, i.id)].error]
    print(
        f"{model}: {len(todo)} questions to answer ({len(questions) - len(todo)} done)", flush=True
    )
    try:
        for n, (qset, item) in enumerate(todo, 1):
            messages = [{"role": "user", "content": item.prompt}]
            if routing.local_system_prompt:
                messages.insert(0, {"role": "system", "content": routing.local_system_prompt})
            started = time.perf_counter()
            try:
                r = await local.chat(
                    {
                        "messages": messages,
                        "temperature": 0,
                        "max_tokens": routing.local_max_completion_tokens,
                    }
                )
                choice = r["choices"][0]
                text = choice["message"].get("content")
                a = LocalAnswer(
                    item.id,
                    qset,
                    text,
                    choice.get("finish_reason"),
                    (r.get("usage") or {}).get("completion_tokens", 0),
                    round((time.perf_counter() - started) * 1000, 1),
                )
                if qset == "math":
                    a.correct = final_number(text) == final_number(item.reference)
            except ProviderError as exc:
                a = LocalAnswer(item.id, qset, None, None, 0, 0.0, error=str(exc.status_code))
            answers[(qset, item.id)] = a
            _save(path, answers)
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)
    finally:
        await local.aclose()
    return answers


async def judge_all(answers: dict, items: dict[str, WorkItem], path: Path, tpm: int) -> None:
    refs = collect_references(RESULTS_DIR)
    pending = [
        a for a in answers.values()
        if a.set == "knowledge" and a.verdict in (None, "error") and a.answer and a.id in refs
    ]  # fmt: skip
    print(f"Judging {len(pending)} knowledge answers against stored references", flush=True)
    provider = OpenAICompatibleProvider(get_settings().premium)
    groq = GroqClient(provider, TokenBudget(tpm))
    try:
        for n, a in enumerate(pending, 1):
            prompt = LOCAL_TEMPLATE.format(
                question=items[a.id].prompt[:MAX_CHARS],
                reference=refs[a.id][:MAX_CHARS],
                candidate=a.answer[:MAX_CHARS],
            )
            try:
                a.verdict, a.reason = await groq.judge(prompt)
            except DailyLimitReached as exc:
                print(f"Stopping: Groq daily token cap reached. {exc}", flush=True)
                break
            except ProviderError as exc:
                a.verdict, a.reason = "error", f"Groq error {exc.status_code}"
            _save(path, answers)
            if n % 10 == 0 or n == len(pending):
                print(f"  {n}/{len(pending)} judged", flush=True)
    finally:
        await provider.aclose()


def summarize(model: str, answers: dict) -> dict:
    def stats(qset: str) -> dict:
        rows = [a for a in answers.values() if a.set == qset and not a.error]
        if not rows:
            return {"n": 0}
        latencies = [a.latency_ms for a in rows]
        tokens_per_s = [
            a.completion_tokens / (a.latency_ms / 1000) for a in rows if a.latency_ms > 0
        ]
        out = {
            "n": len(rows),
            "truncated": sum(a.finish_reason == "length" for a in rows),
            "p50_ms": round(statistics.median(latencies)),
            "avg_tokens_per_s": round(statistics.mean(tokens_per_s), 1) if tokens_per_s else None,
        }
        if qset == "math":
            out["accuracy"] = sum(bool(a.correct) for a in rows) / len(rows)
        judged = [a for a in rows if a.verdict in ("pass", "fail")]
        if judged:
            out["judged"] = len(judged)
            out["pass_rate"] = sum(a.verdict == "pass" for a in judged) / len(judged)
        return out

    return {"model": model, "math": stats("math"), "knowledge": stats("knowledge")}


async def amain(args: argparse.Namespace) -> None:
    items = {i.id: i for i in load(WORKLOAD_FILE)}
    directory = out_dir(args.model)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "answers.jsonl"
    answers = await answer_all(args.model, question_sets(items, args.baseline_run), path)
    if args.judge:
        await judge_all(answers, items, path, args.tpm)
    summary = summarize(args.model, answers)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline-run", default="full-1")
    parser.add_argument("--judge", action="store_true", help="also judge knowledge answers")
    parser.add_argument("--tpm", type=int, default=7000)
    asyncio.run(amain(parser.parse_args()))


if __name__ == "__main__":
    main()
