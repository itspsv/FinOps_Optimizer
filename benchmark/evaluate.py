"""Measure answer quality for a replay run.

Three checks:
  1. GSM8K exact match (free): the final number in the answer vs the dataset's answer.
  2. Local vs premium: for a sample of locally answered requests, ask Groq for a reference
     answer, then an LLM judge decides whether the local answer is as good.
  3. Semantic cache hits: the judge decides whether the cached answer (written for an
     earlier, similar question) correctly answers the new question.

Judge and reference calls go straight to Groq, not through the proxy, so they don't
appear in the run's ledger. Re-running resumes where it stopped.
"""

import argparse
import asyncio
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmark.replay import Pacer, ReplayResult, load_results, run_dir
from benchmark.workload import WORKLOAD_FILE, WorkItem, load
from finops.core.config import get_settings
from finops.providers import OpenAICompatibleProvider, Provider, ProviderError

MAX_CHARS = 4000  # per answer in judge prompts, to bound tokens

JUDGE_SYSTEM = (
    "You are a strict, fair evaluator of AI assistant answers. "
    'Reply only with JSON: {"verdict": "pass" or "fail", "reason": "<one sentence>"}.'
)

LOCAL_TEMPLATE = """QUESTION:
{question}

REFERENCE ANSWER (from a stronger model):
{reference}

CANDIDATE ANSWER:
{candidate}

Is the CANDIDATE an acceptable answer to the QUESTION? Grade it on its own merits:
- "fail" if it contains a factual error, gives a wrong answer, ignores the instructions,
  or leaves out something the question explicitly asked for (e.g. "name all ...").
- Otherwise "pass". Being shorter or less detailed than the reference is NOT a reason to
  fail: the candidate was deliberately told to be concise. Use the reference to check facts."""

SEMANTIC_TEMPLATE = """QUESTION:
{question}

ANSWER (originally written for a different but similar question):
{candidate}

Does this ANSWER correctly and directly answer this exact QUESTION? If it answers a
different question, even a closely related one, the verdict is "fail"."""


@dataclass
class Judgement:
    id: str
    check: str  # local_vs_reference | semantic_hit
    verdict: str  # pass | fail | error
    reason: str
    reference: str | None = None


def final_number(text: str | None) -> str | None:
    """Last number in an answer, normalised ("1,234.50" -> "1234.5")."""
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    if not numbers:
        return None
    value = numbers[-1].replace(",", "").rstrip(".")
    return value.rstrip("0").rstrip(".") if "." in value else value


def gsm8k_accuracy(items: dict[str, WorkItem], results: dict[str, ReplayResult]) -> dict:
    by_route: dict[str, list[bool]] = {}
    for result in results.values():
        item = items[result.id]
        if item.source != "gsm8k" or result.status != 200:
            continue
        correct = final_number(result.answer) == final_number(item.reference)
        by_route.setdefault(result.route or "unknown", []).append(correct)
    return {
        route: {"n": len(v), "correct": sum(v), "accuracy": sum(v) / len(v)}
        for route, v in by_route.items()
    }


def parse_verdict(text: str) -> tuple[str, str]:
    try:
        data = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
        verdict = str(data.get("verdict", "")).lower()
        if verdict in ("pass", "fail"):
            return verdict, str(data.get("reason", ""))
    except (AttributeError, ValueError):
        pass
    lowered = text.lower()
    if '"fail"' in lowered or "verdict: fail" in lowered:
        return "fail", text[:200]
    if '"pass"' in lowered or "verdict: pass" in lowered:
        return "pass", text[:200]
    return "error", f"unparseable judge output: {text[:200]}"


class GroqClient:
    """Direct calls to the premium provider with pacing and 429 retries."""

    def __init__(self, provider: Provider, pacer: Pacer):
        self._provider = provider
        self._pacer = pacer

    async def ask(self, messages: list[dict], json_mode: bool = False) -> str:
        payload: dict = {"messages": messages, "temperature": 0}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        for attempt in range(1, 5):
            self._pacer.wait()
            try:
                response = await self._provider.chat(payload)
                return response["choices"][0]["message"].get("content") or ""
            except ProviderError as exc:
                if exc.status_code == 400 and json_mode:  # model without JSON mode
                    payload.pop("response_format", None)
                    continue
                if exc.status_code not in (429, 500, 502, 503, 504) or attempt == 4:
                    raise
                print(f"    Groq HTTP {exc.status_code}, retrying in {15 * attempt}s")
                time.sleep(15 * attempt)
        raise RuntimeError("unreachable")

    async def judge(self, prompt: str) -> tuple[str, str]:
        text = await self.ask(
            [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            json_mode=True,
        )
        return parse_verdict(text)


async def evaluate(
    items: dict[str, WorkItem],
    results: dict[str, ReplayResult],
    groq: GroqClient,
    out_file: Path,
    local_sample: int,
    seed: int = 7,
) -> list[Judgement]:
    done = {}
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            j = Judgement(**json.loads(line))
            if j.verdict != "error":
                done[(j.id, j.check)] = j

    local = sorted(r.id for r in results.values() if r.status == 200 and r.route == "local")
    local = random.Random(seed).sample(local, min(local_sample, len(local)))
    semantic = sorted(r.id for r in results.values() if r.cache == "hit_semantic")
    tasks = [(i, "local_vs_reference") for i in local] + [(i, "semantic_hit") for i in semantic]
    todo = [t for t in tasks if t not in done]
    print(f"Judging {len(todo)} answers ({len(tasks) - len(todo)} already done)")

    with out_file.open("a") as f:
        for n, (item_id, check) in enumerate(todo, 1):
            item, result = items[item_id], results[item_id]
            candidate = (result.answer or "")[:MAX_CHARS]
            reference = None
            try:
                if check == "local_vs_reference":
                    reference = await groq.ask([{"role": "user", "content": item.prompt}])
                    prompt = LOCAL_TEMPLATE.format(
                        question=item.prompt[:MAX_CHARS],
                        reference=reference[:MAX_CHARS],
                        candidate=candidate,
                    )
                else:
                    prompt = SEMANTIC_TEMPLATE.format(
                        question=item.prompt[:MAX_CHARS], candidate=candidate
                    )
                verdict, reason = await groq.judge(prompt)
            except ProviderError as exc:
                verdict, reason = "error", f"Groq error {exc.status_code}"
            judgement = Judgement(item_id, check, verdict, reason, reference)
            done[(item_id, check)] = judgement
            f.write(json.dumps(asdict(judgement), ensure_ascii=False) + "\n")
            f.flush()
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} judged")
    return [done[t] for t in tasks if t in done]


def summarize(judgements: list[Judgement], items: dict[str, WorkItem]) -> dict:
    def rate(check: str, kinds: set[str] | None = None) -> dict:
        rows = [
            j for j in judgements
            if j.check == check and j.verdict != "error"
            and (kinds is None or items[j.id].kind in kinds)
        ]  # fmt: skip
        passed = sum(j.verdict == "pass" for j in rows)
        return {"n": len(rows), "passed": passed, "pass_rate": passed / len(rows) if rows else None}

    return {
        "local_quality": rate("local_vs_reference"),
        "semantic_hit_precision": rate("semantic_hit"),
        "semantic_hits_on_paraphrases": rate("semantic_hit", {"paraphrase"}),
        "semantic_hits_on_lookalikes": rate("semantic_hit", {"near_miss"}),
        "errors": sum(j.verdict == "error" for j in judgements),
    }


async def amain(run_id: str, local_sample: int, judge_model: str | None, max_rpm: float) -> None:
    out = run_dir(run_id)
    results = load_results(out / "responses.jsonl")
    if not results:
        raise SystemExit(f"No responses for run {run_id}; run `make bench-run` first.")
    items = {i.id: i for i in load(WORKLOAD_FILE)}

    settings = get_settings().premium
    if judge_model:
        settings = settings.model_copy(update={"model": judge_model})
    provider = OpenAICompatibleProvider(settings)
    try:
        judgements = await evaluate(
            items, results, GroqClient(provider, Pacer(max_rpm)), out / "eval.jsonl", local_sample
        )
    finally:
        await provider.aclose()

    summary = {
        "judge_model": settings.model,
        "gsm8k": gsm8k_accuracy(items, results),
        **summarize(judgements, items),
    }
    (out / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nNext: make bench-report RUN={run_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--local-sample", type=int, default=60)
    parser.add_argument("--judge-model", help="defaults to the premium model")
    parser.add_argument("--max-rpm", type=float, default=25)
    args = parser.parse_args()
    asyncio.run(amain(args.run_id, args.local_sample, args.judge_model, args.max_rpm))


if __name__ == "__main__":
    main()
