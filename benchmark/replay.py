"""Send the workload through a running proxy, one request at a time, and record what happened.

    make dev                          # in another terminal
    make bench-run                    # or: uv run python -m benchmark.replay --run-id my-run

Each run uses its own namespace (`bench-<run id>`), so it starts with an empty cache and its
ledger rows can be reported on their own. Re-running the same run id resumes where it stopped.
"""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import httpx

from benchmark.workload import WORKLOAD_FILE, WorkItem, load

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RETRY_STATUSES = {429, 500, 502, 503, 504}


@dataclass
class ReplayResult:
    id: str
    position: int
    status: int
    route: str | None
    cache: str | None
    similarity: float | None
    route_reason: str | None
    escalated: str | None
    request_id: str | None
    client_ms: float
    answer: str | None
    attempts: int
    daily_cap: bool = False


def run_dir(run_id: str) -> Path:
    return RESULTS_DIR / run_id


def load_results(path: Path) -> dict[str, ReplayResult]:
    if not path.exists():
        return {}
    results = [ReplayResult(**json.loads(line)) for line in path.read_text().splitlines() if line]
    return {r.id: r for r in results}


def is_daily_cap(response: httpx.Response) -> bool:
    return response.status_code == 429 and "per day" in response.text


class Pacer:
    """Keeps request starts at least 60/rpm seconds apart (Groq free-tier friendly)."""

    def __init__(self, max_rpm: float, clock=time.monotonic, sleep=time.sleep):
        self._interval = 60.0 / max_rpm if max_rpm > 0 else 0.0
        self._clock, self._sleep = clock, sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None and (delay := self._last + self._interval - now) > 0:
            self._sleep(delay)
            now = self._clock()
        self._last = now


def send(
    client: httpx.Client,
    item: WorkItem,
    position: int,
    namespace: str,
    pacer: Pacer,
    max_attempts: int = 4,
    sleep=time.sleep,
) -> ReplayResult:
    body = {
        "model": "gpt-4o",  # what a typical app would request; the proxy decides
        "temperature": 0,
        "messages": [{"role": "user", "content": item.prompt}],
    }
    for attempt in range(1, max_attempts + 1):
        pacer.wait()
        started = time.perf_counter()
        try:
            response = client.post(
                "/v1/chat/completions", json=body, headers={"x-finops-namespace": namespace}
            )
        except httpx.HTTPError as exc:  # proxy restarted or timed out: record and move on
            response = httpx.Response(599, json={"error": {"message": str(exc)}})
        client_ms = (time.perf_counter() - started) * 1000
        if response.status_code not in RETRY_STATUSES or attempt == max_attempts:
            break
        if is_daily_cap(response):  # retrying can't help until the 24-hour window frees up
            break
        retry_after = float(response.headers.get("retry-after") or 10 * attempt)
        print(f"    {item.id}: HTTP {response.status_code}, retrying in {retry_after:.0f}s")
        sleep(min(retry_after, 60))

    h = response.headers
    answer = None
    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"].get("content")
    similarity = h.get("x-finops-cache-similarity")
    return ReplayResult(
        id=item.id,
        position=position,
        status=response.status_code,
        route=h.get("x-finops-route"),
        cache=h.get("x-finops-cache"),
        similarity=float(similarity) if similarity else None,
        route_reason=h.get("x-finops-route-reason"),
        escalated=h.get("x-finops-escalated"),
        request_id=h.get("x-finops-request-id"),
        client_ms=round(client_ms, 1),
        answer=answer,
        attempts=attempt,
        daily_cap=is_daily_cap(response),
    )


def replay(
    items: list[WorkItem],
    run_id: str,
    client: httpx.Client,
    pacer: Pacer,
    limit: int | None = None,
) -> Path:
    out_dir = run_dir(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_file = out_dir / "responses.jsonl"
    done = {k for k, r in load_results(results_file).items() if r.status == 200}
    namespace = f"bench-{run_id}"
    todo = [(i, item) for i, item in enumerate(items) if item.id not in done][:limit]
    print(f"Run {run_id}: {len(done)} done, {len(todo)} to send (namespace {namespace})")

    started = time.monotonic()
    with results_file.open("a") as f:
        for n, (position, item) in enumerate(todo, 1):
            result = send(client, item, position, namespace, pacer)
            f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
            f.flush()
            if result.daily_cap:
                print(
                    f"  Stopping at {n}/{len(todo)}: Groq's daily token cap is reached. "
                    "Re-run the same command later to resume."
                )
                break
            if n % 10 == 0 or n == len(todo):
                elapsed = time.monotonic() - started
                print(
                    f"  {n}/{len(todo)}  last: {item.id} -> {result.route} "
                    f"({result.client_ms:.0f} ms)  elapsed {elapsed / 60:.1f} min"
                )
    return results_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M"))
    parser.add_argument("--workload", type=Path, default=WORKLOAD_FILE)
    # Groq's free tier allows 8,000 tokens/minute and premium answers average ~1,200 tokens.
    parser.add_argument("--max-rpm", type=float, default=5, help="pace requests for Groq")
    parser.add_argument("--limit", type=int, help="send at most this many (for a smoke run)")
    args = parser.parse_args()

    with httpx.Client(base_url=args.base_url, timeout=180) as client:
        health = client.get("/health").json()
        print(f"Proxy health: {health}")
        missing = [k for k in ("cache", "ledger", "routing") if health.get(k) != "enabled"]
        if missing:
            raise SystemExit(f"Proxy has {missing} disabled; the benchmark needs all three.")
        items = load(args.workload)
        path = replay(items, args.run_id, client, Pacer(args.max_rpm), args.limit)
    print(f"\nSaved {path}\nNext: make bench-eval RUN={args.run_id}")


if __name__ == "__main__":
    main()
