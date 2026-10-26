# AI FinOps Optimizer

A local-first proxy that sits between applications and paid LLM APIs and cuts cost by:

1. **Semantic caching**: serving answers to repeated or similar questions from a local pgvector cache.
2. **Dynamic routing**: sending simple requests to a free local model (Ollama) and only hard ones to the premium API.

Every decision is recorded in a cost ledger and shown in a Streamlit dashboard.

> **Status:** Step 5 of 6. Proxy, semantic cache, LangGraph router, priced ledger, and a
> reproducible benchmark on public datasets. Next: Streamlit dashboard.

## Architecture (target)

```
App ──▶ FastAPI proxy ──▶ 1. Cache hit?  ──yes──▶ cached answer
                              │ no
                              ▼
                          2. Simple?     ──yes──▶ Ollama (local) ──fails check──┐
                              │ no                                             │
                              ▼                                                ▼
                          Premium API (Groq)  ◀────────────────────────────────┘
                              │
                              ▼
                          3. Cost ledger (Postgres) ──▶ Streamlit dashboard
```

## Setup (macOS, no Docker)

Requirements: [uv](https://docs.astral.sh/uv/), Homebrew PostgreSQL 18 with pgvector, Ollama.

```bash
# In its own terminal: one model in memory at a time, kept warm between requests
OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=10m ollama serve
ollama pull llama3.2:3b      # ~2 GB local model

make install                 # creates .venv with Python 3.12 and installs deps
cp .env.example .env         # then put your Groq API key in .env
createdb finops              # once; needs `brew install pgvector`
make db-init                 # creates the pgvector extension, cache and ledger tables
make dev                     # starts the proxy on http://localhost:8000
make report                  # savings summary from the ledger
```

## Using the proxy

The proxy speaks the OpenAI Chat Completions API, so existing clients only change `base_url`:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="gpt-4o",  # recorded as the requested model; the proxy picks the real one
    messages=[{"role": "user", "content": "Hello"}],
)
```

Every response carries headers that say what the proxy did:

| Header | Meaning |
|---|---|
| `x-finops-request-id` | Unique ID for tracing this request |
| `x-finops-route` | `cache`, `local` or `premium`: what actually answered |
| `x-finops-route-reason` | Why the router chose that tier, e.g. `simple_task`, `code`, `tools` |
| `x-finops-complexity` | The router's complexity score |
| `x-finops-escalated` | Present when a local attempt failed, e.g. `local_truncated` |
| `x-finops-cache` | `hit_exact`, `hit_semantic`, `miss`, `skip`, `bypass` or `error` |
| `x-finops-cache-similarity` | Similarity to the closest cached prompt, also sent on misses |
| `x-finops-provider` | Upstream that served the request |
| `x-finops-latency-ms` | Time to answer (non-streaming requests) |

Request headers the proxy understands:

| Header | Effect |
|---|---|
| `x-finops-namespace` | Cache partition, e.g. one per tenant. Defaults to `default`. |
| `x-finops-cache-control: bypass` | Skip the cache for this request |
| `x-finops-route-override` | Force `local` or `premium`, e.g. for benchmarking |

## How the cache decides

1. **Skip** requests that shouldn't be reused: tool calls, `n > 1`, high temperature, images.
2. **Exact tier**: SHA-256 of model + full payload. No embedding cost.
3. **Semantic tier**: embed the last user message (bge-small, local ONNX) and find the nearest
   entry with the same conversation history, params, model and *guard fingerprint*.
4. **Guard fingerprint**: numbers, negations and opposite words ("enable"/"disable",
   "buy"/"sell") must match exactly, so "Q3 2024 revenue" never hits "Q3 2025 revenue".
5. **Store** only complete answers (`finish_reason: stop`), after the response is sent.

The cache fails open: if Postgres or the embedder is down, requests still go to the API.
Streamed premium answers are not cached yet; streamed requests can still be served from the cache.

## How the router decides

Requests the cache can't answer go through a LangGraph workflow
([engine.py](src/finops/router/engine.py)):

```
classify ─┬─ local ──▶ Ollama ──▶ answer ok? ── yes ──▶ done
          │                          │ no / Ollama down
          │                          ▼
          └─ premium ──────────▶ Groq ──▶ done
```

1. **Hard limits → premium**: tools, images, JSON schemas, prompts over ~1,500 tokens or
   requested answers over 512 tokens. The 3B model on an 8 GB M1 writes ~22 tokens/s, so long
   jobs would be slow locally.
2. **Complexity score**: keyword signals for code (+2), math (+2) and reasoning (+1), plus
   long prompts and long conversations. Simple tasks (classify, translate, summarize) get −1.
   A score of 2 or more goes to premium. Rules, not an LLM, so routing costs nothing.
3. **Answer check**: a local answer that is truncated, empty, a refusal or invalid JSON is
   discarded and the request **escalates** to premium with the original payload.
4. **Circuit breaker**: after 3 local failures in a row, local is skipped for 30 s, so a
   stopped Ollama doesn't add a failed call to every request.

### Routing policies

| Policy | Local gets | Premium gets |
|---|---|---|
| `permissive` (v1) | anything without complexity signals | code, math expressions, heavy reasoning |
| `grounded` (v2, default) | tasks done from the given text: classify, summarize, extract, rewrite, questions with a supplied passage | also open knowledge questions and math word problems |

Why v2: benchmark run `full-1` (v1) kept 89% of requests off premium and saved 60%, but an
LLM judge rated only 54% of local answers acceptable. The 3B model handled extraction and
summaries well and made up facts on open questions. It also sent 25 of 30 GSM8K word problems
local, because they contain no math keywords. Set `routing.local_policy` to choose; a
stronger local model (e.g. 8B on a 16 GB machine) makes `permissive` more viable.

Local answers are cached like premium ones. Streamed requests that go local are answered in
full, checked, then replayed as a stream.

## Cost ledger

Every request writes one row to the `request_log` table after the response is sent: route,
cache result, router decision, escalation, tokens, latency and cost. Prices live in
[config/pricing.yaml](config/pricing.yaml) with a version that is stored on every row.

| Column | Meaning |
|---|---|
| `baseline_cost_usd` | What the request would cost with no optimizer: all traffic on the baseline model (GPT-6.1 Sol list price) |
| `cost_usd` | What its route costs: premium priced as the baseline model, local and cache $0 |
| `saved_usd` | `baseline_cost_usd - cost_usd` |
| `billed_usd` | What was actually paid (Groq's real price), kept separate for honesty |
| `tokens_estimated` | True when a streamed answer reported no usage and tokens were estimated |

Premium calls go to Groq during development but are **priced as a commercial API**, so the
savings model a company paying list prices. Token counts are those of the answer actually
served; for local answers that is the local model's own count.

```bash
make report                              # all traffic
make report ARGS="--namespace demo"      # one tenant
make report ARGS="--since-minutes 60 --json"
curl localhost:8000/finops/summary       # same numbers over HTTP
```

| Metric | Definition |
|---|---|
| Deflection rate | Share of served requests answered by cache or local |
| Token deflection rate | Share of tokens answered by cache or local |
| Savings | `sum(saved_usd) / sum(baseline_cost_usd)` |
| Latency | p50 / p95 per route, from the ledger |

## Benchmark

A reproducible replay of public prompts measures everything above
([benchmark/](benchmark/)). Data is fetched from the Hugging Face rows API and never committed.

| Source | Used for |
|---|---|
| Databricks Dolly 15k | everyday tasks across 8 categories (mostly simple) |
| GSM8K | math word problems (complex; answers checked exactly) |
| HumanEval | coding problems (complex) |
| Quora Question Pairs | human paraphrases (should hit the cache) and look-alikes (must not) |

```bash
make bench-data                 # download samples (~3 MB), build a 330-request workload
make bench-calibrate            # pick the cache threshold on 2,000 labeled pairs (offline)
make dev                        # in another terminal
make bench-run RUN=full-1       # replay (~30 min on an M1; resumable)
make bench-eval RUN=full-1      # LLM judge + GSM8K exact match
make bench-report RUN=full-1    # -> benchmark/results/full-1/report.md
make bench-compare RUNS="full-1 full-2"   # side-by-side table
```

Groq's free tier caps gpt-oss-120b at **200K tokens/day** (and 8K/minute). One full replay
plus judging can exceed it. The replay and evaluation both resume where they stopped, and the
evaluator reuses reference answers from earlier runs. To split a replay over two days, keep
cached answers alive between sessions with `FINOPS_CACHE__TTL_S=259200 make dev`.

What the benchmark has already changed:

- **Cache threshold 0.92 → 0.96.** On Quora pairs, hit precision was 0.80 at 0.92 and 0.89
  at 0.96 (the ceiling given label noise).
- **A brevity hint for the local model.** Open questions took 15–30 s locally and often hit
  the token cap, then escalated anyway. With the hint, a 20-request sample took 2.2 min
  instead of 4.9, with no escalations.

## Configuration

Non-secret defaults live in `config/settings.yaml`. Environment variables override them, using
the `FINOPS_` prefix and `__` for nesting (e.g. `FINOPS_PREMIUM__MODEL`). Secrets go in `.env`.

## Development

```bash
make test     # upstream is mocked; store tests use a finops_test database (skipped if absent)
make lint
```
