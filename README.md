# AI FinOps Optimizer

A local-first proxy that sits between applications and paid LLM APIs and cuts cost by:

1. **Semantic caching**: serving answers to repeated or similar questions from a local pgvector cache.
2. **Dynamic routing**: sending simple requests to a free local model (Ollama) and only hard ones to the premium API.

Every decision is recorded in a cost ledger and shown in a Streamlit dashboard.

> **Status:** Step 2 of 5. OpenAI-compatible proxy to the premium tier (Groq), with a
> two-tier exact + semantic cache on pgvector.

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
make install                 # creates .venv with Python 3.12 and installs deps
cp .env.example .env         # then put your Groq API key in .env
createdb finops              # once; needs `brew install pgvector`
make db-init                 # creates the pgvector extension and cache table
make dev                     # starts the proxy on http://localhost:8000
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
| `x-finops-route` | `cache` or `premium`; later also `local` |
| `x-finops-cache` | `hit_exact`, `hit_semantic`, `miss`, `skip`, `bypass` or `error` |
| `x-finops-cache-similarity` | Similarity to the closest cached prompt, also sent on misses |
| `x-finops-provider` | Upstream that served the request |
| `x-finops-latency-ms` | Time to answer (non-streaming requests) |

Request headers the proxy understands:

| Header | Effect |
|---|---|
| `x-finops-namespace` | Cache partition, e.g. one per tenant. Defaults to `default`. |
| `x-finops-cache-control: bypass` | Skip the cache for this request |

## How the cache decides

1. **Skip** requests that shouldn't be reused: tool calls, `n > 1`, high temperature, images.
2. **Exact tier**: SHA-256 of model + full payload. No embedding cost.
3. **Semantic tier**: embed the last user message (bge-small, local ONNX) and find the nearest
   entry with the same conversation history, params, model and *guard fingerprint*.
4. **Guard fingerprint**: numbers, negations and opposite words ("enable"/"disable",
   "buy"/"sell") must match exactly, so "Q3 2024 revenue" never hits "Q3 2025 revenue".
5. **Store** only complete answers (`finish_reason: stop`), after the response is sent.

The cache fails open: if Postgres or the embedder is down, requests still go to the API.
Streamed misses are not cached yet; streamed requests can still be served from the cache.

## Configuration

Non-secret defaults live in `config/settings.yaml`. Environment variables override them, using
the `FINOPS_` prefix and `__` for nesting (e.g. `FINOPS_PREMIUM__MODEL`). Secrets go in `.env`.

## Development

```bash
make test     # upstream is mocked; store tests use a finops_test database (skipped if absent)
make lint
```
