-- Idempotent: safe to run on every setup.
-- {embedding_dim} is filled in from config by finops.db.schema.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS cache_entries (
    id                BIGSERIAL PRIMARY KEY,
    namespace         TEXT        NOT NULL,  -- tenant isolation
    request_hash      TEXT        NOT NULL,  -- exact-match key (model + full payload)
    context_hash      TEXT        NOT NULL,  -- model + params + earlier messages
    prompt_text       TEXT        NOT NULL,  -- last user message, the part we embed
    guard_fingerprint TEXT        NOT NULL,  -- numbers / negation / polarity words
    embedding         vector({embedding_dim}) NOT NULL,
    embedding_model   TEXT        NOT NULL,
    upstream_model    TEXT        NOT NULL,
    response          JSONB       NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at        TIMESTAMPTZ NOT NULL,
    hit_count         INTEGER     NOT NULL DEFAULT 0,
    last_hit_at       TIMESTAMPTZ,
    UNIQUE (namespace, request_hash)
);

CREATE INDEX IF NOT EXISTS cache_entries_embedding_hnsw
    ON cache_entries USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS cache_entries_lookup
    ON cache_entries (namespace, context_hash, embedding_model, guard_fingerprint);

-- One row per proxied request: the FinOps ledger behind every reported metric.
CREATE TABLE IF NOT EXISTS request_log (
    id                 BIGSERIAL PRIMARY KEY,
    request_id         TEXT        NOT NULL UNIQUE,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    namespace          TEXT        NOT NULL,
    requested_model    TEXT,                  -- what the client asked for
    stream             BOOLEAN     NOT NULL,
    status_code        INTEGER     NOT NULL,
    route              TEXT        NOT NULL,  -- cache | local | premium
    cache_status       TEXT,
    cache_similarity   REAL,
    decision_tier      TEXT,                  -- router's choice before escalation
    decision_reason    TEXT,
    complexity_score   INTEGER,
    escalation_reason  TEXT,                  -- set when local failed and premium answered
    provider           TEXT,
    upstream_model     TEXT,                  -- model that produced the answer
    prompt_tokens      INTEGER     NOT NULL DEFAULT 0,
    completion_tokens  INTEGER     NOT NULL DEFAULT 0,
    tokens_estimated   BOOLEAN     NOT NULL DEFAULT false,
    latency_ms         REAL        NOT NULL,
    local_latency_ms   REAL,
    premium_latency_ms REAL,
    pricing_version    TEXT        NOT NULL,
    baseline_model     TEXT        NOT NULL,
    baseline_cost_usd  NUMERIC(14, 8) NOT NULL,  -- if sent to the baseline model
    cost_usd           NUMERIC(14, 8) NOT NULL,  -- what this route costs at list price
    saved_usd          NUMERIC(14, 8) NOT NULL,
    billed_usd         NUMERIC(14, 8)            -- what we actually paid (Groq), if known
);

CREATE INDEX IF NOT EXISTS request_log_created ON request_log (created_at);
CREATE INDEX IF NOT EXISTS request_log_namespace ON request_log (namespace, created_at);
