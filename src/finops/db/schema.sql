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
