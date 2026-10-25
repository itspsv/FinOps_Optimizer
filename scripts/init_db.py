"""Create the pgvector extension and tables in the configured database."""

import asyncio

import psycopg

from finops.core.config import get_settings
from finops.db.schema import apply_schema


async def main() -> None:
    settings = get_settings()
    async with await psycopg.AsyncConnection.connect(settings.database_url) as conn:
        await apply_schema(conn, settings.cache.embedding_dim)
    print(f"Schema applied to {settings.database_url}")


if __name__ == "__main__":
    asyncio.run(main())
