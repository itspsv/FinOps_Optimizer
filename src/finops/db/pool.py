import psycopg
from pgvector.psycopg import register_vector_async
from psycopg_pool import AsyncConnectionPool


async def _configure(conn: psycopg.AsyncConnection) -> None:
    await register_vector_async(conn)
    await conn.commit()


async def open_pool(database_url: str, timeout_s: float = 5.0) -> AsyncConnectionPool:
    """Open a small pool; raises if Postgres can't be reached within the timeout."""
    pool = AsyncConnectionPool(
        database_url, min_size=1, max_size=5, open=False, configure=_configure
    )
    await pool.open(wait=True, timeout=timeout_s)
    return pool
