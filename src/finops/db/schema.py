from pathlib import Path

import psycopg

SCHEMA_FILE = Path(__file__).with_name("schema.sql")


async def apply_schema(conn: psycopg.AsyncConnection, embedding_dim: int) -> None:
    sql = SCHEMA_FILE.read_text().replace("{embedding_dim}", str(int(embedding_dim)))
    await conn.execute(sql)
    await conn.commit()
