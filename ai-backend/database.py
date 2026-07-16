"""Small async PostgreSQL layer used by the FastAPI application."""
from __future__ import annotations

import os
from typing import Any

import asyncpg

from config import env

_pool: asyncpg.Pool | None = None


def database_settings() -> dict[str, Any]:
    return {
        "host": env("POSTGRES_HOST", "127.0.0.1"),
        "port": int(env("POSTGRES_PORT", "5432")),
        "user": env("POSTGRES_USER", "postgres"),
        "password": env("POSTGRES_PASSWORD"),
        "database": env("POSTGRES_DATABASE", "logiccore_db"),
    }


async def connect_database() -> None:
    global _pool
    if _pool is not None:
        return
    _pool = await asyncpg.create_pool(
        **database_settings(),
        min_size=1,
        max_size=5,
        command_timeout=20,
        server_settings={"application_name": "ultimateai_backend"},
    )
    async with _pool.acquire() as connection:
        users_table = await connection.fetchval("SELECT to_regclass('public.users')")
    if users_table is None:
        await close_database()
        raise RuntimeError(
            "The PostgreSQL schema is missing. Run the LogicCore table creation script in logiccore_db."
        )


async def close_database() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database connection pool is not ready.")
    return _pool


async def database_is_healthy() -> bool:
    try:
        return await pool().fetchval("SELECT 1") == 1
    except (asyncpg.PostgresError, RuntimeError):
        return False
