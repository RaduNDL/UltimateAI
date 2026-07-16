"""Password hashing and short-lived access tokens for local accounts."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import UUID

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from config import env

PASSWORD_HASHER = PasswordHasher()
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_MINUTES = int(env("ACCESS_TOKEN_MINUTES", "720"))


def hash_password(password: str) -> str:
    return PASSWORD_HASHER.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return PASSWORD_HASHER.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


def create_access_token(user_id: UUID) -> tuple[str, datetime]:
    expires_at = datetime.now(UTC) + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    token = jwt.encode(
        {"sub": str(user_id), "exp": expires_at, "type": "access"},
        env("JWT_SECRET"),
        algorithm=JWT_ALGORITHM,
    )
    return token, expires_at


def read_access_token(token: str) -> UUID | None:
    try:
        payload = jwt.decode(token, env("JWT_SECRET"), algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            return None
        return UUID(str(payload["sub"]))
    except (jwt.PyJWTError, KeyError, TypeError, ValueError):
        return None
