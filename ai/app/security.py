import time

import bcrypt
import jwt

from app.config import get_settings

_ALGO = "HS256"


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        return False


def create_token(user_id: int, email: str) -> str:
    settings = get_settings()
    now = int(time.time())
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": now,
        "exp": now + settings.token_ttl_seconds,
    }
    return jwt.encode(payload, settings.auth_secret, algorithm=_ALGO)


def decode_token(token: str) -> dict:
    settings = get_settings()
    return jwt.decode(token, settings.auth_secret, algorithms=[_ALGO])
