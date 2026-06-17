"""Seed the classroom logins: group8 .. group16, shared password.

Run standalone with `uv run python -m app.seed`, or rely on the app's startup
hook (see app/main.py), which calls ensure_seed_users() when seed_users is on.
"""

import asyncio

from sqlalchemy import select

from app.config import get_settings
from app.db import SessionLocal
from app.models import User
from app.security import hash_password

# group8, group9, ..., group16
GROUP_USERNAMES = [f"group{i}" for i in range(8, 17)]


async def ensure_seed_users() -> None:
    settings = get_settings()
    hashed = hash_password(settings.seed_password)
    async with SessionLocal() as session:
        for username in GROUP_USERNAMES:
            existing = (
                await session.execute(select(User).where(User.email == username))
            ).scalar_one_or_none()
            if existing is None:
                session.add(User(email=username, hashed_pw=hashed))
        await session.commit()


if __name__ == "__main__":
    asyncio.run(ensure_seed_users())
    print(f"Seeded users: {', '.join(GROUP_USERNAMES)}")
