from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Env-driven config. Values come from the process env or ai/.env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database (SQLAlchemy async URL). SQLite for the MVP.
    database_url: str = "sqlite+aiosqlite:///./playground.db"

    # Auth.
    auth_secret: str = "dev-insecure-change-me"
    token_ttl_seconds: int = 60 * 60 * 12  # 12h

    # Per-run trained-model checkpoints live under {work_dir}/run_{id}/.
    work_dir: str = "./.runs"

    # Seed the group logins on startup so the app is usable out of the box.
    # Users group8..group16 share one password (classroom setup).
    seed_users: bool = True
    seed_password: str = "trainllmwithucf"


@lru_cache
def get_settings() -> Settings:
    return Settings()
