import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db import engine
from app.models import Base
from app.routers import auth, catalog, health, runs
from app.seed import ensure_seed_users

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Create tables (SQLite, no migrations for the MVP) and seed the group logins.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    if get_settings().seed_users:
        await ensure_seed_users()
    yield


app = FastAPI(
    title="LLM Fine-Tuning Playground — Training Server",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the browser frontend (different origin / tunnel) to call ai directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(catalog.router)
app.include_router(runs.router)
