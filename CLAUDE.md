# LLM Fine-Tuning Playground

A monorepo for a web app that lets users launch LoRA fine-tuning jobs and watch
live training metrics. Two independently runnable, independently deployable apps:

```
ui/   # Next.js (App Router, TypeScript) — deploys to a web host
ai/   # FastAPI (Python, uv)            — API + in-process trainer
```

There is **no shared build tool**. Each folder has its own deps, lint, and deploy.

---

## Architecture rules

> **Note — deviation from the original brief:** the original spec made `ai/` a
> Slurm *control plane* that shells out to `sbatch` and never trains in-process.
> By explicit decision, this app runs training **in-process** instead. There is
> no Slurm, no `sbatch`, no separate trainer process. The rules below reflect the
> in-process design.

### `ai/` is the API **and** the trainer (in-process)
- A single FastAPI app accepts requests, authenticates them, writes to Postgres,
  and runs training **inside the server process** as a background task.
- `submit` must **not block the request**: it persists the run, launches the
  training task in the background, and returns immediately. Status and metrics
  are read back from Postgres by polling, never by holding a request open.
- The dummy MVP trainer is an async coroutine (just `await asyncio.sleep`), so
  `asyncio.create_task` is fine. The **real** TRL/PEFT trainer is CPU/GPU-bound
  and must run via `loop.run_in_executor` (thread/process pool) so it does not
  block the event loop. Keep that boundary in `app/runner.py`.
- In-process tasks are **not durable** across restarts. Acceptable for the MVP;
  a restart leaves any `running` rows stale (reconcile/clean up later).

### The training code is a callable in `ai/training/`
- Lives in `ai/training/` and is invoked in-process by the runner. It loads its
  config from a **run id** + the DB, runs, and writes progress rows back to
  Postgres (`run_metrics`) plus terminal status back to `runs`.

### The database is the source of truth
- Users, runs, and metrics all live in the DB (**SQLite** for the MVP, via
  `aiosqlite`; schema created with `Base.metadata.create_all`). No in-memory job
  registry that survives a restart is relied upon for correctness.

### Auth gates every job-affecting endpoint
- Every endpoint that can **submit**, **cancel**, or read another user's runs
  requires authentication. There is no unauthenticated path that starts training.

---

## UI <-> AI wiring

- `ui` is **stateless** — it has no database of its own. Auth state is the JWT
  cookie; all users/runs/metrics live in `ai`'s SQLite. The browser talks **only**
  to `ui`'s own Next.js route handlers. It never sees the `ai` URL.
- Those route handlers (server-side) call `ai` using a **server-only** env var
  `API_URL`. Do not prefix it with `NEXT_PUBLIC_`; do not leak it to the client.
- Types stay in sync via a **generated TS client** in `ui`, produced from `ai`'s
  OpenAPI schema (`ai` serves `/openapi.json`). Regenerate when the API changes;
  never hand-edit generated types.

---

## Data model (SQLite for the MVP, minimal to start)

```
users (id, email, hashed_pw, created_at)

runs (id, user_id, status, job_id, dataset, base_model,
      config_json, created_at, updated_at)
      status ∈ {queued, submitted, running, succeeded, failed, cancelled}
      job_id = in-process task handle (kept generic; was slurm_job_id in the brief)

run_metrics (id, run_id, step, loss, extra_json, ts)
```

`run_metrics` is append-only; the loss curve is built from these rows.

---

## Defaults (flag deviations)

- **Fine-tuning:** HuggingFace + TRL/PEFT, LoRA on small base models
  (e.g. `Qwen/Qwen2.5-0.5B`, `HuggingFaceTB/SmolLM2-360M`, `gpt2`).
- **Datasets:** a fixed preset list for the MVP. No uploads yet.
- **Auth:** seeded email+password users (no public signup for the MVP). `ai`
  issues a JWT bearer on login; `ui` keeps it in an httpOnly cookie and forwards
  it server-side. Can graduate to Better Auth later.
- The first vertical slice uses a **dummy ~30-second "training" task** that writes
  a few fake loss rows, so the full submit→train→metrics→UI loop works without GPUs.

---

## Build order (one vertical slice end-to-end first)

1. **ai:** `/health`, Postgres connection + migrations, runs CRUD, and a `submit`
   endpoint that writes a run row and launches the in-process dummy trainer.
2. **ui:** auth, a "new run" form (dataset + base_model + a couple hyperparams),
   a runs list, and a run detail page that polls status and plots the loss curve
   from `run_metrics`.
3. **Verify the full loop:** submit → train task runs → metrics rows written → ui
   shows live status + curve → completion.

**Then** expand: swap the dummy trainer for the real TRL LoRA run (in an executor),
add cancel, add checkpoint/adapter download.

---

## Conventions

- `ai`: **uv** for deps, **ruff** for lint/format, **pydantic-settings** for
  env-driven config. Single Python app — server and trainer share one env.
- `ui`: **pnpm**. Note this is **Next.js 16** (App Router) — APIs may differ from
  older versions; consult `ui/node_modules/next/dist/docs/` before writing.
- Keep the in-process job runner in `app/runner.py`, isolated from request
  handlers; keep training logic in `ai/training/`.
- No external services for the MVP: `ai` uses a local **SQLite** file
  (`ai/playground.db`); `ui` has no DB. Both folders ship a `.env.example` and a
  real `.env` for local dev (the latter git-ignored).
- Each folder is independently runnable and deployable.
