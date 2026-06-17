# LLM Fine-Tuning Playground

A small web app to launch tiny LLM training runs and watch the loss curve live.
Monorepo with two independently runnable apps:

```
ai/   # FastAPI training server (Python, uv) — trains a tiny char-level model in-process
ui/   # Next.js frontend (App Router, TypeScript) — auth, new-run form, live loss curve
```

See [`CLAUDE.md`](./CLAUDE.md) for the architecture rules.

---

## Quick start — `ai/` (training server)

Prereqs: [`uv`](https://docs.astral.sh/uv/) and Python 3.13. No database server
needed — the app uses a local SQLite file (`ai/playground.db`).

```bash
cd ai

# 1. Install dependencies (creates .venv from pyproject + uv.lock)
uv sync

# 2. Create your local env file from the template
cp .env.example .env          # defaults work as-is for local dev

# 3. Run the server (auto-creates tables + seeds the group logins on startup)
uv run uvicorn app.main:app --reload --port 8000
```

Then open:

- API docs (Swagger UI): http://localhost:8000/docs
- Health check: http://localhost:8000/health

### Logins

On startup the app seeds 9 users — **`group8` … `group16`** — all with password
**`trainllmwithucf`**. (Re-seed manually any time with `uv run python -m app.seed`.)

### Try the full loop from the terminal

```bash
# Log in -> get a bearer token
TOKEN=$(curl -s localhost:8000/auth/login \
  -H 'content-type: application/json' \
  -d '{"email":"group8","password":"trainllmwithucf"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')

# See available datasets + model sizes + hyperparam defaults
curl -s localhost:8000/catalog -H "authorization: Bearer $TOKEN"

# Create a run (queued)
RUN=$(curl -s localhost:8000/runs -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' \
  -d '{"dataset":"PoemBot","base_model":"tiny-gru","config":{"training_steps":300}}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')

# Submit it -> training starts in a background thread
curl -s -X POST localhost:8000/runs/$RUN/submit -H "authorization: Bearer $TOKEN"

# Poll status + loss metrics while it trains
curl -s localhost:8000/runs/$RUN -H "authorization: Bearer $TOKEN"

# After it succeeds, generate text from this run's own trained model
curl -s -X POST localhost:8000/runs/$RUN/generate -H "authorization: Bearer $TOKEN" \
  -H 'content-type: application/json' -d '{"start_text":"The moon","length":200}'
```

### How training works

- `submit` launches the trainer in a **thread executor** (real PyTorch training is
  CPU-bound, so it never blocks the event loop). Cancel is cooperative via a flag.
- The trainer (a tiny char-level GRU, trained from scratch — no model downloads)
  writes a loss row every few steps to `run_metrics`, then saves a per-run
  checkpoint so each user can generate from **their own** trained model.
- All runs are scoped by user; every job-affecting endpoint requires the bearer.

### Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET  | `/health` | – | DB ping |
| POST | `/auth/login` | – | email/username + password → JWT |
| GET  | `/auth/me` | ✓ | current user |
| GET  | `/catalog` | ✓ | preset datasets, model sizes, hyperparam defaults |
| POST | `/runs` | ✓ | create a run (queued) |
| GET  | `/runs` | ✓ | list my runs |
| GET  | `/runs/{id}` | ✓ | run detail + metrics |
| GET  | `/runs/{id}/metrics` | ✓ | loss rows (for polling) |
| POST | `/runs/{id}/submit` | ✓ | start training |
| POST | `/runs/{id}/cancel` | ✓ | cancel a running job |
| POST | `/runs/{id}/generate` | ✓ | generate text from this run's model |

---

## `ui/` (frontend)

Slice 2 — not built yet. Will be a Next.js app that talks **only** to its own
route handlers, which proxy to `ai` via the server-only `API_URL` env var. See
`ui/.env.example`.
