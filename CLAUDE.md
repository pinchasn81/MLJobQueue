# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

An MLOps job interview assignment: a FastAPI server implementing a priority-based task queue for ML training jobs. Handles job lifecycle, priority scheduling, exponential-backoff retry, worker assignment, and persistence via SQLite.

## Commands

```bash
# Install dependencies
uv sync

# Start the server (port 8000)
uv run uvicorn honeycomb.main:app --reload

# Or via the CLI script
uv run honeycomb

# Run tests (when added)
uv run pytest tests/ -v

# Install dev dependencies
uv sync --extra dev
```

## Project Structure

```
src/honeycomb/
├── models.py         # SQLAlchemy ORM: TaskModel, WorkerModel tables
├── database.py       # Engine (WAL mode), SessionLocal, init_db(), get_db()
├── task_queue.py     # Core domain logic — TaskQueue class + enums
├── schemas.py        # Pydantic request/response models (separate from ORM)
├── logging_config.py # Stdlib logging setup
└── main.py           # FastAPI app, lifespan, route handlers
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/tasks` | Submit a job (body: `task_id`, `priority`, `max_retries`) |
| `POST` | `/tasks/assign` | Assign pending tasks to idle workers |
| `POST` | `/tasks/complete` | Mark a task done/failed (body: `task_id`, `success`) |
| `GET`  | `/tasks/{task_id}` | Query task status |
| `GET`  | `/stats` | Queue performance metrics |

FastAPI auto-generates OpenAPI docs at `/docs`.

## Configuration

Stored in `pyproject.toml` under `[tool.honeycomb]`:
- `num_workers` — size of the compute worker pool (default: 4)
- `db_path` — SQLite file path (default: `"honeycomb.db"`); use `":memory:"` in tests

Read at startup via Python's `tomllib` (stdlib, Python 3.11+).

## Key Architecture Decisions

**TaskQueue + Session injection**: `TaskQueue.__init__` accepts an optional `session` parameter. In the FastAPI layer, a per-request session is injected via `Depends(get_task_queue)`. TaskQueue only flushes (never commits) — the session lifecycle is owned by `get_db()`.

**SQLite WAL mode**: Allows concurrent readers during writes. Default journal mode locks the whole file per write, which breaks under concurrent FastAPI requests.

**Composite index `ix_tasks_assignment`** on `(status, priority, submission_order)`: Covers the hot `assign_tasks()` query (`WHERE status IN (...) ORDER BY priority DESC, submission_order ASC`). Without it, every assignment is a full table scan.

**`submission_order` counter for FIFO**: An integer monotonic counter instead of `created_at` timestamp — timestamps collide within the same millisecond under load.

**`backoff` for DB resilience only**: The `backoff` library wraps session operations to retry on `OperationalError` (SQLite lock contention). Task retry is stateful domain logic (counter + re-enqueue) — not function-call retry — and is implemented directly in `complete_task()`.

**Sync endpoints**: `def`, not `async def`. SQLite is local I/O; async here would block the event loop. FastAPI runs sync endpoints in a threadpool automatically.

**No Alembic**: Tables are created via `Base.metadata.create_all()` at startup. Workers are seeded once. Add Alembic when the first schema migration is needed.
