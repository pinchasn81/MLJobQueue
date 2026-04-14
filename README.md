# Honeycomb — ML Training Job Queue

A priority-based task queue for ML training jobs. Submitting a job writes it to SQLite as PENDING; a separate `POST /assign` call triggers execution — workers pick up jobs from Redis queues and process them in background threads.

---

## Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) (package manager)

---

## Installation

```bash
# Clone and enter the project
cd honeycomb

# Install all dependencies (creates .venv automatically)
uv sync

# Install dev dependencies (pytest, httpx, assertpy)
uv sync --extra dev
```

---

## Running the Server

```bash
# Option 1 — via uvicorn directly (with hot-reload)
uv run uvicorn honeycomb.main:app --reload

# Option 2 — via the CLI entry point (no hot-reload)
uv run honeycomb
```

The server starts on **http://localhost:8000**. Interactive API docs are at **http://localhost:8000/docs**.

### Configuration

Edit `[tool.honeycomb]` in `pyproject.toml`:

| Key | Default | Description |
|-----|---------|-------------|
| `num_workers` | `4` | Worker thread pool size |
| `db_path` | `"honeycomb.db"` | SQLite file path |

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tasks` | Submit a job |
| `POST` | `/assign` | Assign pending tasks to idle workers and start execution |
| `GET` | `/tasks/{task_id}` | Query task status |
| `GET` | `/stats` | Queue performance metrics |

### Submit a job

```bash
curl -X POST http://localhost:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"task_id": "train_fraud_v2", "priority": 2, "max_retries": 3}'
```

Priority values: `2` = HIGH, `1` = NORMAL, `0` = LOW.

### Trigger assignment

```bash
curl -X POST http://localhost:8000/assign
```

Returns the list of `(task_id, worker_id)` pairs assigned in this call. Call this whenever you want to drain the pending queue — typically on a schedule or after batch submission.

### Query a task

```bash
curl http://localhost:8000/tasks/train_fraud_v2
```

### View stats

```bash
curl http://localhost:8000/stats
```

---

## Running Tests

```bash
# Run the full test suite
uv run pytest tests/ -v

# Run a single test by name
uv run pytest tests/test_queue.py::TestQueueAPI::test_all_priorities_complete -v

# Run with output (useful for debugging — workers log to stdout)
uv run pytest tests/ -v -s
```

The session fixture starts the FastAPI server once on a random port in a background thread. Each test resets the DB and Redis state automatically via the `clean_state` autouse fixture.

---

## Project Layout

```
src/honeycomb/
├── main.py           # FastAPI app, lifespan, route handlers
├── task_queue.py     # Core domain logic — TaskQueue class + enums
├── worker.py         # Background worker threads + RQ job function
├── models.py         # SQLAlchemy ORM: TaskModel, WorkerModel
├── database.py       # Engine (WAL mode), SessionLocal, init_db()
├── schemas.py        # Pydantic request/response models
└── logging_config.py # Logging setup

tests/
├── conftest.py       # Session server fixture + per-test clean_state
├── logics.py         # Shared test helpers (submit, poll, verify)
└── test_queue.py     # Integration test suite (9 tests)
```
