# Architecture — Honeycomb ML Job Queue

## System Overview

Honeycomb is a priority-based task queue for ML training jobs. It accepts job submissions from a CI/CD pipeline, stores them in SQLite, and dispatches them to background worker threads via Redis queues.

```
CI/CD Pipeline
     │
     │  POST /tasks (submit)
     ▼
┌─────────────┐    SQLite (WAL)    ┌──────────────────────┐
│  FastAPI    │ ◄────────────────► │  TaskModel           │
│  Server     │                    │  WorkerModel         │
└─────────────┘                    └──────────────────────┘
     │
     │  POST /assign (trigger)
     │  enqueue(process_task, task_id)
     ▼
┌─────────────┐    FakeRedis       ┌──────────────────────┐
│  RQ Queues  │ ◄────────────────► │  high / normal / low │
│  (3 queues) │                    │  named queues        │
└─────────────┘                    └──────────────────────┘
     │
     │  blpop (blocking dequeue)
     ▼
┌─────────────────────────────────────────────────────────┐
│  Worker Threads (N daemon threads, 1 per worker_id)     │
│  Each runs _ThreadedWorker(SimpleWorker) in a loop      │
│  process_task() → sleep → complete_task() → IDLE        │
└─────────────────────────────────────────────────────────┘
```

---

## Two-Step Lifecycle

Tasks do not start automatically on submission. The pipeline must explicitly call `POST /assign` to trigger execution.

```
POST /tasks   → PENDING  (task recorded, not yet assigned)
POST /assign  → RUNNING  (matched to idle worker, pushed to Redis)
worker runs   → COMPLETED | RETRYING | FAILED
```

**Why not auto-start on submit?**
The CI/CD pipeline controls when work begins. This allows:
- Batch submission followed by a single assignment call
- Rate-limiting execution without rate-limiting intake
- Replaying a queue after infrastructure recovery without re-accepting new jobs

---

## Component Decisions

### FastAPI over Flask / Django

FastAPI generates OpenAPI docs automatically from Pydantic schemas. Sync endpoints (`def`, not `async def`) run in a threadpool, which is correct for local SQLite I/O — `async def` would block the event loop on every DB call. Django REST Framework adds unnecessary weight for a single-service API with no templating.

### SQLite + WAL mode over Postgres

Zero infrastructure to spin up. WAL (Write-Ahead Logging) mode allows multiple concurrent readers while a single writer commits, which matches the FastAPI threadpool access pattern. The default DELETE journal mode locks the entire file per write, causing contention under concurrent requests.

SQLite is the right choice for an assignment that must run locally with `uv sync`. Swap `create_engine("sqlite:///...")` for `create_engine("postgresql://...")` to migrate — the ORM layer is identical.

### Composite index `ix_tasks_assignment` on `(status, priority, submission_order)`

The hot path in `assign_tasks()` is:
```sql
SELECT ... FROM tasks
WHERE status IN (0, 2)          -- PENDING or RETRYING
ORDER BY priority DESC, submission_order ASC
LIMIT ?
```

Without the index this is a full table scan — O(n) with every assignment call. The composite index makes it O(log n). The column order matters: `status` narrows the result set first, `priority DESC` sorts high-priority tasks to the front, `submission_order ASC` enforces FIFO within a tier.

### `submission_order` integer counter over `created_at` timestamp

Timestamps from `datetime.now()` can collide within the same millisecond under load, breaking FIFO guarantees. A monotonic integer counter (`_submission_counter` seeded from `MAX(submission_order)` at startup) is strictly ordered and cheap to compare.

### RQ (Redis Queue) + FakeRedis over asyncio queues or direct threading

`asyncio.Queue` is in-process and doesn't survive restarts. Raw threading with a shared `queue.Queue` requires building job serialization, worker state management, and priority routing by hand. RQ provides all of that, plus named queues for priority lanes and a proven execution model.

FakeRedis is a thread-safe, in-process Redis emulator. Swapping it for `Redis.from_url(os.environ["REDIS_URL"])` requires changing one line in `database.py` — no other code changes.

### Three named queues (`high`, `normal`, `low`) over one queue with priority field

Workers listen in order `[high, normal, low]`. RQ's `blpop` on a list of keys checks keys left-to-right, so a worker always drains `high` before touching `normal`. This gives true preemptive priority: a HIGH task submitted while workers are processing NORMAL tasks will be picked up as soon as any worker finishes.

A single queue with a priority field would require the worker to sort the queue on dequeue, which is O(n log n) and requires locking.


### Thread-local `worker_id` over passing it as a job argument

RQ serializes job functions by import path and arguments. Passing `worker_id` as an argument would mean any worker thread could claim any worker_id, breaking the 1:1 mapping between DB workers and threads. Thread-local storage (`threading.local`) binds the worker_id to the thread at startup — `process_task()` reads it without any argument passing.

### `backoff` at the DB session level only

`backoff` wraps `get_db()` to retry on `OperationalError` (SQLite lock contention under concurrent requests). This is the right place: it handles transient infrastructure failures transparently without callers knowing.

Task retry is different — it is stateful domain logic: the DB must record `retry_count`, `submission_order` must advance (FIFO placement), and the worker must know whether to re-enqueue. That logic lives in `complete_task()`, not at the function-retry level.

### Commit before enqueue

In `POST /assign`:
```python
db.commit()   # ← status RUNNING is durable in SQLite
target_queue.enqueue(process_task, task_id)
```

If the commit happened after enqueue, a fast worker could dequeue and call `get(TaskModel, task_id)` before the HTTP session committed, finding no row. Committing first guarantees the worker always finds the task.

### SQLAlchemy `flush()` in domain logic, `commit()` in the HTTP layer

`TaskQueue` methods call `session.flush()` — they write changes to the DB transaction buffer without committing. The HTTP layer (via `get_db()`) owns the transaction and commits at the end of the request. This makes `TaskQueue` testable in isolation (callers can roll back) and prevents partial commits from leaked sessions.

---

## Data Model

```
tasks
├── task_id          TEXT PRIMARY KEY
├── priority         INTEGER  (0=LOW, 1=NORMAL, 2=HIGH)
├── max_retries      INTEGER
├── status           INTEGER  (0=PENDING, 1=RUNNING, 2=RETRYING, 3=COMPLETED, 4=FAILED)
├── retry_count      INTEGER
├── worker_id        INTEGER FK → workers.worker_id (nullable)
├── submission_order INTEGER  (monotonic counter for FIFO)
├── created_at       DATETIME
└── updated_at       DATETIME

workers
├── worker_id  INTEGER PRIMARY KEY
└── status     INTEGER  (0=IDLE, 1=BUSY, 2=FAILED)

Index: ix_tasks_assignment ON tasks(status, priority, submission_order)
```

---

## State Machine

```
              submit_task()
PENDING ◄──────────────────────────────────┐
   │                                       │
   │  assign_tasks()                       │ complete_task(success=False)
   ▼                                       │ retry_count <= max_retries
RUNNING ──────────────────────────────► RETRYING
   │
   │  complete_task(success=True)
   ▼
COMPLETED

   │  complete_task(success=False)
   │  retry_count > max_retries
   ▼
FAILED
```

---

## Test Architecture

Tests use a **session-scoped server fixture**: uvicorn starts once in a daemon thread, on a random OS-assigned port, and serves all tests. This avoids the per-test startup cost while sharing the same in-process module state — critical for `monkeypatch` to work across test and server threads.

Each test uses the `clean_state` autouse fixture which:
1. Deletes all tasks from SQLite
2. Resets all workers to IDLE
3. Deletes RQ queue keys from Redis

`monkeypatch.setattr` on `FAILURE_RATE` and `WORK_DURATION` works because worker threads are in the same process and read the same module-level variables.
