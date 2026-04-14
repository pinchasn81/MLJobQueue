"""
FastAPI application — entry point and route definitions.

Two-step job lifecycle:
  POST /tasks        — writes task to SQLite as PENDING (no execution yet)
  POST /assign       — calls assign_tasks(), pushes assigned jobs to Redis workers

This makes the assignment explicit and controllable — callers decide when to
trigger assignment, rather than tasks auto-starting on submission.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from rq import Queue
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from honeycomb.database import NUM_WORKERS, get_db, init_db, redis_conn
from honeycomb.logging_config import setup_logging
from honeycomb.models import TaskModel
from honeycomb.schemas import QueueStats, SubmitTaskRequest, TaskStatusResponse
from honeycomb.task_queue import Priority, TaskQueue, TaskStatus
from honeycomb.worker import process_task, start_workers

logger = logging.getLogger(__name__)

# Initialize 3 separate RQ queues for priority levels.
# Workers check them in this exact order: HIGH -> NORMAL -> LOW.
queues = {
    Priority.HIGH: Queue(name=Priority.HIGH.name.lower(), connection=redis_conn),
    Priority.NORMAL: Queue(name=Priority.NORMAL.name.lower(), connection=redis_conn),
    Priority.LOW: Queue(name=Priority.LOW.name.lower(), connection=redis_conn),
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_logging()
    logger.info("Starting Honeycomb ML Job Queue server")
    init_db(num_workers=NUM_WORKERS)
    
    # Start workers listening to all queues in priority order (highest value first)
    priority_queues = [queues[p] for p in sorted(queues, reverse=True)]
    start_workers(priority_queues=priority_queues)
    
    logger.info("Server ready — %d workers listening on priority queues (%s)", NUM_WORKERS, ", ".join(q.name for q in priority_queues))
    yield
    logger.info("Server shutting down")


app = FastAPI(
    title="Honeycomb ML Job Queue",
    description="Priority-based task queue for ML training jobs.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_task_queue(db: Session = Depends(get_db)) -> TaskQueue:
    """Inject a TaskQueue bound to the current request's DB session."""
    return TaskQueue(num_workers=NUM_WORKERS, session=db)


# ─── Pipeline-facing endpoints ────────────────────────────────────────────────


@app.post("/tasks", status_code=status.HTTP_201_CREATED, summary="Submit a training job")
def submit_task(
    request: SubmitTaskRequest,
    task_queue: TaskQueue = Depends(get_task_queue),
    db: Session = Depends(get_db),
) -> dict:
    """
    Submit a job to the queue. Writes PENDING to SQLite, then enqueues to Redis
    so a worker picks it up automatically — no manual assign call needed.

    We commit to SQLite before enqueueing to Redis so the worker always finds the
    task in the DB. Without this, a fast worker could pick up the job before the
    HTTP session's end-of-request commit, resulting in a "task not found" error.
    """
    try:
        task_queue.submit_task(
            task_id=request.task_id,
            priority=Priority(value=request.priority),
            max_retries=request.max_retries,
        )
    except IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Task '{request.task_id}' already exists.",
        )
    return {"task_id": request.task_id, "status": TaskStatus.PENDING.name.lower()}


@app.post("/assign", summary="Assign pending tasks to idle workers")
def assign_tasks(
    task_queue: TaskQueue = Depends(get_task_queue),
    db: Session = Depends(get_db),
) -> dict:
    """
    Assign pending (and retrying) tasks to idle workers in priority order.

    This is the trigger that actually starts execution:
      1. assign_tasks() matches idle workers to pending tasks (priority DESC, FIFO).
      2. Each assigned task is pushed to its priority-specific Redis queue.
      3. A worker thread picks it up and calls process_task().

    Returns the list of (task_id, worker_id) assignments made in this call.
    Call this endpoint whenever you want to drain the pending queue.
    """
    assignments = task_queue.assign_tasks()
    db.commit()  # commit RUNNING status before pushing to Redis

    if assignments:
        # Batch fetch priorities — one query for all assigned tasks instead of N
        assigned_ids = [task_id for task_id, _ in assignments]
        priority_map = {
            t.task_id: t.priority
            for t in db.query(TaskModel).filter(TaskModel.task_id.in_(assigned_ids)).all()
        }
        for task_id, _ in assignments:
            target_queue = queues[Priority(value=priority_map[task_id])]
            # job_timeout=-1 disables SIGALRM-based timeout (main-thread-only signal).
            target_queue.enqueue(f=process_task, args=(task_id,), job_timeout=-1)

    logger.info("Assigned %d task(s) to workers", len(assignments))
    return {
        "assigned_count": len(assignments),
        "assignments": [
            {"task_id": task_id, "worker_id": worker_id}
            for task_id, worker_id in assignments
        ],
    }


@app.get("/tasks/{task_id}", summary="Query task status")
def get_task_status(task_id: str, db: Session = Depends(get_db)) -> TaskStatusResponse:
    """Return the current status, priority, and retry count for a task."""
    task = db.get(TaskModel, task_id)
    if task is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Task '{task_id}' not found")
    task_status = TaskStatus(task.status)
    return TaskStatusResponse(
        task_id=task_id,
        status=task_status,
        status_name=task_status.name,
        priority=Priority(value=task.priority),
        retry_count=task.retry_count,
        worker_id=task.worker_id,
    )


@app.get("/stats", summary="Queue performance metrics")
def get_stats(task_queue: TaskQueue = Depends(get_task_queue)) -> QueueStats:
    return QueueStats(**task_queue.get_stats())


# ─── Entry point ──────────────────────────────────────────────────────────────


def start() -> None:
    uvicorn.run(app="honeycomb.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    start()
