"""
Mock worker — simulates ML training job execution.

Design choice: workers are daemon threads running RQ's SimpleWorker, not separate
processes. SimpleWorker executes jobs in the current thread (no fork), which is
cross-platform and keeps the entire system in one process — easy to run and demo.

Design choice: worker_id is stored in thread-local storage. Each thread sets
_local.worker_id at startup; process_task reads it. Since SimpleWorker is
sequential (one job at a time per thread), there is no race condition.

Design choice: state transitions (RUNNING, BUSY) are set by assign_tasks() before
the job is pushed to Redis. process_task() only handles the outcome (success/retry/fail)
via complete_task(). Retrying tasks go back to PENDING-equivalent (RETRYING status)
and are picked up by the next POST /assign call — no direct Redis re-enqueue here.
"""

import logging
import random
import threading
import time

from rq import Queue, SimpleWorker


class _NoPenalty:
    """No-op replacement for RQ's SIGALRM-based job timeout enforcement.

    RQ's UnixSignalDeathPenalty uses SIGALRM, which only works in the main thread.
    Since our workers run as daemon threads (bounded sleep, no runaway jobs), we
    disable it entirely. The context manager protocol is still needed by RQ's internals.
    """

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> "_NoPenalty":
        return self

    def __exit__(self, *args) -> None:
        pass

    def cancel_death_penalty(self) -> None:
        pass


class _ThreadedWorker(SimpleWorker):
    """SimpleWorker safe to run in a background thread.

    Overrides both signal-based mechanisms that require the main thread:
    - _install_signal_handlers: SIGINT/SIGTERM handlers for graceful shutdown
    - death_penalty_class: SIGALRM-based job timeout enforcement
    Daemon threads are killed when the main process exits, so no signal cleanup is needed.
    """

    death_penalty_class = _NoPenalty

    def _install_signal_handlers(self) -> None:
        pass

from honeycomb.database import NUM_WORKERS, SessionLocal, redis_conn
from honeycomb.models import TaskModel, WorkerModel
from honeycomb.task_queue import Priority, TaskQueue, TaskStatus, WorkerStatus

logger = logging.getLogger(__name__)

FAILURE_RATE = 0.3        # 30% of jobs fail — demonstrates retry logic visibly
WORK_DURATION = (0.5, 3.0)  # seconds of simulated work per job

# Thread-local storage: each worker thread binds its own worker_id here at startup.
_local = threading.local()


def process_task(task_id: str) -> None:
    """
    RQ job function — called by a worker thread for each task pulled from the queue.

    Must be a module-level function (not a method or lambda) so RQ can serialize
    it by import path for job persistence in Redis.

    State on entry: task is already RUNNING and worker is already BUSY — both were
    set by assign_tasks() in the POST /assign endpoint before this job was pushed
    to Redis. This function only handles the outcome.
    """
    worker_id: int = _local.worker_id

    # Verify the task exists before doing any work
    with SessionLocal() as session:
        task = session.get(entity=TaskModel, ident=task_id)
        if task is None:
            logger.error("process_task: task not found | task_id=%s", task_id)
            return

    # ── Simulate work (outside the DB session to avoid holding a connection) ──
    duration = random.uniform(*WORK_DURATION)
    logger.info("Worker %d processing task %s (%.1fs simulated work)", worker_id, task_id, duration)
    time.sleep(duration)
    success = random.random() > FAILURE_RATE

    # ── Report result ──────────────────────────────────────────────────────────
    # complete_task() transitions: RUNNING → COMPLETED, RUNNING → RETRYING, or RUNNING → FAILED.
    # If RETRYING, the task stays in SQLite and waits for the next POST /assign call.
    with SessionLocal() as session:
        queue = TaskQueue(num_workers=NUM_WORKERS, session=session)
        queue.complete_task(task_id=task_id, success=success)
        session.commit()


def _worker_thread(worker_id: int, priority_queues: list[Queue]) -> None:
    """Thread target: binds worker_id to thread-local, then runs the RQ worker loop."""
    _local.worker_id = worker_id
    logger.info("Worker %d started", worker_id)
    # The order of queues in the list determines priority (HIGH -> NORMAL -> LOW)
    _ThreadedWorker(queues=priority_queues, connection=redis_conn).work(burst=False)


def start_workers(priority_queues: list[Queue]) -> None:
    """
    Spawn NUM_WORKERS daemon threads, each running a SimpleWorker.
    Daemon threads exit automatically when the main process stops — no cleanup needed.
    """
    for worker_id in range(NUM_WORKERS):
        t = threading.Thread(
            target=_worker_thread,
            args=(worker_id, priority_queues),
            daemon=True,
            name=f"worker-{worker_id}",
        )
        t.start()
    logger.info("Started %d mock workers listening on queues: %s", NUM_WORKERS, [q.name for q in priority_queues])
