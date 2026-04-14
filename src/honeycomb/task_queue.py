"""
TaskQueue — core domain logic for ML training job scheduling.

Original class interface is preserved exactly (same method signatures as the assignment).
The only addition is an optional `session` parameter on __init__ for dependency injection
from the FastAPI layer.

Design choice: all state lives in SQLite (via SQLAlchemy sessions) rather than in-memory
data structures. This gives us persistence across restarts, atomic transactions for state
transitions, and SQL aggregations for stats — at sub-millisecond latency for local I/O.
"""

import logging
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from sqlalchemy import case, func
from sqlalchemy.orm import Session

from honeycomb.models import TaskModel, WorkerModel

logger = logging.getLogger(__name__)


# ─── Enums ─────────────────────────────────────────────────────────────────────
# Design choice: IntEnum instead of plain Enum. IntEnum values can be stored directly
# in SQLite INTEGER columns and used in ORDER BY expressions without serialization
# overhead. Priority ordering (HIGH > NORMAL > LOW) maps naturally to (2 > 1 > 0).


class Priority(IntEnum):
    """Priority levels for tasks."""

    LOW = 0
    NORMAL = 1
    HIGH = 2


class TaskStatus(IntEnum):
    """Task lifecycle states."""

    PENDING = 0
    RUNNING = 1
    RETRYING = 2
    COMPLETED = 3
    FAILED = 4


class WorkerStatus(IntEnum):
    """Worker availability states."""

    IDLE = 0
    BUSY = 1
    FAILED = 2


# ─── TaskQueue ─────────────────────────────────────────────────────────────────


class TaskQueue:
    """
    Priority-based task queue for ML training jobs.

    Manages job lifecycle: submission → assignment → completion/retry/failure.
    Workers are pre-seeded in the DB at startup; this class only manages their
    IDLE/BUSY status transitions.
    """

    def __init__(self, num_workers: int, session: Optional[Session] = None) -> None:
        """
        Initialize the task queue with a worker pool.

        Args:
            num_workers: Number of workers in the compute pool. Used only when
                         operating without a pre-seeded DB (e.g., standalone mode).
            session: Optional SQLAlchemy session for dependency injection (FastAPI).
                     If None, the queue operates in standalone mode (caller manages sessions).
        """
        self._num_workers = num_workers
        self._session = session
        # Monotonic counter for strict FIFO ordering within a priority tier.
        # None = not yet seeded; seeded lazily on first call to _next_order().
        self._submission_counter: Optional[int] = None

        logger.debug("TaskQueue initialized with %d workers", num_workers)

    def _next_order(self) -> int:
        if self._submission_counter is None:
            if self._session is None:
                self._submission_counter = 0
            else:
                result = self._session.query(func.max(TaskModel.submission_order)).scalar()
                self._submission_counter = (result or 0) + 1
        counter = self._submission_counter
        self._submission_counter += 1
        return counter

    # ─── Public API ──────────────────────────────────────────────────────────────

    def submit_task(
        self,
        task_id: str,
        priority: Priority = Priority.NORMAL,
        max_retries: int = 3,
    ) -> None:
        """
        Submit a new training job to the queue.

        Args:
            task_id: Unique identifier for the task (e.g., "train_model_v2")
            priority: Scheduling priority — HIGH tasks are always assigned before NORMAL/LOW
            max_retries: Maximum retry attempts after initial failure (default: 3)
        """
        task = TaskModel(
            task_id=task_id,
            priority=int(priority),
            max_retries=max_retries,
            status=int(TaskStatus.PENDING),
            retry_count=0,
            submission_order=self._next_order(),
        )
        self._session.add(task)
        self._session.commit()  # write to DB within current transaction
        logger.info(
            "Task submitted | task_id=%s priority=%s max_retries=%d",
            task_id,
            priority.name,
            max_retries,
        )

    def assign_tasks(self) -> List[Tuple[str, int, int]]:
        """
        Assign pending tasks to idle workers, respecting priority then FIFO order.

        Design choice: single sorted query + bulk update in one transaction.
        The composite index on (status, priority, submission_order) makes this
        O(log n) instead of a full table scan. All assignments are committed atomically
        — no partial state where some workers show busy but their tasks aren't recorded.

        Returns:
            List of (task_id, worker_id) assignments made in this call.
        """
        # Fetch idle workers — small result set, no index needed
        idle_workers = (
            self._session.query(WorkerModel)
            .filter(WorkerModel.status == int(WorkerStatus.IDLE))
            .all()
        )
        if not idle_workers:
            logger.debug("No idle workers available for assignment")
            return []

        # Fetch pending/retrying tasks ordered by priority DESC then FIFO
        # Limit to available idle workers — no point fetching more than we can assign
        pending_tasks = (
            self._session.query(TaskModel)
            .filter(
                TaskModel.status.in_(
                    [int(TaskStatus.PENDING), int(TaskStatus.RETRYING)]
                )
            )
            .order_by(TaskModel.priority.desc(), TaskModel.submission_order.asc())
            .limit(len(idle_workers))
            .all()
        )
        if not pending_tasks:
            logger.debug("No pending tasks to assign")
            return []

        assignments: List[Tuple[str, int, int]] = []

        for task, worker in zip(pending_tasks, idle_workers):
            task.status = int(TaskStatus.RUNNING)
            task.worker_id = worker.worker_id
            worker.status = int(WorkerStatus.BUSY)

            assignments.append((task.task_id, worker.worker_id, task.priority))
            logger.info(
                "Task assigned | task_id=%s worker_id=%d", task.task_id, worker.worker_id
            )

        self._session.flush()
        return assignments

    def complete_task(self, task_id: str, success: bool) -> None:
        """
        Mark a running task as completed or failed.

        On success: status → COMPLETED, worker freed.
        On failure with retries remaining: status → RETRYING, re-queued at back of
            its priority tier (new submission_order), worker freed.
        On failure with no retries: status → FAILED, worker freed.

        Design choice: the entire state transition (task update + worker free) happens
        in one flush — callers see either the full transition or none of it.

        Backoff calculation: 2^retry_count seconds. We track the value but don't sleep
        — in a real system this would be a scheduled delay. Here it's observable via
        the task record for monitoring purposes.

        Args:
            task_id: ID of the task that finished
            success: True if task succeeded, False if it failed
        """
        task = self._session.query(TaskModel).filter_by(task_id=task_id).first()
        if task is None:
            logger.error("complete_task called for unknown task_id=%s", task_id)
            return

        # Free the worker regardless of outcome
        if task.worker_id is not None:
            worker = self._session.get(entity=WorkerModel, ident=task.worker_id)
            if worker:
                worker.status = int(WorkerStatus.IDLE)
            task.worker_id = None

        if success:
            task.status = int(TaskStatus.COMPLETED)
            logger.info("Task completed successfully | task_id=%s", task_id)

        else:
            task.retry_count += 1
            if task.retry_count <= task.max_retries:
                # Re-enqueue at the back of its priority tier
                # New submission_order places it after all currently-pending tasks
                task.status = int(TaskStatus.RETRYING)
                task.submission_order = self._next_order()
                logger.warning(
                    "Task failed, scheduling retry | task_id=%s retry=%d/%d backoff=%ds",
                    task_id,
                    task.retry_count,
                    task.max_retries,
                    2 ** task.retry_count,
                )
            else:
                task.status = int(TaskStatus.FAILED)
                logger.error(
                    "Task permanently failed | task_id=%s total_retries=%d",
                    task_id,
                    task.retry_count - 1,  # subtract the final failed attempt
                )

        self._session.flush()

    def get_task_status(self, task_id: str) -> TaskStatus:
        """
        Query the current status of a task.

        Args:
            task_id: Task identifier

        Returns:
            TaskStatus enum value (PENDING, RUNNING, RETRYING, COMPLETED, FAILED)
        """
        task = self._session.query(TaskModel).filter_by(task_id=task_id).first()
        if task is None:
            logger.warning("get_task_status called for unknown task_id=%s", task_id)
            raise KeyError(f"Task '{task_id}' not found")
        return TaskStatus(task.status)

    def get_stats(self) -> Dict[str, float]:
        """
        Get queue performance statistics via SQL aggregations.

        Design choice: delegate aggregation to the DB engine rather than fetching all
        rows and computing in Python. COUNT/AVG on an indexed column is O(log n) in
        SQLite; pulling all rows would be O(n) in memory.

        Returns:
            Dict with: total_submitted, total_completed, total_failed,
                       completion_rate, avg_retries, worker_utilization
        """
        # Single query for all task metrics via conditional aggregation
        task_row = self._session.query(
            func.count(TaskModel.task_id).label("total"),
            func.sum(case((TaskModel.status == int(TaskStatus.COMPLETED), 1), else_=0)).label("completed"),
            func.sum(case((TaskModel.status == int(TaskStatus.FAILED), 1), else_=0)).label("failed"),
            func.avg(TaskModel.retry_count).label("avg_retries"),
        ).one()
        total_submitted = task_row.total or 0
        completed = task_row.completed or 0
        failed = task_row.failed or 0
        avg_retries = float(task_row.avg_retries or 0.0)

        # Single query for worker metrics
        worker_row = self._session.query(
            func.count(WorkerModel.worker_id).label("total"),
            func.sum(case((WorkerModel.status == int(WorkerStatus.BUSY), 1), else_=0)).label("busy"),
        ).one()
        total_workers = worker_row.total or 0
        busy_workers = worker_row.busy or 0

        finished = completed + failed
        # Guard against division by zero when no tasks have finished yet
        completion_rate = completed / finished if finished > 0 else 0.0
        worker_utilization = busy_workers / total_workers if total_workers > 0 else 0.0

        return {
            "total_submitted": float(total_submitted),
            "total_completed": float(completed),
            "total_failed": float(failed),
            "completion_rate": completion_rate,
            "avg_retries": float(avg_retries),
            "worker_utilization": worker_utilization,
        }

    def num_pending(self) -> int:
        """Return the number of tasks in PENDING or RETRYING status."""
        return (
            self._session.query(func.count(TaskModel.task_id))
            .filter(
                TaskModel.status.in_(
                    [int(TaskStatus.PENDING), int(TaskStatus.RETRYING)]
                )
            )
            .scalar()
            or 0
        )

    def num_running(self) -> int:
        """Return the number of tasks currently RUNNING."""
        return (
            self._session.query(func.count(TaskModel.task_id))
            .filter(TaskModel.status == int(TaskStatus.RUNNING))
            .scalar()
            or 0
        )
