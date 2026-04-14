"""
SQLAlchemy ORM table definitions.

Design choice: Declarative ORM (SQLAlchemy 2.0 style with Mapped / mapped_column) over
raw SQL or Core. It gives us type-safe column definitions, FK constraints for referential
integrity, and session-based atomic transactions — all while keeping the schema readable
as documentation. Anyone reading this file understands the full DB contract.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class WorkerModel(Base):
    """
    One row per worker in the compute pool. Workers are seeded at startup.

    Design choice: store workers in the DB (not just an in-memory list) so that
    task.worker_id can carry a FK constraint. This prevents assigning tasks to
    nonexistent workers and enables future worker metadata (labels, capacity, etc.)
    without schema changes elsewhere.
    """

    __tablename__ = "workers"

    worker_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # WorkerStatus enum value — stored as int for compact storage and fast comparison
    status: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    tasks: Mapped[list["TaskModel"]] = relationship("TaskModel", back_populates="worker")


class TaskModel(Base):
    """
    One row per submitted training job. Status transitions drive the queue lifecycle:
        PENDING → RUNNING → COMPLETED
        PENDING → RUNNING → RETRYING → ... → COMPLETED
        PENDING → RUNNING → RETRYING → ... → FAILED
    """

    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String, primary_key=True)

    # Priority stored as int (HIGH=2, NORMAL=1, LOW=0) — enables ORDER BY priority DESC
    # without string comparisons or CASE expressions.
    priority: Mapped[int] = mapped_column(Integer, nullable=False)

    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)

    # TaskStatus enum value (PENDING=0, RUNNING=1, RETRYING=2, COMPLETED=3, FAILED=4)
    status: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # FK to workers — None when task is not currently assigned.
    # Referential integrity: can't assign a task to a worker that doesn't exist.
    worker_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("workers.worker_id"), nullable=True
    )

    # Monotonic counter for strict FIFO ordering within a priority tier.
    # Design choice: use a counter instead of created_at timestamp — timestamps can
    # collide within the same millisecond when tasks are submitted in a tight loop.
    # A counter guarantees strict insertion order at zero extra cost.
    submission_order: Mapped[int] = mapped_column(Integer, nullable=False)

    # Audit columns — useful for debugging and observability dashboards.
    created_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, onupdate=func.now(), nullable=True
    )

    worker: Mapped[Optional["WorkerModel"]] = relationship(
        "WorkerModel", back_populates="tasks"
    )

    # Composite covering index for the hot assignment query:
    #   WHERE status IN (PENDING, RETRYING) ORDER BY priority DESC, submission_order ASC
    # SQLite can satisfy this query entirely from the index without touching the table rows.
    # This is the single most impactful performance optimization — assignment is O(log n)
    # rather than a full table scan on every assign_tasks() call.
    __table_args__ = (
        Index("ix_tasks_assignment", "status", "priority", "submission_order"),
    )
