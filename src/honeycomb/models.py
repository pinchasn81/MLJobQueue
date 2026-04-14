"""
SQLAlchemy ORM table definitions.

Design choice: Declarative ORM (SQLAlchemy 2.0 style with Mapped / mapped_column) over
raw SQL or Core. It gives us type-safe column definitions, FK constraints for referential
integrity, and session-based atomic transactions — all while keeping the schema readable
as documentation. Anyone reading this file understands the full DB contract.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Index, Integer, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


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

    # Slot index (0..NUM_WORKERS-1) set while task is RUNNING, cleared on completion.
    worker_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

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

    # Composite covering index for the hot assignment query:
    #   WHERE status IN (PENDING, RETRYING) ORDER BY priority DESC, submission_order ASC
    # SQLite can satisfy this query entirely from the index without touching the table rows.
    # This is the single most impactful performance optimization — assignment is O(log n)
    # rather than a full table scan on every assign_tasks() call.
    __table_args__ = (
        Index("ix_tasks_assignment", "status", "priority", "submission_order"),
    )
