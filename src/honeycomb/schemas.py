"""
Pydantic request/response schemas for the API layer.

Design choice: separate schemas.py from models.py (ORM). The API contract and the
database schema look similar now but evolve independently — API fields may be renamed,
versioned, or filtered without touching the DB schema.
"""

from typing import Optional

from pydantic import BaseModel, Field

from honeycomb.task_queue import Priority, TaskStatus


class SubmitTaskRequest(BaseModel):
    task_id: str = Field(description="Unique identifier for the training job")
    priority: Priority = Field(default=Priority.NORMAL, description="HIGH=2, NORMAL=1, LOW=0")
    max_retries: int = Field(default=3, ge=0, description="Max retry attempts after failure")

    model_config = {"use_enum_values": True}


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    status_name: str
    priority: Priority
    retry_count: int
    worker_id: Optional[int]

    model_config = {"use_enum_values": True}


class QueueStats(BaseModel):
    total_submitted: float
    total_completed: float
    total_failed: float
    completion_rate: float = Field(description="Success rate among finished tasks (0–1)")
    avg_retries: float = Field(description="Average retries per submitted task")
    worker_utilization: float = Field(description="Proportion of workers currently busy (0–1)")
