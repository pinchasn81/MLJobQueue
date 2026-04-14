"""
Helper logic for integration tests, keeping test files clean and repeatable.
"""

import time

import httpx
import pytest
from assertpy import assert_that

from honeycomb.task_queue import TaskStatus
import honeycomb.worker as worker_module

# Terminal statuses — a task in one of these will never move again.
TERMINAL = {TaskStatus.COMPLETED.name, TaskStatus.FAILED.name}


def mock_rates(
    monkeypatch: pytest.MonkeyPatch, failure: float = 0.0, duration: tuple[float, float] = (0.05, 0.1)
) -> None:
    """Mock worker execution parameters to ensure deterministic test behavior."""
    monkeypatch.setattr(target=worker_module, name="FAILURE_RATE", value=failure)
    monkeypatch.setattr(target=worker_module, name="WORK_DURATION", value=duration)


def submit_tasks(client: httpx.Client, tasks: list[dict]) -> list[str]:
    """Submit a batch of tasks to the queue."""
    task_ids = []
    for task in tasks:
        resp = client.post(url="/tasks", json=task)
        assert_that(resp.status_code).described_as(f"Submit failed: {resp.text}").is_equal_to(201)
        task_ids.append(task["task_id"])
    return task_ids


def verify_tasks_statuses(client: httpx.Client, task_ids: list[str], expected_status: str) -> None:
    """Verify that all provided task IDs have the expected status name."""
    for task_id in task_ids:
        resp = client.get(url=f"/tasks/{task_id}")
        assert_that(resp.status_code).is_equal_to(200)
        assert_that(resp.json()["status_name"]).is_equal_to(expected_status)


def poll_until_terminal(
    client: httpx.Client,
    task_ids: list[str],
    timeout: float = 30.0,
) -> dict[str, str]:
    """
    Repeatedly call POST /assign and poll task statuses until all tasks reach
    a terminal state (COMPLETED or FAILED), or the timeout expires.
    """
    deadline = time.time() + timeout

    while time.time() < deadline:
        assign_resp = client.post(url="/assign")
        assert_that(assign_resp.status_code).is_equal_to(200)

        statuses = {
            task_id: client.get(url=f"/tasks/{task_id}").json()["status_name"]
            for task_id in task_ids
        }

        if all(s in TERMINAL for s in statuses.values()):
            return statuses

        time.sleep(0.2)

    # Timeout — return whatever we have for clear failure messages
    return {
        task_id: client.get(url=f"/tasks/{task_id}").json()["status_name"]
        for task_id in task_ids
    }
