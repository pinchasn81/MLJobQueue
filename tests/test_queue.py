"""
Integration tests for the Honeycomb ML Job Queue API.

Tests call the real HTTP server (started by the session fixture in conftest.py)
and verify end-to-end behaviour including worker execution. Each test uses the
autouse clean_state fixture to start from a known-empty DB.
"""

import time

import httpx
import pytest
from assertpy import assert_that

from honeycomb.task_queue import Priority, TaskStatus
from tests.logics import (
    mock_rates,
    poll_until_terminal,
    submit_tasks,
    verify_tasks_statuses,
)


class TestQueueAPI:
    def test_all_priorities_complete(self, server_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Submit 9 tasks (3 per priority), verify all reach COMPLETED."""
        mock_rates(monkeypatch=monkeypatch, failure=0.0, duration=(0.05, 0.1))
        client = httpx.Client(base_url=server_url, timeout=10)

        tasks_to_submit = [
            {"task_id": f"task_{priority.name.lower()}_{i}", "priority": priority.value, "max_retries": 3}
            for priority in [Priority.HIGH, Priority.NORMAL, Priority.LOW]
            for i in range(3)
        ]
        task_ids = submit_tasks(client=client, tasks=tasks_to_submit)
        verify_tasks_statuses(client=client, task_ids=task_ids, expected_status=TaskStatus.PENDING.name)
        poll_until_terminal(client=client, task_ids=task_ids, timeout=30)
        verify_tasks_statuses(client=client, task_ids=task_ids, expected_status=TaskStatus.COMPLETED.name)

    def test_priority_ordering(self, server_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """HIGH priority tasks are assigned before LOW when workers are limited."""
        mock_rates(monkeypatch=monkeypatch, failure=0.0, duration=(0.05, 0.1))
        client = httpx.Client(base_url=server_url, timeout=10)

        # Submit 2 LOW first, then 4 HIGH — queue has exactly 4 workers
        submit_tasks(client=client, tasks=[
            {"task_id": f"low_{i}", "priority": Priority.LOW.value, "max_retries": 0} for i in range(2)
        ])
        high_ids = submit_tasks(client=client, tasks=[
            {"task_id": f"high_{i}", "priority": Priority.HIGH.value, "max_retries": 0} for i in range(4)
        ])

        # One /assign call fills all 4 workers — the response tells us exactly what was assigned
        resp = client.post("/assign")
        assigned_ids = {a["task_id"] for a in resp.json()["assignments"]}
        assert_that(resp.json()["assigned_count"]).is_equal_to(4)
        # All 4 assigned slots should be HIGH tasks, not LOW ones
        assert_that(assigned_ids).is_equal_to(set(high_ids))

    def test_retry_then_complete(self, server_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A task that fails once succeeds on the next retry."""
        mock_rates(monkeypatch=monkeypatch, failure=1.0, duration=(0.05, 0.1))
        client = httpx.Client(base_url=server_url, timeout=10)

        submit_tasks(client=client, tasks=[{"task_id": "retry_task", "priority": Priority.NORMAL.value, "max_retries": 3}])

        # Trigger first execution — do NOT call /assign again until we've confirmed RETRYING.
        # poll_until_terminal calls /assign on every iteration, which would immediately
        # re-assign the RETRYING task back to RUNNING before we can observe it.
        client.post("/assign")

        # Wait for the worker to finish its 0.05-0.1 s job and mark the task RETRYING
        deadline = time.time() + 10
        while time.time() < deadline:
            resp = client.get("/tasks/retry_task").json()
            if resp["status_name"] == TaskStatus.RETRYING.name:
                break
            time.sleep(0.05)

        assert_that(resp["status_name"]).is_equal_to(TaskStatus.RETRYING.name)
        assert_that(resp["retry_count"]).is_equal_to(1)

        # Allow the retry to succeed and drive to completion
        mock_rates(monkeypatch=monkeypatch, failure=0.0, duration=(0.05, 0.1))
        poll_until_terminal(client=client, task_ids=["retry_task"], timeout=15)

        final = client.get("/tasks/retry_task").json()
        assert_that(final["status_name"]).is_equal_to(TaskStatus.COMPLETED.name)
        assert_that(final["retry_count"]).is_equal_to(1)

    def test_max_retries_exhausted(self, server_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """A task with max_retries=0 is permanently FAILED on its first failure."""
        mock_rates(monkeypatch=monkeypatch, failure=1.0, duration=(0.05, 0.1))
        client = httpx.Client(base_url=server_url, timeout=10)

        submit_tasks(client=client, tasks=[{"task_id": "doomed_task", "priority": Priority.NORMAL.value, "max_retries": 0}])
        poll_until_terminal(client=client, task_ids=["doomed_task"], timeout=15)

        assert_that(client.get("/tasks/doomed_task").json()["status_name"]).is_equal_to(TaskStatus.FAILED.name)

    def test_duplicate_task_rejected(self, server_url: str) -> None:
        """Submitting the same task_id twice returns 409 Conflict."""
        client = httpx.Client(base_url=server_url, timeout=10)
        payload = {"task_id": "dup_task", "priority": Priority.NORMAL.value, "max_retries": 3}

        assert_that(client.post("/tasks", json=payload).status_code).is_equal_to(201)
        assert_that(client.post("/tasks", json=payload).status_code).is_equal_to(409)

    def test_unknown_task_returns_404(self, server_url: str) -> None:
        """GET /tasks/{id} for a non-existent task returns 404."""
        client = httpx.Client(base_url=server_url, timeout=10)
        assert_that(client.get("/tasks/ghost_task").status_code).is_equal_to(404)

    def test_worker_capacity_limit(self, server_url: str) -> None:
        """A single /assign call assigns at most NUM_WORKERS tasks (4)."""
        client = httpx.Client(base_url=server_url, timeout=10)

        # Submit 8 tasks — more than the 4-worker pool
        task_ids = submit_tasks(
            client=client,
            tasks=[{"task_id": f"cap_{i}", "priority": Priority.NORMAL.value, "max_retries": 0} for i in range(8)],
        )

        # Verify the response: only 4 slots available
        resp = client.post("/assign")
        assert_that(resp.json()["assigned_count"]).is_equal_to(4)
        assert_that(resp.json()["assignments"]).is_length(4)

        # 4 more tasks remain pending — second call assigns the rest once workers free up
        resp2 = client.post("/assign")
        assert_that(resp2.json()["assigned_count"]).is_equal_to(0)
        assert_that(len([t for t in task_ids if client.get(f"/tasks/{t}").json()["status_name"] == TaskStatus.PENDING.name])).is_equal_to(4)

    def test_stats_accuracy(self, server_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stats reflect actual completion/failure counts after all tasks finish."""
        mock_rates(monkeypatch=monkeypatch, failure=0.0, duration=(0.05, 0.1))
        client = httpx.Client(base_url=server_url, timeout=10)

        task_ids = submit_tasks(
            client=client,
            tasks=[{"task_id": f"stat_{i}", "priority": Priority.NORMAL.value, "max_retries": 3} for i in range(4)],
        )
        poll_until_terminal(client=client, task_ids=task_ids, timeout=30)

        stats = client.get("/stats").json()
        assert_that(stats["total_submitted"]).is_equal_to(4.0)
        assert_that(stats["total_completed"]).is_equal_to(4.0)
        assert_that(stats["total_failed"]).is_equal_to(0.0)
        assert_that(stats["completion_rate"]).is_equal_to(1.0)
        
        # Verify all workers back to IDLE
        for worker_id, status in stats["workers"].items():
            assert_that(status).is_equal_to("IDLE")

    def test_worker_status_while_running(self, server_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stats reflect busy workers while tasks are executing."""
        # Use a long duration to ensure we catch them running
        mock_rates(monkeypatch=monkeypatch, failure=0.0, duration=(1.0, 2.0))
        client = httpx.Client(base_url=server_url, timeout=10)

        task_id = "busy_task"
        submit_tasks(client=client, tasks=[{"task_id": task_id, "priority": Priority.NORMAL.value, "max_retries": 0}])
        client.post("/assign")

        # Give it a tiny bit of time to start
        time.sleep(0.1)

        stats = client.get("/stats").json()
        # At least one worker should be busy with our task
        busy_workers = [v for v in stats["workers"].values() if v == task_id]
        assert_that(busy_workers).is_length(1)
        
        # Other workers should be IDLE (default is 4 workers)
        idle_workers = [v for v in stats["workers"].values() if v == "IDLE"]
        assert_that(idle_workers).is_length(3)

    def test_load_high_throughput(self, server_url: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """100 tasks complete within 60 s; stats show full completion and low avg retries."""
        mock_rates(monkeypatch=monkeypatch, failure=0.0, duration=(0.02, 0.05))
        client = httpx.Client(base_url=server_url, timeout=30)

        task_ids = submit_tasks(
            client=client,
            tasks=[
                {"task_id": f"load_{i}", "priority": (i % 3), "max_retries": 2}
                for i in range(100)
            ],
        )

        start = time.time()
        final_statuses = poll_until_terminal(client=client, task_ids=task_ids, timeout=60)
        elapsed = time.time() - start

        completed = sum(1 for s in final_statuses.values() if s == TaskStatus.COMPLETED.name)
        assert_that(completed).is_equal_to(100)
        assert_that(elapsed).is_less_than(60)

        stats = client.get("/stats").json()
        assert_that(stats["total_completed"]).is_equal_to(100.0)
        assert_that(stats["completion_rate"]).is_equal_to(1.0)
        assert_that(stats["avg_retries"]).is_less_than(1.0)
