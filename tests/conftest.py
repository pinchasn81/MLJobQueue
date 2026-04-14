"""
Pytest fixtures for the Honeycomb test suite.

Session fixture: starts the FastAPI app once on a random port in a daemon thread.
  - Uses uvicorn.Server directly (not subprocess) so the same process shares the
    module-level redis_conn and DB — monkeypatching works across test and server.
  - Waits on server.started (uvicorn's own flag) rather than polling HTTP.

Per-test fixture (autouse): resets shared state between tests:
  - Deletes all tasks from SQLite.
  - Resets all workers to IDLE (they may be BUSY from the previous test's jobs).
  - Empties all RQ queues in Redis — prevents leftover jobs from one test
    executing during the next.
"""

import socket
import threading
import time

import pytest
import uvicorn

from honeycomb.database import SessionLocal, redis_conn
from honeycomb.main import app, queues
from honeycomb.models import TaskModel, WorkerModel
from honeycomb.task_queue import WorkerStatus


def _free_port() -> int:
    """Ask the OS for a free port by binding briefly, then releasing it."""
    with socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def server_url() -> str:
    """
    Start the FastAPI app once for the entire test session.

    Scope: session — the server stays up for all tests, which avoids the cost
    of starting/stopping uvicorn per test and mirrors real usage (workers are
    long-lived).
    """
    port = _free_port()
    config = uvicorn.Config(app=app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config=config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for uvicorn to signal it has finished startup
    deadline = time.time() + 10
    while not server.started:
        if time.time() > deadline:
            raise RuntimeError("Test server did not start within 10 seconds")
        time.sleep(0.05)

    return f"http://127.0.0.1:{port}"


@pytest.fixture(autouse=True)
def clean_state() -> None:
    """
    Reset all shared state before every test.

    Three things must be cleaned:
    1. SQLite tasks — delete all rows so each test starts with an empty queue.
    2. SQLite workers — reset to IDLE (previous test may have left workers BUSY
       if a job was still in-flight when the test ended).
    3. Redis queues — empty all RQ queues so lingering jobs from the previous
       test don't execute mid-test and corrupt task state.
    """
    with SessionLocal() as session:
        session.query(TaskModel).delete()
        session.query(WorkerModel).update({"status": int(WorkerStatus.IDLE)})
        session.commit()

    # Clear all RQ queue keys directly — q.empty() uses Lua EVALSHA which
    # fakeredis doesn't support. Deleting the underlying Redis list key is equivalent.
    queue_keys = [q.key for q in queues.values()]
    if queue_keys:
        redis_conn.delete(*queue_keys)
