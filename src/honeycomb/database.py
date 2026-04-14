"""
Database engine, session factory, and initialization.

Design choice: file-backed SQLite in WAL (Write-Ahead Logging) mode over the
in-memory + crash-backup pattern we considered. WAL mode persists on every commit
automatically, survives SIGKILL, and is fast enough for local I/O (sub-millisecond
writes). The in-memory approach would only protect against SIGTERM/SIGINT — real
crashes (OOM kills, power loss) would lose all state. Simpler AND more correct.

Design choice: SQLite over Postgres/MySQL. This is a self-contained assignment that
should run with zero infrastructure. SQLite in WAL mode handles concurrent FastAPI
requests well (multiple simultaneous readers, one writer at a time).
"""

import logging
import tomllib
from pathlib import Path

import backoff
from fakeredis import FakeRedis
from sqlalchemy import create_engine, event, Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from honeycomb.models import Base, WorkerModel

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    """Read [tool.honeycomb] settings from pyproject.toml."""
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data.get("tool", {}).get("honeycomb", {})


_config = _load_config()
DB_PATH: str = _config.get("db_path", "honeycomb.db")
NUM_WORKERS: int = _config.get("num_workers", 4)

# ─── Engine ───────────────────────────────────────────────────────────────────

def _create_db_engine(db_path: str) -> Engine:
    """
    Create an SQLAlchemy engine with WAL mode enabled.

    WAL mode allows concurrent readers while a write is in progress — essential
    when FastAPI handles multiple requests simultaneously. The default journal mode
    (DELETE) locks the entire database file on every write.

    check_same_thread=False: required for SQLite when used across multiple threads
    (FastAPI runs sync endpoints in a threadpool).
    """
    connect_args = {"check_same_thread": False}

    if db_path == ":memory:":
        # In-memory DB for tests — no WAL needed (single connection, no concurrency)
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={**connect_args, "uri": False},
        )
    else:
        engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args=connect_args,
        )
        # Enable WAL mode on first connection
        @event.listens_for(target=engine, identifier="connect")
        def set_wal_mode(dbapi_connection, connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")  # safe with WAL, faster than FULL
            cursor.close()

    return engine


engine = _create_db_engine(DB_PATH)

# sessionmaker factory — each request gets its own Session via get_db()
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Shared Redis connection (fakeredis = zero infrastructure, thread-safe).
# Swap for Redis.from_url(os.environ["REDIS_URL"]) in production — no other changes needed.
redis_conn = FakeRedis(decode_responses=False)

# ─── Initialization ───────────────────────────────────────────────────────────

def init_db(num_workers: int = NUM_WORKERS) -> None:
    """
    Create all tables and seed the worker pool.

    Design choice: create_all() instead of Alembic. We have no migration history —
    tables are created from scratch. Alembic adds versions/, env.py, and alembic.ini
    for zero migrations. We'd add it the moment a schema change is needed in production.
    """
    logger.info("Initializing database schema")
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as session:
        existing = session.query(WorkerModel).count()
        if existing == 0:
            workers = [WorkerModel(worker_id=i, status=0) for i in range(num_workers)]
            session.add_all(workers)
            session.commit()
            logger.info("Seeded %d workers into the worker pool", num_workers)
        else:
            logger.info("Worker pool already initialized (%d workers)", existing)


# ─── FastAPI dependency ────────────────────────────────────────────────────────

@backoff.on_exception(
    backoff.expo,
    OperationalError,
    max_tries=3,
    max_time=5,
    # Design choice: backoff is used here — at the DB session level — not in the task
    # retry logic. OperationalError from SQLite (e.g., "database is locked") is a
    # transient infrastructure failure that backoff handles perfectly. Task retry is
    # stateful domain logic (counter tracking, re-queueing) — a different problem entirely.
    on_backoff=lambda details: logger.warning(
        "DB operation failed, retrying (attempt %d): %s",
        details["tries"],
        details.get("exception"),
    ),
)
def get_db() -> Session:  # type: ignore[return]
    """
    FastAPI dependency that yields a database session.
    Session is committed on success, rolled back on exception, always closed.
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
