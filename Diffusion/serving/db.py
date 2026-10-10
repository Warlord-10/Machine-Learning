"""Durable source of truth: the `jobs` and `images` tables.

Redis is a best-effort cache in front of this; the DB is authoritative for
terminal state. SQLite by default (swap DB_URL for Postgres in prod).
"""

from __future__ import annotations

import datetime as dt
from contextlib import contextmanager

from sqlalchemy import (JSON, DateTime, ForeignKey, String, UniqueConstraint,
                        create_engine, func)
from sqlalchemy.orm import (DeclarativeBase, Mapped, mapped_column, relationship,
                            sessionmaker)

from config import DB_URL

# Terminal + in-flight job states. In-flight lives mostly in Redis; the DB row
# is created 'pending' at submit so a job that dies leaves a trace.
PENDING, STARTED, SUCCESS, FAILED, CANCELLED = (
    "pending", "started", "success", "failed", "cancelled")
TERMINAL = {SUCCESS, FAILED, CANCELLED}

engine = create_engine(DB_URL, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)   # uuid4 hex
    created_by: Mapped[str] = mapped_column(String(64), index=True)  # client IP (demo identity)
    params: Mapped[dict] = mapped_column(JSON)                       # {labels, seed, sampler, steps}
    status: Mapped[str] = mapped_column(String(16), default=PENDING, index=True)
    reason: Mapped[str | None] = mapped_column(String(128), nullable=True)  # NA on success
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)

    images: Mapped[list["Image"]] = relationship(back_populates="job")


class Image(Base):
    __tablename__ = "images"
    # Celery runs with task_acks_late=True, so a worker crash REDELIVERS the task.
    # The S3 key is idempotent (same bytes overwrite); this insert is not. Without
    # the constraint a retry silently duplicates rows.
    __table_args__ = (UniqueConstraint("job_id", "s3_key", name="uq_images_job_key"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    created_by: Mapped[str] = mapped_column(String(64), index=True)
    s3_key: Mapped[str] = mapped_column(String(256))   # object key, NOT a presigned URL
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, server_default=func.now())

    job: Mapped["Job"] = relationship(back_populates="images")


def init_db() -> None:
    Base.metadata.create_all(engine)


@contextmanager
def session_scope():
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
