import enum
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    func,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


class Base(DeclarativeBase):
    pass


class RunStatus(enum.StrEnum):
    queued = "queued"
    submitted = "submitted"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_pw: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    runs: Mapped[list["Run"]] = relationship(back_populates="user")


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    status: Mapped[RunStatus] = mapped_column(
        SAEnum(RunStatus, name="run_status"),
        default=RunStatus.queued,
        index=True,
    )
    # In-process task handle (e.g. "task-12"). Kept generic; was slurm_job_id in the brief.
    job_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    dataset: Mapped[str] = mapped_column(String(255))
    base_model: Mapped[str] = mapped_column(String(255))
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    # Training outcome: sample text, vocab size, first/last loss, device, etc.
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Path to this run's saved model checkpoint (per-user trained model instance).
    checkpoint_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    user: Mapped["User"] = relationship(back_populates="runs")
    metrics: Mapped[list["RunMetric"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="RunMetric.step",
    )


class RunMetric(Base):
    __tablename__ = "run_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    step: Mapped[int] = mapped_column(Integer)
    loss: Mapped[float] = mapped_column(Float)
    extra_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped["Run"] = relationship(back_populates="metrics")
