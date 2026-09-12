"""Persistence for everything the API produces: predictions, audit reports
and drift reports. SQLite by default (``configs/default.yaml``'s
``api.database_url``, zero setup) and Postgres in one line for the
containerized deployment (``docker/docker-compose.yml`` sets
``DATABASE_URL`` to a Postgres DSN) -- SQLAlchemy is the only thing that
needs to know which.

This is also the answer to the case statement's "plus: bancos de dados"
requirement in the way that made sense here: the client's own database
(built by their software team, per the case statement) already owns the
raw sensor data, so this project does not duplicate it. What this database
owns is what only the AI system produces -- predictions, audits, drift
reports -- so anything downstream (a shop-floor dashboard, a monthly
report, an alert rule) has one durable place to read from regardless of
which model version made a given call.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_name: Mapped[str] = mapped_column(String(64))
    predicted_class: Mapped[str] = mapped_column(String(64))
    probabilities: Mapped[dict] = mapped_column(JSON)
    conformal_set: Mapped[list] = mapped_column(JSON)
    is_silent: Mapped[bool] = mapped_column(default=False)
    sensor_names: Mapped[list] = mapped_column(JSON)
    top_shap_features: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class AuditRecord(Base):
    __tablename__ = "audits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    report: Mapped[dict] = mapped_column(JSON)


class DriftRecord(Base):
    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    n_alarms: Mapped[int] = mapped_column(Integer)
    n_reference: Mapped[int] = mapped_column(Integer)
    n_current: Mapped[int] = mapped_column(Integer)
    report: Mapped[dict] = mapped_column(JSON)


def get_engine(database_url: str):
    if not database_url.startswith("sqlite"):
        return create_engine(database_url)

    connect_args = {"check_same_thread": False}
    if ":memory:" in database_url:
        # SQLAlchemy's default SQLite pool opens a fresh connection per
        # checkout, and a fresh connection to ":memory:" is a brand new,
        # empty database -- the tables init_db() just created would vanish
        # on the very next query. StaticPool keeps a single connection alive
        # for the engine's lifetime, which is what makes an in-memory
        # database usable at all beyond a single connection.
        return create_engine(database_url, connect_args=connect_args, poolclass=StaticPool)
    return create_engine(database_url, connect_args=connect_args)


def init_db(database_url: str):
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
