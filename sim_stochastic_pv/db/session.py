from __future__ import annotations

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "SQLAlchemy must be installed to use the database features "
        "(pip install sqlalchemy)."
    ) from exc

from ..config import get_database_url

DATABASE_URL = get_database_url()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)

Base = declarative_base()


def init_db() -> None:
    """
    Import models and create database tables when missing.
    """
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
