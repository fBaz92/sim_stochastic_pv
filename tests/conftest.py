from __future__ import annotations

import pytest
from pathlib import Path
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_stochastic_pv.db.session import Base  # noqa: E402
from sim_stochastic_pv.persistence import PersistenceService  # noqa: E402


@pytest.fixture()
def sqlite_session_factory():
    """Provide a session factory bound to an in-memory SQLite database."""
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    yield Session
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def persistence(sqlite_session_factory):
    """Provide a PersistenceService bound to the temporary SQLite DB."""
    return PersistenceService(session_factory=sqlite_session_factory)
