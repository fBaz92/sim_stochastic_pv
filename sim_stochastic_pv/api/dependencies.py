from __future__ import annotations

from functools import lru_cache

from ..application import SimulationApplication
from ..persistence import PersistenceService
from ..result_builder import ResultBuilder
from ..db.session import init_db


@lru_cache()
def get_persistence_service() -> PersistenceService:
    """
    Provide a cached PersistenceService instance for API routes.
    """
    init_db()
    return PersistenceService()


def get_result_builder() -> ResultBuilder:
    """
    Provide a ResultBuilder for optional CLI-style exports.
    """
    return ResultBuilder()


def get_application_service() -> SimulationApplication:
    """
    Provide a SimulationApplication configured for API usage.
    """
    persistence = get_persistence_service()
    # API does not save graphical outputs by default
    return SimulationApplication(
        save_outputs=False,
        persistence=persistence,
        result_builder=None,
    )
