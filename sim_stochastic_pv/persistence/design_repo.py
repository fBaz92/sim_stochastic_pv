"""
Plant-design repository for CRUD operations on :class:`PlantDesignModel`.

A plant design ("Impianto") is the first-class description of one specific
PV system — either a received commercial offer (``essential`` level) or a
full electrical design (``detailed`` level). This repository mirrors the
shape of the other aggregate repositories so the
:class:`~sim_stochastic_pv.persistence.PersistenceService` facade exposes
them symmetrically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..db.models import PlantDesignModel


class PlantDesignRepository:
    """
    CRUD operations for plant designs.

    Attributes:
        _session_factory: SQLAlchemy session factory for database
            connections.

    Example:
        ```python
        from sim_stochastic_pv.db.session import SessionLocal
        from sim_stochastic_pv.persistence.design_repo import PlantDesignRepository

        repo = PlantDesignRepository(SessionLocal)
        design = repo.upsert_design({
            "name": "Offerta Rossi 6kW",
            "design_level": "essential",
            "data": {"p_ac_kw": 6.0, "total_cost_eur": 14500.0},
        })
        ```

    Notes:
        - ``name`` is the unique upsert key, consistent with every other
          repository in this package.
        - All methods open their own short-lived session and commit before
          returning (records are detached and safe to read afterwards).
    """

    def __init__(self, session_factory) -> None:
        """
        Initialize repository with a session factory.

        Args:
            session_factory: SQLAlchemy sessionmaker instance used to open
                a fresh session per operation.
        """
        self._session_factory = session_factory

    def upsert_design(self, data: Dict[str, Any]) -> PlantDesignModel:
        """
        Insert or update a plant design by unique ``name``.

        Args:
            data: Field dict matching :class:`PlantDesignModel` columns.
                Must include ``name`` (str) and ``data`` (dict payload).
                Optional: ``design_level`` (default ``"essential"``),
                ``description``, ``inverter_id``, ``panel_id``,
                ``battery_id``, ``location_id``.

        Returns:
            The created or updated record (refreshed, detached).

        Raises:
            KeyError: If the required ``name`` key is missing.
        """
        with self._session_factory() as session:
            existing = (
                session.query(PlantDesignModel)
                .filter_by(name=data["name"])
                .first()
            )
            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
                record = existing
            else:
                record = PlantDesignModel(**data)
                session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def list_designs(self) -> List[PlantDesignModel]:
        """
        List all plant designs ordered by name.

        Returns:
            All records sorted alphabetically (empty list when none exist).
        """
        with self._session_factory() as session:
            return (
                session.query(PlantDesignModel)
                .order_by(PlantDesignModel.name)
                .all()
            )

    def get_design_by_id(self, design_id: int) -> Optional[PlantDesignModel]:
        """
        Fetch a design by primary key.

        Args:
            design_id: Primary key of the design.

        Returns:
            The matching record, or ``None`` when not found.
        """
        with self._session_factory() as session:
            return session.get(PlantDesignModel, design_id)

    def get_design_by_name(self, name: str) -> Optional[PlantDesignModel]:
        """
        Fetch a design by its unique name (case-sensitive exact match).

        Args:
            name: Unique design identifier.

        Returns:
            The matching record, or ``None`` when not found.
        """
        with self._session_factory() as session:
            return session.query(PlantDesignModel).filter_by(name=name).first()

    def update_design(
        self, design_id: int, data: Dict[str, Any]
    ) -> Optional[PlantDesignModel]:
        """
        Update a design by primary key (partial — allows rename).

        Only the keys present in ``data`` are written. Renaming is checked
        against the unique constraint so the caller gets a clean error
        instead of an IntegrityError.

        Args:
            design_id: Primary key of the design to update.
            data: New field values (any subset of columns).

        Returns:
            The updated record, or ``None`` when ``design_id`` does not
            exist.

        Raises:
            ValueError: If the requested ``name`` is already used by a
                different design.
        """
        if not data:
            return self.get_design_by_id(design_id)
        with self._session_factory() as session:
            record = session.get(PlantDesignModel, design_id)
            if record is None:
                return None
            new_name = data.get("name")
            if new_name and new_name != record.name:
                clash = (
                    session.query(PlantDesignModel)
                    .filter(PlantDesignModel.name == new_name)
                    .first()
                )
                if clash is not None and clash.id != design_id:
                    raise ValueError(
                        f"Plant design name '{new_name}' is already used by id={clash.id}"
                    )
            for key, value in data.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            session.commit()
            session.refresh(record)
            return record

    def delete_design(self, design_id: int) -> bool:
        """
        Delete a design by primary key.

        Runs already executed from this design are untouched (their config
        snapshot is frozen in the run record).

        Args:
            design_id: Primary key of the design to delete.

        Returns:
            ``True`` if the design existed and was deleted, ``False``
            otherwise.
        """
        with self._session_factory() as session:
            record = session.get(PlantDesignModel, design_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True
