"""
Location repository for CRUD operations on installation sites.

Provides the database access layer for :class:`LocationModel` plus the
single-transaction persistence step used by the unified "add location"
import flow (location + solar profile + climate profile written atomically).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..db.models import ClimateProfileModel, LocationModel, SolarProfileModel


class LocationRepository:
    """
    Repository for installation-site CRUD and transactional profile linking.

    A location is the durable anchor of every site-specific dataset (solar
    production profile, calibrated climate profile). This repository owns
    both the plain CRUD operations and :meth:`persist_import` — the single
    write transaction behind ``POST /api/locations/import`` that guarantees
    the address and its downloaded profiles are saved together or not at all.

    Attributes:
        _session_factory: SQLAlchemy session factory for database connections.

    Example:
        ```python
        from sim_stochastic_pv.db.session import SessionLocal
        from sim_stochastic_pv.persistence.location_repo import LocationRepository

        repo = LocationRepository(SessionLocal)
        site = repo.upsert_location({
            "name": "Pavullo",
            "display_name": "Pavullo nel Frignano, Modena, Italia",
            "latitude": 44.336,
            "longitude": 10.831,
        })
        solar_list, climate_list = repo.linked_profiles(site.id)
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

    def upsert_location(self, data: Dict[str, Any]) -> LocationModel:
        """
        Insert or update a location by unique ``name``.

        Args:
            data: Field dict matching :class:`LocationModel` columns. Must
                include ``name`` (str), ``latitude`` and ``longitude``
                (float). Optional: ``address``, ``display_name``,
                ``elevation_m``, ``notes``.

        Returns:
            The created or updated :class:`LocationModel` record (refreshed,
            detached).

        Raises:
            KeyError: If the required ``name`` key is missing.

        Example:
            ```python
            site = repo.upsert_location({
                "name": "Pavullo", "latitude": 44.336, "longitude": 10.831,
            })
            ```
        """
        with self._session_factory() as session:
            existing = (
                session.query(LocationModel).filter_by(name=data["name"]).first()
            )
            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
                record = existing
            else:
                record = LocationModel(**data)
                session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def list_locations(self) -> List[LocationModel]:
        """
        List all locations ordered by name.

        Returns:
            All :class:`LocationModel` records sorted alphabetically
            (empty list when none exist).
        """
        with self._session_factory() as session:
            return (
                session.query(LocationModel).order_by(LocationModel.name).all()
            )

    def get_location_by_id(self, location_id: int) -> Optional[LocationModel]:
        """
        Fetch a location by primary key.

        Args:
            location_id: Primary key of the location.

        Returns:
            The matching record, or ``None`` when not found.
        """
        with self._session_factory() as session:
            return session.get(LocationModel, location_id)

    def get_location_by_name(self, name: str) -> Optional[LocationModel]:
        """
        Fetch a location by its unique name (case-sensitive exact match).

        Args:
            name: Unique short identifier of the site.

        Returns:
            The matching record, or ``None`` when not found.
        """
        with self._session_factory() as session:
            return session.query(LocationModel).filter_by(name=name).first()

    def update_location(
        self, location_id: int, data: Dict[str, Any]
    ) -> Optional[LocationModel]:
        """
        Update a location by primary key (partial — allows rename).

        Only the keys present in ``data`` are written. Renaming is checked
        against the unique constraint so the caller gets a clean error
        instead of an IntegrityError.

        Args:
            location_id: Primary key of the location to update.
            data: New field values (any subset of :class:`LocationModel`
                columns).

        Returns:
            The updated record, or ``None`` when ``location_id`` does not
            exist.

        Raises:
            ValueError: If the requested ``name`` is already used by a
                different location.
        """
        if not data:
            return self.get_location_by_id(location_id)
        with self._session_factory() as session:
            record = session.get(LocationModel, location_id)
            if record is None:
                return None
            new_name = data.get("name")
            if new_name and new_name != record.name:
                clash = (
                    session.query(LocationModel)
                    .filter(LocationModel.name == new_name)
                    .first()
                )
                if clash is not None and clash.id != location_id:
                    raise ValueError(
                        f"Location name '{new_name}' is already used by id={clash.id}"
                    )
            for key, value in data.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            session.commit()
            session.refresh(record)
            return record

    def delete_location(
        self, location_id: int, *, delete_profiles: bool = False
    ) -> bool:
        """
        Delete a location, detaching or deleting its linked profiles.

        By default the linked solar/climate profiles survive with their
        ``location_id`` set to NULL (they remain usable by scenarios that
        reference them by id). With ``delete_profiles=True`` the profiles
        are hard-deleted in the same transaction.

        Args:
            location_id: Primary key of the location to delete.
            delete_profiles: When ``True``, also delete every solar and
                climate profile linked to this location. Default ``False``
                (detach only).

        Returns:
            ``True`` if the location existed and was deleted, ``False``
            otherwise.
        """
        with self._session_factory() as session:
            record = session.get(LocationModel, location_id)
            if record is None:
                return False
            solar_rows = (
                session.query(SolarProfileModel)
                .filter_by(location_id=location_id)
                .all()
            )
            climate_rows = (
                session.query(ClimateProfileModel)
                .filter_by(location_id=location_id)
                .all()
            )
            for row in (*solar_rows, *climate_rows):
                if delete_profiles:
                    session.delete(row)
                else:
                    row.location_id = None
            session.delete(record)
            session.commit()
            return True

    def linked_profiles(
        self, location_id: int
    ) -> Tuple[List[SolarProfileModel], List[ClimateProfileModel]]:
        """
        Fetch the solar and climate profiles owned by a location.

        Args:
            location_id: Primary key of the location.

        Returns:
            Tuple ``(solar_profiles, climate_profiles)`` ordered by name.
            Either list may be empty (e.g. site saved without downloads).
        """
        with self._session_factory() as session:
            solar = (
                session.query(SolarProfileModel)
                .filter_by(location_id=location_id)
                .order_by(SolarProfileModel.name)
                .all()
            )
            climate = (
                session.query(ClimateProfileModel)
                .filter_by(location_id=location_id)
                .order_by(ClimateProfileModel.name)
                .all()
            )
            return solar, climate

    def persist_import(
        self,
        location_data: Dict[str, Any],
        solar_data: Optional[Dict[str, Any]] = None,
        climate_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[
        LocationModel,
        Optional[SolarProfileModel],
        Optional[ClimateProfileModel],
    ]:
        """
        Persist a location and its downloaded profiles in ONE transaction.

        This is the write half of the unified "add location" flow: the
        caller fetches every external dataset *first* (PVGIS, Open-Meteo)
        and only then calls this method, so a network failure can never
        leave half-saved state, and a database failure rolls back the
        whole import including the location row.

        Profiles are upserted by their unique ``name`` and linked to the
        location via ``location_id``. An existing same-named profile that
        predates the locations table (legacy row) is updated in place and
        adopted by the location — this is the migration path for profiles
        created before this flow existed.

        Args:
            location_data: Field dict for :meth:`upsert_location` (must
                include ``name``, ``latitude``, ``longitude``).
            solar_data: Optional field dict for the solar profile (same
                schema as ``SolarProfileRepository.upsert_solar_profile``,
                without ``location_id`` — it is set here). ``None`` skips
                the solar write.
            climate_data: Optional field dict for the climate profile
                (same schema as
                ``ClimateProfileRepository.upsert_climate_profile``,
                without ``location_id``). ``None`` skips the climate write.

        Returns:
            Tuple ``(location, solar_profile | None, climate_profile | None)``
            of refreshed, detached records.

        Raises:
            KeyError: If ``location_data`` lacks the ``name`` key.
            sqlalchemy.exc.SQLAlchemyError: On database failure — nothing
                is committed in that case.

        Example:
            ```python
            site, solar, climate = repo.persist_import(
                {"name": "Pavullo", "latitude": 44.336, "longitude": 10.831},
                solar_data={...},     # fetched from PVGIS beforehand
                climate_data=None,    # climate download skipped/failed
            )
            ```
        """
        with self._session_factory() as session:
            location = (
                session.query(LocationModel)
                .filter_by(name=location_data["name"])
                .first()
            )
            if location:
                for key, value in location_data.items():
                    setattr(location, key, value)
            else:
                location = LocationModel(**location_data)
                session.add(location)
            session.flush()  # populate location.id for the FK links below

            solar_record: Optional[SolarProfileModel] = None
            if solar_data is not None:
                solar_record = (
                    session.query(SolarProfileModel)
                    .filter_by(name=solar_data["name"])
                    .first()
                )
                if solar_record:
                    for key, value in solar_data.items():
                        setattr(solar_record, key, value)
                else:
                    solar_record = SolarProfileModel(**solar_data)
                    session.add(solar_record)
                solar_record.location_id = location.id

            climate_record: Optional[ClimateProfileModel] = None
            if climate_data is not None:
                climate_record = (
                    session.query(ClimateProfileModel)
                    .filter_by(name=climate_data["name"])
                    .first()
                )
                if climate_record:
                    for key, value in climate_data.items():
                        setattr(climate_record, key, value)
                else:
                    climate_record = ClimateProfileModel(**climate_data)
                    session.add(climate_record)
                climate_record.location_id = location.id

            session.commit()
            session.refresh(location)
            if solar_record is not None:
                session.refresh(solar_record)
            if climate_record is not None:
                session.refresh(climate_record)
            return location, solar_record, climate_record
