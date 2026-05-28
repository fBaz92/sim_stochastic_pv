"""
Solar profile repository for CRUD operations on solar irradiance data.

Provides database access layer for SolarProfileModel with methods for
creating, reading, updating, and deleting solar profiles by location.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session

from ..db.models import SolarProfileModel


class SolarProfileRepository:
    """
    Repository for solar profile CRUD operations.

    Provides database access methods for managing solar irradiance profiles,
    including upsert by name (unique key), retrieval, and deletion operations.

    Attributes:
        _session_factory: SQLAlchemy session factory for database connections.

    Example:
        ```python
        from sim_stochastic_pv.db.session import SessionLocal
        from sim_stochastic_pv.persistence.solar_repo import SolarProfileRepository

        repo = SolarProfileRepository(SessionLocal)

        # Upsert a solar profile
        profile_data = {
            "name": "Bologna",
            "location_name": "Bologna, Emilia-Romagna, Italy",
            "latitude": 44.49,
            "longitude": 11.34,
            "elevation_m": 54,
            "optimal_tilt_degrees": 35.0,
            "optimal_azimuth_degrees": 180.0,
            "avg_daily_kwh_per_kwp": [...],  # 12 monthly values
            "p_sunny": [...],  # 12 monthly values
            "sunny_factor": 1.2,
            "cloudy_factor": 0.3,
            "source": "PVGIS",
            "notes": "Custom profile"
        }
        profile = repo.upsert_solar_profile(profile_data)

        # Retrieve all profiles
        profiles = repo.list_solar_profiles()

        # Get by ID or name
        profile = repo.get_solar_profile_by_id(1)
        profile = repo.get_solar_profile_by_name("Bologna")

        # Delete
        deleted = repo.delete_solar_profile(1)
        ```
    """

    def __init__(self, session_factory):
        """
        Initialize repository with session factory.

        Args:
            session_factory: SQLAlchemy sessionmaker instance for creating
                database sessions.
        """
        self._session_factory = session_factory

    def upsert_solar_profile(self, data: Dict[str, Any]) -> SolarProfileModel:
        """
        Insert or update solar profile by name (unique key).

        If a profile with the given name exists, all fields are updated.
        Otherwise, a new profile is created. This enables idempotent
        configuration management.

        Args:
            data: Dictionary containing solar profile fields matching
                SolarProfileModel schema. Must include 'name' field.
                Required keys:
                - name: Unique identifier
                - location_name: Full location description
                - latitude: Latitude in degrees
                - longitude: Longitude in degrees
                - optimal_tilt_degrees: Recommended tilt
                - optimal_azimuth_degrees: Recommended azimuth
                - avg_daily_kwh_per_kwp: List of 12 monthly values
                - p_sunny: List of 12 monthly probabilities
                Optional keys:
                - elevation_m, sunny_factor, cloudy_factor, source, notes

        Returns:
            SolarProfileModel: The created or updated database record.

        Raises:
            KeyError: If required 'name' field is missing.
            ValueError: If data validation fails (e.g., wrong array lengths).

        Example:
            ```python
            data = {
                "name": "Torino",
                "location_name": "Torino, Piemonte, Italy",
                "latitude": 45.07,
                "longitude": 7.69,
                "optimal_tilt_degrees": 35.0,
                "optimal_azimuth_degrees": 180.0,
                "avg_daily_kwh_per_kwp": [1.4, 2.2, ..., 1.3],  # 12 values
                "p_sunny": [0.38, 0.42, ..., 0.35]  # 12 values
            }
            profile = repo.upsert_solar_profile(data)
            print(f"Saved profile: {profile.name} (ID: {profile.id})")
            ```

        Notes:
            - Commits immediately (not deferred)
            - Refreshes record to populate computed fields
            - Safe to call multiple times (idempotent)
        """
        with self._session_factory() as session:
            existing = session.query(SolarProfileModel).filter_by(
                name=data["name"]
            ).first()

            if existing:
                # Update existing record
                for key, value in data.items():
                    setattr(existing, key, value)
                record = existing
            else:
                # Insert new record
                record = SolarProfileModel(**data)
                session.add(record)

            session.commit()
            session.refresh(record)
            return record

    def list_solar_profiles(self) -> List[SolarProfileModel]:
        """
        List all solar profiles ordered by name.

        Retrieves all solar profiles from the database, sorted alphabetically
        by name for consistent display in CLI and API responses.

        Returns:
            List[SolarProfileModel]: List of all solar profile records.
                Empty list if no profiles exist.

        Example:
            ```python
            profiles = repo.list_solar_profiles()
            for profile in profiles:
                annual_kwh = sum(profile.avg_daily_kwh_per_kwp) * 30
                print(f"{profile.name}: {annual_kwh:.0f} kWh/kWp/year")
            ```

        Notes:
            - Returns detached objects (safe after session closes)
            - Includes all profile fields (metadata + monthly data)
            - Sorted by name for UI consistency
        """
        with self._session_factory() as session:
            return session.query(SolarProfileModel).order_by(
                SolarProfileModel.name
            ).all()

    def get_solar_profile_by_id(self, profile_id: int) -> Optional[SolarProfileModel]:
        """
        Get solar profile by primary key ID.

        Retrieves a single solar profile by its database ID. Preferred
        method for foreign key lookups and direct record access.

        Args:
            profile_id: Primary key ID of the solar profile.

        Returns:
            Optional[SolarProfileModel]: The matching profile record,
                or None if no profile exists with that ID.

        Example:
            ```python
            profile = repo.get_solar_profile_by_id(1)
            if profile:
                print(f"Found: {profile.name} at {profile.latitude}°N")
            else:
                print("Profile not found")
            ```

        Notes:
            - Returns detached object (safe after session closes)
            - None return indicates not found (not an error)
            - Use for foreign key resolution in scenarios
        """
        with self._session_factory() as session:
            return session.get(SolarProfileModel, profile_id)

    def get_solar_profile_by_name(self, name: str) -> Optional[SolarProfileModel]:
        """
        Get solar profile by unique name.

        Retrieves a single solar profile by its unique name identifier.
        Preferred method for user-facing lookups and configuration files.

        Args:
            name: Unique name of the solar profile (e.g., "Pavullo_nel_Frignano").

        Returns:
            Optional[SolarProfileModel]: The matching profile record,
                or None if no profile exists with that name.

        Example:
            ```python
            profile = repo.get_solar_profile_by_name("Milano")
            if profile:
                print(f"Milano annual production: "
                      f"{sum(profile.avg_daily_kwh_per_kwp) * 30:.0f} kWh/kWp")
            else:
                print("Profile not found. Available profiles:")
                for p in repo.list_solar_profiles():
                    print(f"  - {p.name}")
            ```

        Notes:
            - Returns detached object (safe after session closes)
            - None return indicates not found (not an error)
            - Use for scenario configuration with friendly names
            - Case-sensitive match (exact name required)
        """
        with self._session_factory() as session:
            return session.query(SolarProfileModel).filter_by(name=name).first()

    def update_solar_profile(
        self, profile_id: int, data: Dict[str, Any]
    ) -> Optional[SolarProfileModel]:
        """
        Update an existing solar profile by primary key (allows rename).

        Unlike :meth:`upsert_solar_profile`, this method uses the ID as the
        lookup key, so the user-visible ``name`` column can be edited
        without colliding with the upsert-by-name behaviour. Only the keys
        present in ``data`` are written — other columns are left untouched.

        Args:
            profile_id: Primary key of the profile to update.
            data: Dictionary with the new field values. Allowed keys are
                any column on :class:`SolarProfileModel`.

        Returns:
            The updated :class:`SolarProfileModel`, or ``None`` if no
            record matches ``profile_id``.

        Raises:
            ValueError: If the requested ``name`` is already used by a
                different profile (uniqueness violation).

        Example:
            ```python
            updated = repo.update_solar_profile(
                12,
                {"name": "Pavullo_v2", "notes": "Renamed for clarity"},
            )
            ```

        Notes:
            - Commits immediately.
            - Returns the refreshed record so callers get DB-side defaults.
        """
        if not data:
            return self.get_solar_profile_by_id(profile_id)
        with self._session_factory() as session:
            record = session.get(SolarProfileModel, profile_id)
            if record is None:
                return None
            new_name = data.get("name")
            if new_name and new_name != record.name:
                clash = (
                    session.query(SolarProfileModel)
                    .filter(SolarProfileModel.name == new_name)
                    .first()
                )
                if clash is not None and clash.id != profile_id:
                    raise ValueError(
                        f"Solar profile name '{new_name}' is already used by id={clash.id}"
                    )
            for key, value in data.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            session.commit()
            session.refresh(record)
            return record

    def delete_solar_profile(self, profile_id: int) -> bool:
        """
        Delete solar profile by ID.

        Permanently removes a solar profile from the database. Use with
        caution as this operation cannot be undone. Scenarios referencing
        this profile may break if not updated.

        Args:
            profile_id: Primary key ID of the profile to delete.

        Returns:
            bool: True if a profile was deleted, False if no profile
                existed with that ID.

        Example:
            ```python
            # Delete profile
            deleted = repo.delete_solar_profile(5)
            if deleted:
                print("Profile deleted successfully")
            else:
                print("Profile not found")

            # Safe deletion with existence check
            profile = repo.get_solar_profile_by_id(5)
            if profile:
                repo.delete_solar_profile(5)
                print(f"Deleted: {profile.name}")
            ```

        Notes:
            - Commits immediately (not deferred)
            - No cascade delete to scenarios (foreign key not enforced)
            - Returns False for non-existent IDs (not an error)
            - Consider checking for references before deletion
        """
        with self._session_factory() as session:
            profile = session.get(SolarProfileModel, profile_id)
            if profile:
                session.delete(profile)
                session.commit()
                return True
            return False
