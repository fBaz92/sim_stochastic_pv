"""
Configuration repository for CRUD operations on saved configurations, load profiles, and price profiles.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import LoadProfileModel, PriceProfileModel, SavedConfigurationModel


def _delete_by_id(session_factory, model_class, record_id: int) -> bool:
    """Delete a record by primary key. Returns True if deleted, False if not found."""
    with session_factory() as session:
        record = session.get(model_class, record_id)
        if record is None:
            return False
        session.delete(record)
        session.commit()
        return True


class ConfigurationRepository:
    """
    Repository for configuration persistence (scenarios, optimizations, profiles).

    Handles saved configurations, load profiles, and price profiles.
    """

    def __init__(self, session_factory):
        """
        Initialize configuration repository with session factory.

        Args:
            session_factory: SQLAlchemy session factory for creating database connections.
        """
        self._session_factory = session_factory

    def upsert_load_profile(self, name: str, profile_type: str, data: dict) -> LoadProfileModel:
        """
        Insert or update a load profile.

        Args:
            name: Profile name (unique key).
            profile_type: Type of load profile.
            data: Profile configuration data.

        Returns:
            Persisted LoadProfileModel.
        """
        with self._session_factory() as session:
            stmt = select(LoadProfileModel).where(LoadProfileModel.name == name)
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                record = LoadProfileModel(name=name, profile_type=profile_type, data=data)
                session.add(record)
            else:
                record.profile_type = profile_type
                record.data = data
            session.flush()
            session.commit()
            return record

    def list_load_profiles(self) -> list[LoadProfileModel]:
        """List all saved load profiles."""
        with self._session_factory() as session:
            stmt = select(LoadProfileModel).order_by(LoadProfileModel.name)
            return list(session.execute(stmt).scalars().all())

    def get_load_profile_by_id(self, profile_id: int) -> LoadProfileModel | None:
        """
        Fetch a single load profile by primary key.

        Args:
            profile_id: Primary-key ID of the load profile.

        Returns:
            The :class:`LoadProfileModel` record, or ``None`` if not found.
        """
        with self._session_factory() as session:
            return session.get(LoadProfileModel, profile_id)

    def delete_load_profile(self, profile_id: int) -> bool:
        """
        Delete a load profile by primary key.

        Args:
            profile_id: Primary-key ID of the load profile to delete.

        Returns:
            True if the record was found and deleted, False if not found.
        """
        return _delete_by_id(self._session_factory, LoadProfileModel, profile_id)

    def update_load_profile(
        self, profile_id: int, name: str, profile_type: str, data: dict
    ) -> LoadProfileModel | None:
        """
        Update an existing load profile by primary key (allows rename).

        Args:
            profile_id: Primary-key ID of the profile to update.
            name: New profile name (may differ from existing).
            profile_type: Profile type identifier.
            data: New profile configuration payload.

        Returns:
            Updated :class:`LoadProfileModel`, or ``None`` if the ID does
            not exist.

        Raises:
            ValueError: If the new ``name`` is already used by a different
                record (uniqueness violation).
        """
        with self._session_factory() as session:
            record = session.get(LoadProfileModel, profile_id)
            if record is None:
                return None
            if name and name != record.name:
                clash = session.execute(
                    select(LoadProfileModel).where(LoadProfileModel.name == name)
                ).scalar_one_or_none()
                if clash is not None and clash.id != profile_id:
                    raise ValueError(
                        f"Load profile name '{name}' is already used by id={clash.id}"
                    )
                record.name = name
            record.profile_type = profile_type
            record.data = data
            session.flush()
            session.commit()
            session.refresh(record)
            return record

    def upsert_price_profile(self, name: str, data: dict) -> PriceProfileModel:
        """
        Insert or update a price profile.

        Args:
            name: Profile name (unique key).
            data: Price profile configuration data.

        Returns:
            Persisted PriceProfileModel.
        """
        with self._session_factory() as session:
            stmt = select(PriceProfileModel).where(PriceProfileModel.name == name)
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                record = PriceProfileModel(name=name, data=data)
                session.add(record)
            else:
                record.data = data
            session.flush()
            session.commit()
            return record

    def list_price_profiles(self) -> list[PriceProfileModel]:
        """List all saved price profiles."""
        with self._session_factory() as session:
            stmt = select(PriceProfileModel).order_by(PriceProfileModel.name)
            return list(session.execute(stmt).scalars().all())

    def delete_price_profile(self, profile_id: int) -> bool:
        """
        Delete a price profile by primary key.

        Args:
            profile_id: Primary-key ID of the price profile to delete.

        Returns:
            True if the record was found and deleted, False if not found.
        """
        return _delete_by_id(self._session_factory, PriceProfileModel, profile_id)

    def update_price_profile(
        self, profile_id: int, name: str, data: dict
    ) -> PriceProfileModel | None:
        """
        Update an existing price profile by primary key (allows rename).

        Args:
            profile_id: Primary-key ID of the profile to update.
            name: New profile name.
            data: New price model configuration payload.

        Returns:
            Updated :class:`PriceProfileModel`, or ``None`` if not found.

        Raises:
            ValueError: If the new ``name`` clashes with another record.
        """
        with self._session_factory() as session:
            record = session.get(PriceProfileModel, profile_id)
            if record is None:
                return None
            if name and name != record.name:
                clash = session.execute(
                    select(PriceProfileModel).where(PriceProfileModel.name == name)
                ).scalar_one_or_none()
                if clash is not None and clash.id != profile_id:
                    raise ValueError(
                        f"Price profile name '{name}' is already used by id={clash.id}"
                    )
                record.name = name
            record.data = data
            session.flush()
            session.commit()
            session.refresh(record)
            return record

    def save_configuration(self, name: str, config_type: str, data: dict) -> SavedConfigurationModel:
        """
        Save or update a configuration (scenario or optimization).

        Args:
            name: Configuration name (unique key).
            config_type: Type of configuration ("scenario" or "optimization").
            data: Configuration data.

        Returns:
            Persisted SavedConfigurationModel.
        """
        with self._session_factory() as session:
            stmt = select(SavedConfigurationModel).where(SavedConfigurationModel.name == name)
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                record = SavedConfigurationModel(name=name, config_type=config_type, data=data)
                session.add(record)
            else:
                record.config_type = config_type
                record.data = data
            session.flush()
            session.commit()
            return record

    def list_configurations(self, config_type: str | None = None) -> list[SavedConfigurationModel]:
        """
        List all saved configurations, optionally filtered by type.

        Args:
            config_type: Optional filter for configuration type ("scenario" or "optimization").

        Returns:
            List of SavedConfigurationModel instances.
        """
        with self._session_factory() as session:
            stmt = select(SavedConfigurationModel)
            if config_type:
                stmt = stmt.where(SavedConfigurationModel.config_type == config_type)
            stmt = stmt.order_by(SavedConfigurationModel.name)
            return list(session.execute(stmt).scalars().all())

    def get_configuration_by_id(self, config_id: int) -> SavedConfigurationModel | None:
        """
        Retrieve a saved configuration by ID.

        Args:
            config_id: The ID of the configuration to retrieve.

        Returns:
            The configuration record or None if not found.
        """
        with self._session_factory() as session:
            stmt = select(SavedConfigurationModel).where(SavedConfigurationModel.id == config_id)
            return session.execute(stmt).scalar_one_or_none()

    def delete_configuration(self, config_id: int) -> bool:
        """
        Delete a saved configuration (scenario or campaign) by primary key.

        Args:
            config_id: Primary-key ID of the configuration to delete.

        Returns:
            True if the record was found and deleted, False if not found.
        """
        return _delete_by_id(self._session_factory, SavedConfigurationModel, config_id)

    def update_configuration(
        self, config_id: int, name: str, config_type: str, data: dict
    ) -> SavedConfigurationModel | None:
        """
        Update an existing saved configuration by primary key (allows rename).

        Args:
            config_id: Primary-key ID of the configuration to update.
            name: New configuration name.
            config_type: Configuration type identifier ("scenario" or
                "optimization").
            data: New configuration payload.

        Returns:
            Updated :class:`SavedConfigurationModel`, or ``None`` if not
            found.

        Raises:
            ValueError: If the new ``name`` clashes with another record.
        """
        with self._session_factory() as session:
            record = session.get(SavedConfigurationModel, config_id)
            if record is None:
                return None
            if name and name != record.name:
                clash = session.execute(
                    select(SavedConfigurationModel).where(
                        SavedConfigurationModel.name == name
                    )
                ).scalar_one_or_none()
                if clash is not None and clash.id != config_id:
                    raise ValueError(
                        f"Configuration name '{name}' is already used by id={clash.id}"
                    )
                record.name = name
            record.config_type = config_type
            record.data = data
            session.flush()
            session.commit()
            session.refresh(record)
            return record

    def get_configuration_by_name(self, name: str) -> SavedConfigurationModel | None:
        """
        Retrieve a saved configuration by its unique name.

        Args:
            name: Configuration name (case-sensitive).

        Returns:
            SavedConfigurationModel or None if not found.
        """
        with self._session_factory() as session:
            stmt = select(SavedConfigurationModel).where(SavedConfigurationModel.name == name)
            return session.execute(stmt).scalar_one_or_none()
