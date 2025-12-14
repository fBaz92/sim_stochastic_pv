"""
Configuration repository for CRUD operations on saved configurations, load profiles, and price profiles.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import LoadProfileModel, PriceProfileModel, SavedConfigurationModel


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
