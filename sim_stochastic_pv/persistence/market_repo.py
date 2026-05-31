"""
Repository for :class:`MarketProfileModel` — reusable electricity-market
profiles backing the dedicated-withdrawal (*ritiro dedicato*) valuation.

The serialisation contract lives with the runtime object
(:meth:`MarketPriceProvider.to_config_dict` /
:meth:`MarketPriceProvider.from_config_dict`); this module is the thin DB
boundary that stores those dicts in the ``market_profiles`` table and hydrates
them back into a runtime :class:`MarketPriceProvider`. It mirrors the shape of
:class:`ClimateProfileRepository` so the :class:`PersistenceService` facade can
expose the two symmetrically.
"""

from __future__ import annotations

from typing import Any

from ..db.models import MarketProfileModel
from ..simulation.market_pricing import MarketPriceProvider


class MarketProfileRepository:
    """
    CRUD operations for :class:`MarketProfileModel` plus a convenience that
    hydrates a runtime :class:`MarketPriceProvider`.

    All methods open a short-lived session from the injected factory and never
    leak ORM instances bound to a closed session beyond what the existing
    repositories already do (callers read attributes immediately or use the
    ``load_market_provider`` helper).
    """

    def __init__(self, session_factory):
        """Args:
            session_factory: SQLAlchemy ``sessionmaker``.
        """
        self._session_factory = session_factory

    # ------------------------------------------------------------------- CRUD

    def upsert_market_profile(self, data: dict[str, Any]) -> MarketProfileModel:
        """
        Insert or update a market profile by ``name``.

        Args:
            data: dict with key ``"name"`` and ``"data"`` (the provider config
                dict from :meth:`MarketPriceProvider.to_config_dict`), plus the
                optional ``"description"``.

        Returns:
            The created / updated :class:`MarketProfileModel`.
        """
        with self._session_factory() as session:
            existing = (
                session.query(MarketProfileModel)
                .filter_by(name=data["name"])
                .first()
            )
            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
                record = existing
            else:
                record = MarketProfileModel(**data)
                session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def list_market_profiles(self) -> list[MarketProfileModel]:
        """Return all market profiles ordered by name."""
        with self._session_factory() as session:
            return (
                session.query(MarketProfileModel)
                .order_by(MarketProfileModel.name)
                .all()
            )

    def get_market_profile_by_id(self, profile_id: int) -> MarketProfileModel | None:
        """Fetch one market profile by primary key, or ``None``."""
        with self._session_factory() as session:
            return session.get(MarketProfileModel, profile_id)

    def get_market_profile_by_name(self, name: str) -> MarketProfileModel | None:
        """Fetch one market profile by unique name, or ``None``."""
        with self._session_factory() as session:
            return (
                session.query(MarketProfileModel)
                .filter_by(name=name)
                .first()
            )

    def update_market_profile(
        self, profile_id: int, data: dict[str, Any]
    ) -> MarketProfileModel | None:
        """
        Update an existing market profile by primary key (allows rename).

        Only the keys present in ``data`` are written. Returns ``None`` when no
        such ID exists.

        Args:
            profile_id: Primary key of the profile to update.
            data: Mapping of new column values (``name``, ``description``,
                ``data``).

        Returns:
            Updated :class:`MarketProfileModel`, or ``None`` if not found.

        Raises:
            ValueError: If the new ``name`` clashes with another record.
        """
        if not data:
            return self.get_market_profile_by_id(profile_id)
        with self._session_factory() as session:
            record = session.get(MarketProfileModel, profile_id)
            if record is None:
                return None
            new_name = data.get("name")
            if new_name and new_name != record.name:
                clash = (
                    session.query(MarketProfileModel)
                    .filter(MarketProfileModel.name == new_name)
                    .first()
                )
                if clash is not None and clash.id != profile_id:
                    raise ValueError(
                        f"Market profile name '{new_name}' is already used by "
                        f"id={clash.id}"
                    )
            for key, value in data.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            session.commit()
            session.refresh(record)
            return record

    def delete_market_profile(self, profile_id: int) -> bool:
        """Delete by ID. Returns ``True`` if a row was removed, else ``False``."""
        with self._session_factory() as session:
            record = session.get(MarketProfileModel, profile_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True

    # ----------------------------------------------------------------- helper

    def load_market_provider(self, profile_id: int) -> MarketPriceProvider | None:
        """
        Fetch by id and hydrate a runtime :class:`MarketPriceProvider`.

        Args:
            profile_id: Primary key of the market profile.

        Returns:
            A :class:`MarketPriceProvider` reconstructed from the stored
            configuration, or ``None`` if the profile does not exist.

        Raises:
            KeyError / ValueError: Propagated from
                :meth:`MarketPriceProvider.from_config_dict` when the stored
                ``data`` blob is malformed.
        """
        record = self.get_market_profile_by_id(profile_id)
        if record is None:
            return None
        return MarketPriceProvider.from_config_dict(record.data)
