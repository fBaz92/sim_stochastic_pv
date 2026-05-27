"""
Hardware repository for CRUD operations on inverters, panels, and batteries.
"""

from __future__ import annotations

from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import BatteryModel, InverterModel, PanelModel
from ._utils import asdict_safe


# ── helpers ──────────────────────────────────────────────────────────────────

def _delete_by_id(session_factory, model_class, record_id: int) -> bool:
    """
    Delete a single record of *model_class* by primary-key *record_id*.

    Args:
        session_factory: SQLAlchemy session factory.
        model_class: The ORM model class to delete from.
        record_id: Primary-key value of the record to delete.

    Returns:
        True if a record was found and deleted, False if not found.
    """
    with session_factory() as session:
        record = session.get(model_class, record_id)
        if record is None:
            return False
        session.delete(record)
        session.commit()
        return True


class HardwareRepository:
    """
    Repository for hardware component persistence (inverters, panels, batteries).

    Handles upsert operations based on component name as unique key.
    """

    def __init__(self, session_factory):
        """
        Initialize hardware repository with session factory.

        Args:
            session_factory: SQLAlchemy session factory for creating database connections.
        """
        self._session_factory = session_factory

    def upsert_inverter(self, inverter_data: Any) -> InverterModel | None:
        """
        Insert or update an inverter record based on its name.

        Args:
            inverter_data: Dataclass or mapping describing the inverter.

        Returns:
            Persisted InverterModel or None if data is missing.
        """
        if inverter_data is None:
            return None
        payload = asdict_safe(inverter_data)
        with self._session_factory() as session:
            stmt = select(InverterModel).where(InverterModel.name == payload.get("name"))
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                record = InverterModel(
                    name=payload.get("name"),
                    manufacturer=payload.get("manufacturer"),
                    model_number=payload.get("model_number"),
                    nominal_power_kw=payload.get("p_ac_max_kw"),
                    datasheet=payload.get("datasheet"),
                    specs=payload,
                )
                session.add(record)
            else:
                record.manufacturer = payload.get("manufacturer")
                record.model_number = payload.get("model_number")
                record.datasheet = payload.get("datasheet")
                record.specs = payload
                record.nominal_power_kw = payload.get("p_ac_max_kw")
            session.flush()
            session.commit()
            return record

    def delete_inverter(self, inverter_id: int) -> bool:
        """
        Delete an inverter record by primary key.

        Args:
            inverter_id: Primary-key ID of the inverter to delete.

        Returns:
            True if the record was found and deleted, False if not found.
        """
        return _delete_by_id(self._session_factory, InverterModel, inverter_id)

    def upsert_panel(self, panel_data: Any) -> PanelModel | None:
        """
        Insert or update a panel record.

        Args:
            panel_data: Dataclass or mapping describing the panel.

        Returns:
            Stored PanelModel or None.
        """
        if panel_data is None:
            return None
        payload = asdict_safe(panel_data)
        with self._session_factory() as session:
            stmt = select(PanelModel).where(PanelModel.name == payload.get("name"))
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                record = PanelModel(
                    name=payload.get("name"),
                    manufacturer=payload.get("manufacturer"),
                    model_number=payload.get("model_number"),
                    power_w=payload.get("power_w"),
                    datasheet=payload.get("datasheet"),
                    specs=payload,
                )
                session.add(record)
            else:
                record.manufacturer = payload.get("manufacturer")
                record.model_number = payload.get("model_number")
                record.datasheet = payload.get("datasheet")
                record.specs = payload
                record.power_w = payload.get("power_w")
            session.flush()
            session.commit()
            return record

    def delete_panel(self, panel_id: int) -> bool:
        """
        Delete a panel record by primary key.

        Args:
            panel_id: Primary-key ID of the panel to delete.

        Returns:
            True if the record was found and deleted, False if not found.
        """
        return _delete_by_id(self._session_factory, PanelModel, panel_id)

    def upsert_battery(self, battery_data: Any) -> BatteryModel | None:
        """
        Insert or update a battery record.

        Args:
            battery_data: Mapping containing at least the battery name.

        Returns:
            Stored BatteryModel or None.
        """
        if battery_data is None:
            return None
        payload = asdict_safe(battery_data)
        name = payload.get("name")
        specs = payload.get("specs") or payload
        with self._session_factory() as session:
            stmt = select(BatteryModel).where(BatteryModel.name == name)
            record = session.execute(stmt).scalar_one_or_none()
            capacity = specs.get("capacity_kwh") if isinstance(specs, Mapping) else None
            if record is None:
                record = BatteryModel(
                    name=name,
                    manufacturer=payload.get("manufacturer"),
                    model_number=payload.get("model_number"),
                    capacity_kwh=capacity,
                    datasheet=payload.get("datasheet"),
                    specs=payload,
                )
                session.add(record)
            else:
                record.manufacturer = payload.get("manufacturer")
                record.model_number = payload.get("model_number")
                record.capacity_kwh = capacity
                record.datasheet = payload.get("datasheet")
                record.specs = payload
            session.flush()
            session.commit()
            return record

    def delete_battery(self, battery_id: int) -> bool:
        """
        Delete a battery record by primary key.

        Args:
            battery_id: Primary-key ID of the battery to delete.

        Returns:
            True if the record was found and deleted, False if not found.
        """
        return _delete_by_id(self._session_factory, BatteryModel, battery_id)

    def list_inverters(self) -> list[InverterModel]:
        """List all available inverters."""
        with self._session_factory() as session:
            stmt = select(InverterModel).order_by(InverterModel.name)
            return list(session.execute(stmt).scalars().all())

    def list_panels(self) -> list[PanelModel]:
        """List all available panels."""
        with self._session_factory() as session:
            stmt = select(PanelModel).order_by(PanelModel.name)
            return list(session.execute(stmt).scalars().all())

    def list_batteries(self) -> list[BatteryModel]:
        """List all available batteries."""
        with self._session_factory() as session:
            stmt = select(BatteryModel).order_by(BatteryModel.name)
            return list(session.execute(stmt).scalars().all())
