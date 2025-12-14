"""
Hardware repository for CRUD operations on inverters, panels, and batteries.
"""

from __future__ import annotations

from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import BatteryModel, InverterModel, PanelModel
from ._utils import asdict_safe


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
                record.datasheet = payload.get("datasheet")
                record.specs = payload
                record.nominal_power_kw = payload.get("p_ac_max_kw")
            session.flush()
            session.commit()
            return record

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
                record.datasheet = payload.get("datasheet")
                record.specs = payload
                record.power_w = payload.get("power_w")
            session.flush()
            session.commit()
            return record

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
                record.capacity_kwh = capacity
                record.datasheet = payload.get("datasheet")
                record.specs = payload
            session.flush()
            session.commit()
            return record

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
