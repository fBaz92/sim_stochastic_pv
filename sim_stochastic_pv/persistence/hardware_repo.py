"""
Hardware repository for CRUD operations on inverters, panels, and batteries.
"""

from __future__ import annotations

from typing import Any, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import (
    BatteryModel,
    CableModel,
    InverterModel,
    PanelModel,
    ProtectionModel,
)
from ._utils import asdict_safe


# Columns writable through the cable/protection CRUD (everything except
# the primary key and the timestamp mixin).
_CABLE_FIELDS = (
    "name", "manufacturer", "section_mm2", "material",
    "price_eur_per_m", "iz_a", "notes",
)
_PROTECTION_FIELDS = (
    "name", "manufacturer", "kind", "rated_current_a",
    "rated_voltage_v", "price_eur", "specs", "notes",
)


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

    def update_inverter(self, inverter_id: int, inverter_data: Any) -> InverterModel | None:
        """
        Update an existing inverter record by primary key.

        Differs from :meth:`upsert_inverter` in that the lookup is by ID
        rather than name, so the *name* itself can be edited (rename).

        Args:
            inverter_id: Primary-key ID of the inverter to update.
            inverter_data: Dataclass or mapping with the new field values.
                Must include ``name`` (string) and ``p_ac_max_kw`` (float).

        Returns:
            The updated :class:`InverterModel`, or ``None`` if the ID does
            not exist.

        Raises:
            ValueError: If the requested ``name`` is already used by a
                *different* inverter record (uniqueness violation).
        """
        if inverter_data is None:
            return None
        payload = asdict_safe(inverter_data)
        with self._session_factory() as session:
            record = session.get(InverterModel, inverter_id)
            if record is None:
                return None
            new_name = payload.get("name")
            if new_name and new_name != record.name:
                clash = session.execute(
                    select(InverterModel).where(InverterModel.name == new_name)
                ).scalar_one_or_none()
                if clash is not None and clash.id != inverter_id:
                    raise ValueError(
                        f"Inverter name '{new_name}' is already used by id={clash.id}"
                    )
                record.name = new_name
            record.manufacturer = payload.get("manufacturer")
            record.model_number = payload.get("model_number")
            record.datasheet = payload.get("datasheet")
            record.specs = payload
            record.nominal_power_kw = payload.get("p_ac_max_kw")
            session.flush()
            session.commit()
            session.refresh(record)
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

    def update_panel(self, panel_id: int, panel_data: Any) -> PanelModel | None:
        """
        Update an existing panel record by primary key (allows rename).

        Args:
            panel_id: Primary-key ID of the panel to update.
            panel_data: Dataclass or mapping with new field values.

        Returns:
            Updated :class:`PanelModel`, or ``None`` if not found.

        Raises:
            ValueError: If ``name`` clashes with another record.
        """
        if panel_data is None:
            return None
        payload = asdict_safe(panel_data)
        with self._session_factory() as session:
            record = session.get(PanelModel, panel_id)
            if record is None:
                return None
            new_name = payload.get("name")
            if new_name and new_name != record.name:
                clash = session.execute(
                    select(PanelModel).where(PanelModel.name == new_name)
                ).scalar_one_or_none()
                if clash is not None and clash.id != panel_id:
                    raise ValueError(
                        f"Panel name '{new_name}' is already used by id={clash.id}"
                    )
                record.name = new_name
            record.manufacturer = payload.get("manufacturer")
            record.model_number = payload.get("model_number")
            record.datasheet = payload.get("datasheet")
            record.specs = payload
            record.power_w = payload.get("power_w")
            session.flush()
            session.commit()
            session.refresh(record)
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

    def update_battery(self, battery_id: int, battery_data: Any) -> BatteryModel | None:
        """
        Update an existing battery record by primary key (allows rename).

        Args:
            battery_id: Primary-key ID of the battery to update.
            battery_data: Mapping with new field values.

        Returns:
            Updated :class:`BatteryModel`, or ``None`` if not found.

        Raises:
            ValueError: If ``name`` clashes with another record.
        """
        if battery_data is None:
            return None
        payload = asdict_safe(battery_data)
        name = payload.get("name")
        specs = payload.get("specs") or payload
        with self._session_factory() as session:
            record = session.get(BatteryModel, battery_id)
            if record is None:
                return None
            if name and name != record.name:
                clash = session.execute(
                    select(BatteryModel).where(BatteryModel.name == name)
                ).scalar_one_or_none()
                if clash is not None and clash.id != battery_id:
                    raise ValueError(
                        f"Battery name '{name}' is already used by id={clash.id}"
                    )
                record.name = name
            capacity = specs.get("capacity_kwh") if isinstance(specs, Mapping) else None
            record.manufacturer = payload.get("manufacturer")
            record.model_number = payload.get("model_number")
            record.capacity_kwh = capacity
            record.datasheet = payload.get("datasheet")
            record.specs = payload
            session.flush()
            session.commit()
            session.refresh(record)
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

    # ── DC cables (electrical-designer catalogue) ─────────────────────────

    def upsert_cable(self, cable_data: Mapping[str, Any]) -> CableModel:
        """
        Insert or update a cable catalogue row by unique ``name``.

        Args:
            cable_data: Mapping with :data:`_CABLE_FIELDS` keys. Must
                include ``name`` and ``section_mm2``.

        Returns:
            The persisted :class:`CableModel` (refreshed, detached).
        """
        payload = {k: cable_data.get(k) for k in _CABLE_FIELDS if k in cable_data}
        with self._session_factory() as session:
            record = session.execute(
                select(CableModel).where(CableModel.name == payload.get("name"))
            ).scalar_one_or_none()
            if record is None:
                record = CableModel(**payload)
                session.add(record)
            else:
                for key, value in payload.items():
                    setattr(record, key, value)
            session.commit()
            session.refresh(record)
            return record

    def list_cables(self) -> list[CableModel]:
        """List all cables ordered by cross-section, then name."""
        with self._session_factory() as session:
            stmt = select(CableModel).order_by(CableModel.section_mm2, CableModel.name)
            return list(session.execute(stmt).scalars().all())

    def update_cable(
        self, cable_id: int, cable_data: Mapping[str, Any]
    ) -> CableModel | None:
        """
        Update a cable by primary key (partial — allows rename).

        Args:
            cable_id: Primary key of the cable to update.
            cable_data: New field values (subset of :data:`_CABLE_FIELDS`).

        Returns:
            Updated record, or ``None`` when the ID does not exist.

        Raises:
            ValueError: New ``name`` already used by another cable.
        """
        with self._session_factory() as session:
            record = session.get(CableModel, cable_id)
            if record is None:
                return None
            new_name = cable_data.get("name")
            if new_name and new_name != record.name:
                clash = session.execute(
                    select(CableModel).where(CableModel.name == new_name)
                ).scalar_one_or_none()
                if clash is not None and clash.id != cable_id:
                    raise ValueError(
                        f"Cable name '{new_name}' is already used by id={clash.id}"
                    )
            for key in _CABLE_FIELDS:
                if key in cable_data:
                    setattr(record, key, cable_data[key])
            session.commit()
            session.refresh(record)
            return record

    def delete_cable(self, cable_id: int) -> bool:
        """Delete a cable by ID. Returns True if deleted, False if missing."""
        return _delete_by_id(self._session_factory, CableModel, cable_id)

    # ── DC protections (electrical-designer catalogue) ────────────────────

    def upsert_protection(self, protection_data: Mapping[str, Any]) -> ProtectionModel:
        """
        Insert or update a protection catalogue row by unique ``name``.

        Args:
            protection_data: Mapping with :data:`_PROTECTION_FIELDS` keys.
                Must include ``name``.

        Returns:
            The persisted :class:`ProtectionModel`.
        """
        payload = {
            k: protection_data.get(k) for k in _PROTECTION_FIELDS if k in protection_data
        }
        with self._session_factory() as session:
            record = session.execute(
                select(ProtectionModel).where(
                    ProtectionModel.name == payload.get("name")
                )
            ).scalar_one_or_none()
            if record is None:
                record = ProtectionModel(**payload)
                session.add(record)
            else:
                for key, value in payload.items():
                    setattr(record, key, value)
            session.commit()
            session.refresh(record)
            return record

    def list_protections(self) -> list[ProtectionModel]:
        """List all protections ordered by kind, rated current, name."""
        with self._session_factory() as session:
            stmt = select(ProtectionModel).order_by(
                ProtectionModel.kind,
                ProtectionModel.rated_current_a,
                ProtectionModel.name,
            )
            return list(session.execute(stmt).scalars().all())

    def update_protection(
        self, protection_id: int, protection_data: Mapping[str, Any]
    ) -> ProtectionModel | None:
        """
        Update a protection by primary key (partial — allows rename).

        Args:
            protection_id: Primary key of the protection to update.
            protection_data: New field values (subset of
                :data:`_PROTECTION_FIELDS`).

        Returns:
            Updated record, or ``None`` when the ID does not exist.

        Raises:
            ValueError: New ``name`` already used by another protection.
        """
        with self._session_factory() as session:
            record = session.get(ProtectionModel, protection_id)
            if record is None:
                return None
            new_name = protection_data.get("name")
            if new_name and new_name != record.name:
                clash = session.execute(
                    select(ProtectionModel).where(ProtectionModel.name == new_name)
                ).scalar_one_or_none()
                if clash is not None and clash.id != protection_id:
                    raise ValueError(
                        f"Protection name '{new_name}' is already used by id={clash.id}"
                    )
            for key in _PROTECTION_FIELDS:
                if key in protection_data:
                    setattr(record, key, protection_data[key])
            session.commit()
            session.refresh(record)
            return record

    def delete_protection(self, protection_id: int) -> bool:
        """Delete a protection by ID. Returns True if deleted, False if missing."""
        return _delete_by_id(self._session_factory, ProtectionModel, protection_id)
