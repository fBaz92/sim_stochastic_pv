from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Mapping

from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from .db.models import (
    BatteryModel,
    InverterModel,
    OptimizationRecord,
    PanelModel,
    RunResultRecord,
    ScenarioRecord,
)
from .db.session import SessionLocal


def _asdict_safe(obj: Any) -> Dict[str, Any]:
    """
    Convert dataclasses or mappings to plain dictionaries for JSON storage.

    Args:
        obj: Dataclass instance, mapping, or None.

    Returns:
        Dictionary representation (empty dict if obj is None).

    Raises:
        TypeError: if the object type is unsupported.
    """
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    raise TypeError(f"Unsupported object type for serialization: {type(obj)!r}")


class PersistenceService:
    """
    Database helper to register components, scenarios, and run results.
    """

    def __init__(self, session_factory: type[Session] | None = None) -> None:
        """
        Args:
            session_factory: Optional SQLAlchemy session factory override.
        """
        self._session_factory = session_factory or SessionLocal

    @contextmanager
    def session(self) -> Iterable[Session]:
        """
        Context manager yielding a SQLAlchemy session with auto commit/rollback.
        """
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

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
        payload = _asdict_safe(inverter_data)
        with self.session() as session:
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
        payload = _asdict_safe(panel_data)
        with self.session() as session:
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
        payload = _asdict_safe(battery_data)
        name = payload.get("name")
        specs = payload.get("specs") or payload
        with self.session() as session:
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
            return record

    def record_scenario(
        self,
        name: str,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        inverter: InverterModel | None = None,
        panel: PanelModel | None = None,
        battery: BatteryModel | None = None,
    ) -> ScenarioRecord:
        """
        Persist a scenario definition for later re-use.

        Args:
            name: Scenario name.
            config: Serialized configuration data.
            metadata: Optional additional info.
            inverter: Foreign key reference.
            panel: Foreign key reference.
            battery: Foreign key reference.

        Returns:
            ScenarioRecord instance.
        """
        with self.session() as session:
            record = ScenarioRecord(
                name=name,
                config=dict(config),
                extra_metadata=dict(metadata or {}),
                inverter_id=inverter.id if inverter else None,
                panel_id=panel.id if panel else None,
                battery_id=battery.id if battery else None,
            )
            session.add(record)
            session.flush()
            return record

    def record_optimization(
        self,
        label: str,
        request_payload: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> OptimizationRecord:
        """
        Persist an optimization request/result summary.
        """
        with self.session() as session:
            record = OptimizationRecord(
                label=label,
                request=dict(request_payload),
                extra_metadata=dict(metadata or {}),
                status="completed",
            )
            session.add(record)
            session.flush()
            return record

    def record_run_result(
        self,
        result_type: str,
        summary: Mapping[str, Any],
        *,
        scenario: ScenarioRecord | None = None,
        optimization: OptimizationRecord | None = None,
        output_dir: str | None = None,
    ) -> RunResultRecord:
        """
        Store the outcome of an analysis or optimization run.

        Args:
            result_type: e.g. \"analysis\" or \"optimization\".
            summary: JSON-serializable metrics.
            scenario: Optional linked scenario.
            optimization: Optional linked optimization.
            output_dir: Filesystem path containing exported artifacts.
        """
        with self.session() as session:
            record = RunResultRecord(
                result_type=result_type,
                summary=dict(summary),
                scenario_id=scenario.id if scenario else None,
                optimization_id=optimization.id if optimization else None,
                output_dir=output_dir,
            )
            session.add(record)
            session.flush()
            return record

    def list_run_results(self, limit: int = 50) -> list[RunResultRecord]:
        """
        Fetch the latest run results ordered by creation date.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of RunResultRecord instances.
        """
        with self.session() as session:
            stmt = (
                select(RunResultRecord)
                .order_by(desc(RunResultRecord.created_at))
                .limit(limit)
            )
            result = session.execute(stmt).scalars().all()
            return list(result)
