"""
Execution repository for CRUD operations on scenarios, optimizations, and run results.
"""

from __future__ import annotations

from typing import Any, Mapping

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ..db.models import OptimizationRecord, RunResultRecord, ScenarioRecord
from ..db.models import InverterModel, PanelModel, BatteryModel


class ExecutionRepository:
    """
    Repository for execution persistence (scenarios, optimizations, run results).

    Handles recording and retrieving simulation execution data.
    """

    def __init__(self, session_factory):
        """
        Initialize execution repository with session factory.

        Args:
            session_factory: SQLAlchemy session factory for creating database connections.
        """
        self._session_factory = session_factory

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
        with self._session_factory() as session:
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
            session.commit()
            return record

    def record_optimization(
        self,
        label: str,
        request_payload: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> OptimizationRecord:
        """
        Persist an optimization request/result summary.

        Args:
            label: Optimization label/name.
            request_payload: Optimization request data.
            metadata: Optional additional metadata.

        Returns:
            OptimizationRecord instance.
        """
        with self._session_factory() as session:
            record = OptimizationRecord(
                label=label,
                request=dict(request_payload),
                extra_metadata=dict(metadata or {}),
                status="completed",
            )
            session.add(record)
            session.flush()
            session.commit()
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
            result_type: e.g. "analysis" or "optimization".
            summary: JSON-serializable metrics.
            scenario: Optional linked scenario.
            optimization: Optional linked optimization.
            output_dir: Filesystem path containing exported artifacts.

        Returns:
            RunResultRecord instance.
        """
        with self._session_factory() as session:
            record = RunResultRecord(
                result_type=result_type,
                summary=dict(summary),
                scenario_id=scenario.id if scenario else None,
                optimization_id=optimization.id if optimization else None,
                output_dir=output_dir,
            )
            session.add(record)
            session.flush()
            session.commit()
            return record

    def list_run_results(self, limit: int = 50) -> list[RunResultRecord]:
        """
        Fetch the latest run results ordered by creation date.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of RunResultRecord instances.
        """
        with self._session_factory() as session:
            stmt = (
                select(RunResultRecord)
                .order_by(desc(RunResultRecord.created_at))
                .limit(limit)
            )
            result = session.execute(stmt).scalars().all()
            return list(result)

    def list_scenarios(self) -> list[ScenarioRecord]:
        """List all saved scenarios (history)."""
        with self._session_factory() as session:
            stmt = select(ScenarioRecord).order_by(desc(ScenarioRecord.created_at))
            return list(session.execute(stmt).scalars().all())
