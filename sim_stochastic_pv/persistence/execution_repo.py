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

    def list_run_results(
        self,
        limit: int = 50,
        offset: int = 0,
        *,
        scenario_name: str | None = None,
        location: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        include_archived: bool = False,
    ) -> list[RunResultRecord]:
        """
        Fetch run results with optional filtering and pagination.

        Phase 12 — extended to support the Dashboard filters
        (name / location / date range) and the archive toggle. All
        filter arguments are optional and combine with logical AND.

        Args:
            limit: Maximum number of records to return.
            offset: Number of leading records to skip (for pagination).
            scenario_name: Case-insensitive substring match on
                ``summary['scenario']``.
            location: Case-insensitive exact match on
                ``summary['location_name']``.
            date_from: ISO date or datetime string; only rows created
                on or after this instant are returned.
            date_to: ISO date or datetime string; only rows created
                on or before this instant are returned.
            include_archived: When False (default), rows with a
                non-null ``archived_at`` are excluded.

        Returns:
            List of matching RunResultRecord, newest first.
        """
        from datetime import datetime

        with self._session_factory() as session:
            stmt = select(RunResultRecord)

            if not include_archived:
                stmt = stmt.where(RunResultRecord.archived_at.is_(None))

            if scenario_name:
                pattern = f"%{scenario_name}%"
                # SQLAlchemy JSON access works on both SQLite and Postgres.
                stmt = stmt.where(
                    RunResultRecord.summary["scenario"]
                    .as_string()
                    .ilike(pattern)
                )
            if location:
                stmt = stmt.where(
                    RunResultRecord.summary["location_name"]
                    .as_string()
                    == location
                )
            if date_from:
                try:
                    dt_from = datetime.fromisoformat(date_from)
                    stmt = stmt.where(RunResultRecord.created_at >= dt_from)
                except ValueError:
                    pass
            if date_to:
                try:
                    dt_to = datetime.fromisoformat(date_to)
                    stmt = stmt.where(RunResultRecord.created_at <= dt_to)
                except ValueError:
                    pass

            stmt = (
                stmt.order_by(desc(RunResultRecord.created_at))
                .offset(offset)
                .limit(limit)
            )
            result = session.execute(stmt).scalars().all()
            return list(result)

    def delete_run_result(self, run_id: int) -> bool:
        """Hard-delete a run by primary key. Returns True if a row was removed."""
        with self._session_factory() as session:
            record = session.get(RunResultRecord, run_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True

    def set_run_archived(
        self, run_id: int, archived: bool
    ) -> RunResultRecord | None:
        """Toggle the ``archived_at`` flag for a run. Returns the updated row."""
        from datetime import datetime, timezone

        with self._session_factory() as session:
            record = session.get(RunResultRecord, run_id)
            if record is None:
                return None
            record.archived_at = datetime.now(timezone.utc) if archived else None
            session.commit()
            session.refresh(record)
            return record

    def list_distinct_locations(self) -> list[str]:
        """Return distinct non-null ``summary.location_name`` values across runs."""
        with self._session_factory() as session:
            # Pull all rows and project in Python — JSON path uniqueness
            # is dialect-specific. Cheap because run_results is small.
            rows = session.execute(select(RunResultRecord.summary)).scalars().all()
            seen = set()
            for s in rows:
                if isinstance(s, dict):
                    loc = s.get("location_name")
                    if loc:
                        seen.add(loc)
            return sorted(seen)

    def get_run_result(self, run_id: int) -> RunResultRecord | None:
        """
        Fetch a single ``RunResultRecord`` by primary key (Phase 11).

        Used by the Excel and PDF export endpoints, which need the full
        ``summary`` JSON without loading the entire run history.

        Args:
            run_id: Primary key of the run.

        Returns:
            The record or None if no row matches the given id.
        """
        with self._session_factory() as session:
            return session.get(RunResultRecord, run_id)

    def list_scenarios(self) -> list[ScenarioRecord]:
        """List all saved scenarios (history)."""
        with self._session_factory() as session:
            stmt = select(ScenarioRecord).order_by(desc(ScenarioRecord.created_at))
            return list(session.execute(stmt).scalars().all())
