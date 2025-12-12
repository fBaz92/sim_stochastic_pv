"""
Database persistence layer for PV system configurations and results.

Provides CRUD operations for hardware components (inverters, panels, batteries),
simulation configurations, and analysis results. Handles serialization of complex
data structures to JSON for database storage.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, Mapping

from sqlalchemy import select, desc
from sqlalchemy.orm import Session

from .db.models import (
    BatteryModel,
    InverterModel,
    LoadProfileModel,
    OptimizationRecord,
    PanelModel,
    PriceProfileModel,
    RunResultRecord,
    SavedConfigurationModel,
    ScenarioRecord,
)
from .db.session import SessionLocal


def _asdict_safe(obj: Any) -> Dict[str, Any]:
    """
    Convert Python objects to plain dictionaries for JSON storage.

    Handles multiple object types commonly used in the application:
    - Dataclasses (Python standard library)
    - Pydantic models (v1 and v2)
    - Mappings (dict, OrderedDict, etc.)
    - None (returns empty dict)

    Args:
        obj: Object to convert. Can be:
            - Dataclass instance: Converted via asdict()
            - Pydantic model: Converted via model_dump() or dict()
            - Mapping: Converted via dict()
            - None: Returns {}

    Returns:
        Dictionary representation of the object, suitable for JSON serialization.
            Empty dict if obj is None.

    Raises:
        TypeError: If obj type is not supported (not dataclass, Pydantic, mapping, or None).

    Example:
        ```python
        from dataclasses import dataclass
        from pydantic import BaseModel

        @dataclass
        class Hardware:
            name: str
            power: float

        class Config(BaseModel):
            pv_kwp: float

        # Dataclass
        hw = Hardware(name="Inverter", power=5.0)
        d1 = _asdict_safe(hw)  # {"name": "Inverter", "power": 5.0}

        # Pydantic
        cfg = Config(pv_kwp=6.5)
        d2 = _asdict_safe(cfg)  # {"pv_kwp": 6.5}

        # Dict
        d3 = _asdict_safe({"key": "value"})  # {"key": "value"}

        # None
        d4 = _asdict_safe(None)  # {}
        ```

    Notes:
        - Pydantic v2 uses model_dump(), v1 uses dict() - both are supported
        - Nested dataclasses/Pydantic models are recursively converted
        - Used internally for database serialization
    """
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Mapping):
        return dict(obj)
    # Handle Pydantic models (v2 style)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # Handle Pydantic models (v1 style)
    if hasattr(obj, "dict"):
        return obj.dict()
    raise TypeError(f"Unsupported object type for serialization: {type(obj)!r}")


class PersistenceService:
    """
    Database persistence service for PV system components and results.

    Provides high-level CRUD operations for:
    - Hardware components (inverters, panels, batteries)
    - Load and price profiles
    - Simulation configurations
    - Scenario definitions
    - Optimization requests
    - Analysis results

    All database operations use transactional sessions with automatic
    commit/rollback handling. The service supports upsert operations
    (insert or update) for hardware components based on unique identifiers.

    Attributes:
        _session_factory: SQLAlchemy session factory for creating database connections.

    Example:
        ```python
        from sim_stochastic_pv.persistence import PersistenceService

        # Create service
        service = PersistenceService()

        # Store hardware
        inverter = service.upsert_inverter({
            "name": "Fronius Primo 5.0",
            "p_ac_max_kw": 5.0,
            "price_eur": 1500.0,
            "manufacturer": "Fronius"
        })

        panel = service.upsert_panel({
            "name": "Longi 540W",
            "power_w": 540.0,
            "price_eur": 150.0
        })

        # List available hardware
        all_inverters = service.list_inverters()
        all_panels = service.list_panels()

        # Store scenario
        scenario = service.record_scenario(
            name="Residential 5kWp",
            config={"pv_kwp": 5.4, "n_batteries": 1},
            inverter=inverter,
            panel=panel
        )

        # Store analysis results
        result = service.record_run_result(
            result_type="analysis",
            summary={"final_gain_eur": 15000.0, "irr": 0.08},
            scenario=scenario
        )
        ```

    Notes:
        - Thread-safe: Each operation uses independent session
        - Transactional: Auto-commit on success, auto-rollback on error
        - Upsert operations use component name as unique key
        - JSON columns support complex nested structures
    """

    def __init__(self, session_factory: type[Session] | None = None) -> None:
        """
        Initialize persistence service with optional session factory.

        Args:
            session_factory: SQLAlchemy session factory class. If None,
                uses the default SessionLocal from db.session module.
                Useful for testing with alternative database configurations.

        Example:
            ```python
            # Default configuration
            service = PersistenceService()

            # Custom session factory (e.g., for testing)
            from sqlalchemy.orm import sessionmaker
            TestSession = sessionmaker(bind=test_engine)
            service_test = PersistenceService(session_factory=TestSession)
            ```
        """
        self._session_factory = session_factory or SessionLocal

    @contextmanager
    def session(self) -> Iterable[Session]:
        """
        Context manager providing transactional database session.

        Yields a SQLAlchemy session with automatic transaction management:
        - Commits on successful completion
        - Rolls back on exception
        - Always closes session in finally block

        Yields:
            Session: Active SQLAlchemy session for database operations.

        Raises:
            Exception: Any exception from database operations (after rollback).

        Example:
            ```python
            service = PersistenceService()

            # Manual session usage
            with service.session() as db:
                inverter = InverterModel(name="Test", nominal_power_kw=5.0)
                db.add(inverter)
                db.flush()
                print(f"Created inverter ID: {inverter.id}")
            # Auto-commit happens here

            # Error handling
            try:
                with service.session() as db:
                    # ... operations ...
                    raise ValueError("Something went wrong")
            except ValueError:
                pass  # Transaction was rolled back automatically
            ```

        Notes:
            - Session is thread-local (not shared across threads)
            - Used internally by all CRUD methods
            - Flush within context to get generated IDs
            - Commit happens automatically on context exit
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

    def list_inverters(self) -> list[InverterModel]:
        """List all available inverters."""
        with self.session() as session:
            stmt = select(InverterModel).order_by(InverterModel.name)
            return list(session.execute(stmt).scalars().all())

    def list_panels(self) -> list[PanelModel]:
        """List all available panels."""
        with self.session() as session:
            stmt = select(PanelModel).order_by(PanelModel.name)
            return list(session.execute(stmt).scalars().all())

    def list_batteries(self) -> list[BatteryModel]:
        """List all available batteries."""
        with self.session() as session:
            stmt = select(BatteryModel).order_by(BatteryModel.name)
            return list(session.execute(stmt).scalars().all())

    def list_scenarios(self) -> list[ScenarioRecord]:
        """List all saved scenarios (history)."""
        with self.session() as session:
            stmt = select(ScenarioRecord).order_by(desc(ScenarioRecord.created_at))
            return list(session.execute(stmt).scalars().all())

    # --- New CRUD methods ---
    from .db.models import LoadProfileModel, PriceProfileModel, SavedConfigurationModel

    def upsert_load_profile(self, name: str, profile_type: str, data: dict) -> LoadProfileModel:
        with self.session() as session:
            stmt = select(LoadProfileModel).where(LoadProfileModel.name == name)
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                record = LoadProfileModel(name=name, profile_type=profile_type, data=data)
                session.add(record)
            else:
                record.profile_type = profile_type
                record.data = data
            session.flush()
            return record

    def list_load_profiles(self) -> list[LoadProfileModel]:
        with self.session() as session:
            stmt = select(LoadProfileModel).order_by(LoadProfileModel.name)
            return list(session.execute(stmt).scalars().all())

    def upsert_price_profile(self, name: str, data: dict) -> PriceProfileModel:
        with self.session() as session:
            stmt = select(PriceProfileModel).where(PriceProfileModel.name == name)
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                record = PriceProfileModel(name=name, data=data)
                session.add(record)
            else:
                record.data = data
            session.flush()
            return record

    def list_price_profiles(self) -> list[PriceProfileModel]:
        with self.session() as session:
            stmt = select(PriceProfileModel).order_by(PriceProfileModel.name)
            return list(session.execute(stmt).scalars().all())

    def save_configuration(self, name: str, config_type: str, data: dict) -> SavedConfigurationModel:
        with self.session() as session:
            stmt = select(SavedConfigurationModel).where(SavedConfigurationModel.name == name)
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                record = SavedConfigurationModel(name=name, config_type=config_type, data=data)
                session.add(record)
            else:
                record.config_type = config_type
                record.data = data
            session.flush()
            return record

    def list_configurations(self, config_type: str | None = None) -> list[SavedConfigurationModel]:
        with self.session() as session:
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
        with self.session() as session:
            stmt = select(SavedConfigurationModel).where(SavedConfigurationModel.id == config_id)
            return session.execute(stmt).scalar_one_or_none()

    def hydrate_scenario_from_ids(
        self,
        scenario_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Hydrate a scenario configuration by replacing hardware/profile IDs with full specs.

        This function takes a scenario dict that contains IDs (inverter_id, panel_id, etc.)
        and fetches the corresponding records from the database to build a complete
        scenario configuration that can be consumed by run_analysis/run_optimization.

        For campaigns, it also handles hardware_selections with multiple IDs, expanding
        them into the optimization.inverter_options/panel_options/battery_options arrays.

        Args:
            scenario_data: Partial scenario dict with IDs instead of full specs.

        Returns:
            Complete scenario dict with all hardware and profile data hydrated.

        Raises:
            ValueError: If a referenced ID is not found in the database.
        """
        with self.session() as session:
            hydrated = dict(scenario_data)

            # Handle campaign hardware selections (multiple IDs for optimization)
            if "hardware_selections" in scenario_data:
                selections = scenario_data["hardware_selections"]

                # Hydrate inverter options for campaign
                if "inverter_ids" in selections and selections["inverter_ids"]:
                    inverter_ids = selections["inverter_ids"]
                    stmt = select(InverterModel).where(InverterModel.id.in_(inverter_ids))
                    inverter_records = list(session.execute(stmt).scalars().all())

                    inverter_options = []
                    for inv in inverter_records:
                        inv_dict = {
                            "id": inv.id,
                            "name": inv.name,
                            "p_ac_max_kw": inv.specs.get("p_ac_max_kw") if inv.specs else inv.nominal_power_kw,
                            "p_dc_max_kw": inv.specs.get("p_dc_max_kw") if inv.specs else None,
                            "price_eur": inv.specs.get("price_eur") if inv.specs else None,
                            "install_cost_eur": inv.specs.get("install_cost_eur") if inv.specs else None,
                        }
                        if inv.specs and "integrated_battery_specs" in inv.specs:
                            inv_dict["integrated_battery_specs"] = inv.specs["integrated_battery_specs"]
                            inv_dict["integrated_battery_price_eur"] = inv.specs.get("integrated_battery_price_eur")
                            inv_dict["integrated_battery_count_options"] = inv.specs.get("integrated_battery_count_options", [])
                        inverter_options.append(inv_dict)

                    if "optimization" not in hydrated:
                        hydrated["optimization"] = {}
                    hydrated["optimization"]["inverter_options"] = inverter_options

                # Hydrate panel options for campaign
                if "panel_ids" in selections and selections["panel_ids"]:
                    panel_ids = selections["panel_ids"]
                    stmt = select(PanelModel).where(PanelModel.id.in_(panel_ids))
                    panel_records = list(session.execute(stmt).scalars().all())

                    panel_options = []
                    for panel in panel_records:
                        panel_options.append({
                            "id": panel.id,
                            "name": panel.name,
                            "power_w": panel.power_w,
                            "price_eur": panel.specs.get("price_eur") if panel.specs else None,
                        })

                    if "optimization" not in hydrated:
                        hydrated["optimization"] = {}
                    hydrated["optimization"]["panel_options"] = panel_options

                # Hydrate battery options for campaign
                if "battery_ids" in selections and selections["battery_ids"]:
                    battery_ids = selections["battery_ids"]
                    stmt = select(BatteryModel).where(BatteryModel.id.in_(battery_ids))
                    battery_records = list(session.execute(stmt).scalars().all())

                    battery_options = []
                    for bat in battery_records:
                        battery_options.append({
                            "id": bat.id,
                            "name": bat.name,
                            "specs": {
                                "capacity_kwh": bat.capacity_kwh or 0.0,
                                "cycles_life": bat.specs.get("cycles_life") if bat.specs else 5000,
                            },
                            "price_eur": bat.specs.get("price_eur") if bat.specs else None,
                            "manufacturer": bat.manufacturer,
                            "model_number": bat.model_number,
                            "datasheet": bat.datasheet,
                        })

                    if "optimization" not in hydrated:
                        hydrated["optimization"] = {}
                    hydrated["optimization"]["battery_options"] = battery_options

            # Hydrate inverter
            if "inverter_id" in scenario_data:
                inverter_id = scenario_data["inverter_id"]
                stmt = select(InverterModel).where(InverterModel.id == inverter_id)
                inverter = session.execute(stmt).scalar_one_or_none()
                if not inverter:
                    raise ValueError(f"Inverter ID {inverter_id} not found")

                # Build energy section with inverter data
                if "energy" not in hydrated:
                    hydrated["energy"] = {}
                # Get p_ac_max from specs or nominal_power_kw
                p_ac_max = (
                    inverter.specs.get("p_ac_max_kw")
                    if inverter.specs
                    else inverter.nominal_power_kw
                )
                if p_ac_max:
                    hydrated["energy"]["inverter_p_ac_max_kw"] = p_ac_max

            # Hydrate panel
            if "panel_id" in scenario_data:
                panel_id = scenario_data["panel_id"]
                stmt = select(PanelModel).where(PanelModel.id == panel_id)
                panel = session.execute(stmt).scalar_one_or_none()
                if not panel:
                    raise ValueError(f"Panel ID {panel_id} not found")

                # Store panel power for later use in solar and energy configs
                if "solar" not in hydrated:
                    hydrated["solar"] = {}
                if "energy" not in hydrated:
                    hydrated["energy"] = {}

                # Panel specs might be used by optimization - store in metadata
                hydrated.setdefault("_hardware_metadata", {})["panel"] = {
                    "id": panel.id,
                    "name": panel.name,
                    "power_w": panel.power_w,
                    "price_eur": panel.specs.get("price_eur") if panel.specs else None,
                }

            # Hydrate battery
            if "battery_id" in scenario_data:
                battery_id = scenario_data["battery_id"]
                stmt = select(BatteryModel).where(BatteryModel.id == battery_id)
                battery = session.execute(stmt).scalar_one_or_none()
                if not battery:
                    raise ValueError(f"Battery ID {battery_id} not found")

                if "energy" not in hydrated:
                    hydrated["energy"] = {}

                hydrated["energy"]["battery_specs"] = {
                    "capacity_kwh": battery.capacity_kwh or 0.0,
                    "cycles_life": (
                        battery.specs.get("cycles_life", 0) if battery.specs else 0
                    ),
                }

                # Store battery metadata
                hydrated.setdefault("_hardware_metadata", {})["battery"] = {
                    "id": battery.id,
                    "name": battery.name,
                    "capacity_kwh": battery.capacity_kwh,
                    "price_eur": battery.specs.get("price_eur") if battery.specs else None,
                }

            # Hydrate load profile
            if "load_profile_id" in scenario_data:
                load_id = scenario_data["load_profile_id"]
                stmt = select(LoadProfileModel).where(LoadProfileModel.id == load_id)
                load_profile = session.execute(stmt).scalar_one_or_none()
                if not load_profile:
                    raise ValueError(f"Load profile ID {load_id} not found")

                hydrated["load_profile"] = load_profile.data

            # Hydrate price profile
            if "price_profile_id" in scenario_data:
                price_id = scenario_data["price_profile_id"]
                stmt = select(PriceProfileModel).where(PriceProfileModel.id == price_id)
                price_profile = session.execute(stmt).scalar_one_or_none()
                if not price_profile:
                    raise ValueError(f"Price profile ID {price_id} not found")

                hydrated["price"] = price_profile.data

            return hydrated
