"""
Database persistence layer for PV system configurations and results.

Provides CRUD operations for hardware components (inverters, panels, batteries),
simulation configurations, and analysis results. Handles serialization of complex
data structures to JSON for database storage.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterable, Mapping

from sqlalchemy.orm import Session

from ..db.models import (
    BatteryModel,
    ClimateProfileModel,
    InverterModel,
    LoadProfileModel,
    MarketProfileModel,
    OptimizationRecord,
    PanelModel,
    PriceProfileModel,
    RunResultRecord,
    SavedConfigurationModel,
    ScenarioRecord,
    SolarProfileModel,
)
from ..db.session import SessionLocal
from .climate_repo import ClimateProfileRepository
from .configuration_repo import ConfigurationRepository
from .execution_repo import ExecutionRepository
from .hardware_repo import HardwareRepository
from .market_repo import MarketProfileRepository
from .solar_repo import SolarProfileRepository
from .hydration import hydrate_scenario, hydrate_optimization, hydrate_scenario_from_ids


class PersistenceService:
    """
    Unified persistence service providing access to all repositories.

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

    Maintains backward compatibility via delegation while enabling
    direct repository access for advanced use cases.

    Attributes:
        _session_factory: SQLAlchemy session factory for creating database connections.
        hardware: HardwareRepository for hardware CRUD operations.
        configurations: ConfigurationRepository for configuration CRUD operations.
        executions: ExecutionRepository for execution CRUD operations.

    Example:
        ```python
        from sim_stochastic_pv.persistence import PersistenceService

        # Create service
        service = PersistenceService()

        # Store hardware (backward compatible API)
        inverter = service.upsert_inverter({
            "name": "Fronius Primo 5.0",
            "p_ac_max_kw": 5.0,
            "price_eur": 1500.0,
            "manufacturer": "Fronius"
        })

        # Or use repository directly (advanced)
        inverter = service.hardware.upsert_inverter({...})

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

        # Initialize repositories
        self.hardware = HardwareRepository(self._session_factory)
        self.configurations = ConfigurationRepository(self._session_factory)
        self.executions = ExecutionRepository(self._session_factory)
        self.solar = SolarProfileRepository(self._session_factory)
        # Phase 15 — calibrated stochastic thermal profiles.
        self.climate = ClimateProfileRepository(self._session_factory)
        # Reusable electricity-market profiles (cached wholesale price surface
        # + dedicated-withdrawal valuation parameters).
        self.market = MarketProfileRepository(self._session_factory)

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

    # Hardware operations (delegate to HardwareRepository)
    def upsert_inverter(self, inverter_data: Any) -> InverterModel | None:
        """Insert or update an inverter record based on its name."""
        return self.hardware.upsert_inverter(inverter_data)

    def upsert_panel(self, panel_data: Any) -> PanelModel | None:
        """Insert or update a panel record."""
        return self.hardware.upsert_panel(panel_data)

    def upsert_battery(self, battery_data: Any) -> BatteryModel | None:
        """Insert or update a battery record."""
        return self.hardware.upsert_battery(battery_data)

    def list_inverters(self) -> list[InverterModel]:
        """List all available inverters."""
        return self.hardware.list_inverters()

    def delete_inverter(self, inverter_id: int) -> bool:
        """Delete an inverter by ID. Returns True if deleted, False if not found."""
        return self.hardware.delete_inverter(inverter_id)

    def update_inverter(self, inverter_id: int, inverter_data: Any) -> InverterModel | None:
        """Update an inverter by ID. Returns None if not found."""
        return self.hardware.update_inverter(inverter_id, inverter_data)

    def list_panels(self) -> list[PanelModel]:
        """List all available panels."""
        return self.hardware.list_panels()

    def delete_panel(self, panel_id: int) -> bool:
        """Delete a panel by ID. Returns True if deleted, False if not found."""
        return self.hardware.delete_panel(panel_id)

    def update_panel(self, panel_id: int, panel_data: Any) -> PanelModel | None:
        """Update a panel by ID. Returns None if not found."""
        return self.hardware.update_panel(panel_id, panel_data)

    def list_batteries(self) -> list[BatteryModel]:
        """List all available batteries."""
        return self.hardware.list_batteries()

    def delete_battery(self, battery_id: int) -> bool:
        """Delete a battery by ID. Returns True if deleted, False if not found."""
        return self.hardware.delete_battery(battery_id)

    def update_battery(self, battery_id: int, battery_data: Any) -> BatteryModel | None:
        """Update a battery by ID. Returns None if not found."""
        return self.hardware.update_battery(battery_id, battery_data)

    # Configuration operations (delegate to ConfigurationRepository)
    def upsert_load_profile(self, name: str, profile_type: str, data: dict) -> LoadProfileModel:
        """Insert or update a load profile."""
        return self.configurations.upsert_load_profile(name, profile_type, data)

    def list_load_profiles(self) -> list[LoadProfileModel]:
        """List all saved load profiles."""
        return self.configurations.list_load_profiles()

    def delete_load_profile(self, profile_id: int) -> bool:
        """Delete a load profile by ID. Returns True if deleted, False if not found."""
        return self.configurations.delete_load_profile(profile_id)

    def update_load_profile(
        self, profile_id: int, name: str, profile_type: str, data: dict
    ) -> LoadProfileModel | None:
        """Update a load profile by ID. Returns None if not found."""
        return self.configurations.update_load_profile(profile_id, name, profile_type, data)

    def upsert_price_profile(self, name: str, data: dict) -> PriceProfileModel:
        """Insert or update a price profile."""
        return self.configurations.upsert_price_profile(name, data)

    def list_price_profiles(self) -> list[PriceProfileModel]:
        """List all saved price profiles."""
        return self.configurations.list_price_profiles()

    def delete_price_profile(self, profile_id: int) -> bool:
        """Delete a price profile by ID. Returns True if deleted, False if not found."""
        return self.configurations.delete_price_profile(profile_id)

    def update_price_profile(
        self, profile_id: int, name: str, data: dict
    ) -> PriceProfileModel | None:
        """Update a price profile by ID. Returns None if not found."""
        return self.configurations.update_price_profile(profile_id, name, data)

    def save_configuration(self, name: str, config_type: str, data: dict) -> SavedConfigurationModel:
        """Save or update a configuration (scenario or optimization)."""
        return self.configurations.save_configuration(name, config_type, data)

    def list_configurations(self, config_type: str | None = None) -> list[SavedConfigurationModel]:
        """List all saved configurations, optionally filtered by type."""
        return self.configurations.list_configurations(config_type)

    def get_configuration_by_id(self, config_id: int) -> SavedConfigurationModel | None:
        """Retrieve a saved configuration by ID."""
        return self.configurations.get_configuration_by_id(config_id)

    def get_configuration_by_name(self, name: str) -> SavedConfigurationModel | None:
        """Retrieve a saved configuration by its unique name."""
        return self.configurations.get_configuration_by_name(name)

    def delete_configuration(self, config_id: int) -> bool:
        """Delete a saved configuration by ID. Returns True if deleted, False if not found."""
        return self.configurations.delete_configuration(config_id)

    def update_configuration(
        self, config_id: int, name: str, config_type: str, data: dict
    ) -> SavedConfigurationModel | None:
        """Update a saved configuration by ID. Returns None if not found."""
        return self.configurations.update_configuration(config_id, name, config_type, data)

    # Execution operations (delegate to ExecutionRepository)
    def record_scenario(
        self,
        name: str,
        config: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        inverter: InverterModel | None = None,
        panel: PanelModel | None = None,
        battery: BatteryModel | None = None,
    ) -> ScenarioRecord:
        """Persist a scenario definition for later re-use."""
        return self.executions.record_scenario(name, config, metadata, inverter, panel, battery)

    def record_optimization(
        self,
        label: str,
        request_payload: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> OptimizationRecord:
        """Persist an optimization request/result summary."""
        return self.executions.record_optimization(label, request_payload, metadata)

    def record_run_result(
        self,
        result_type: str,
        summary: Mapping[str, Any],
        *,
        scenario: ScenarioRecord | None = None,
        optimization: OptimizationRecord | None = None,
        output_dir: str | None = None,
    ) -> RunResultRecord:
        """Store the outcome of an analysis or optimization run."""
        return self.executions.record_run_result(
            result_type, summary, scenario=scenario, optimization=optimization, output_dir=output_dir
        )

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
        """Fetch the latest run results with optional filters and pagination."""
        return self.executions.list_run_results(
            limit=limit,
            offset=offset,
            scenario_name=scenario_name,
            location=location,
            date_from=date_from,
            date_to=date_to,
            include_archived=include_archived,
        )

    def get_run_result(self, run_id: int) -> RunResultRecord | None:
        """Fetch a single run result by primary key, or None if missing (Phase 11)."""
        return self.executions.get_run_result(run_id)

    def delete_run_result(self, run_id: int) -> bool:
        """Hard-delete a run result by id. Returns True if removed (Phase 12)."""
        return self.executions.delete_run_result(run_id)

    def set_run_archived(
        self, run_id: int, archived: bool
    ) -> RunResultRecord | None:
        """Toggle the run's archived flag (Phase 12)."""
        return self.executions.set_run_archived(run_id, archived)

    def list_distinct_run_locations(self) -> list[str]:
        """Distinct location names across all runs (Phase 12 — for filters)."""
        return self.executions.list_distinct_locations()

    def list_scenarios(self) -> list[ScenarioRecord]:
        """List all saved scenarios (history)."""
        return self.executions.list_scenarios()

    # Hydration operations
    def hydrate_scenario(self, scenario_data: dict[str, Any]) -> dict[str, Any]:
        """
        Hydrate a scenario configuration by replacing hardware/profile IDs with full specs.

        Handles single hardware selections (inverter_id, panel_id, battery_id, etc.).

        Args:
            scenario_data: Partial scenario dict with IDs instead of full specs.

        Returns:
            Complete scenario dict with all hardware and profile data hydrated.
        """
        with self.session() as session:
            return hydrate_scenario(scenario_data, session)

    def hydrate_optimization(self, optimization_data: dict[str, Any]) -> dict[str, Any]:
        """
        Hydrate an optimization configuration by expanding hardware_selections.

        Handles multiple hardware selections (hardware_selections.inverter_ids[], etc.).

        Args:
            optimization_data: Partial optimization dict with hardware_selections.

        Returns:
            Complete optimization dict with all hardware options expanded.
        """
        with self.session() as session:
            return hydrate_optimization(optimization_data, session)

    def hydrate_scenario_from_ids(self, scenario_data: dict[str, Any]) -> dict[str, Any]:
        """
        Hydrate a scenario or optimization configuration (backward compatible).

        DEPRECATED: Use hydrate_scenario() or hydrate_optimization() instead.

        Auto-detects whether to call hydrate_scenario() or hydrate_optimization()
        based on the presence of hardware_selections in the data.

        Args:
            scenario_data: Partial scenario/optimization dict with IDs.

        Returns:
            Complete configuration dict with all data hydrated.
        """
        with self.session() as session:
            return hydrate_scenario_from_ids(scenario_data, session)

    # ========================================================================
    # Solar Profile Operations (Delegate to SolarProfileRepository)
    # ========================================================================

    def upsert_solar_profile(self, data: dict[str, Any]) -> SolarProfileModel:
        """
        Insert or update solar profile by name (unique key).

        Delegates to SolarProfileRepository.upsert_solar_profile().
        See SolarProfileRepository for detailed documentation.

        Args:
            data: Dictionary containing solar profile fields.

        Returns:
            SolarProfileModel: The created or updated database record.
        """
        return self.solar.upsert_solar_profile(data)

    def list_solar_profiles(self) -> list[SolarProfileModel]:
        """
        List all solar profiles ordered by name.

        Delegates to SolarProfileRepository.list_solar_profiles().

        Returns:
            List of all solar profile records.
        """
        return self.solar.list_solar_profiles()

    def get_solar_profile_by_id(self, profile_id: int) -> SolarProfileModel | None:
        """
        Get solar profile by primary key ID.

        Delegates to SolarProfileRepository.get_solar_profile_by_id().

        Args:
            profile_id: Primary key ID of the solar profile.

        Returns:
            The matching profile record, or None if not found.
        """
        return self.solar.get_solar_profile_by_id(profile_id)

    def get_solar_profile_by_name(self, name: str) -> SolarProfileModel | None:
        """
        Get solar profile by unique name.

        Delegates to SolarProfileRepository.get_solar_profile_by_name().

        Args:
            name: Unique name of the solar profile.

        Returns:
            The matching profile record, or None if not found.
        """
        return self.solar.get_solar_profile_by_name(name)

    def delete_solar_profile(self, profile_id: int) -> bool:
        """
        Delete solar profile by ID.

        Delegates to SolarProfileRepository.delete_solar_profile().

        Args:
            profile_id: Primary key ID of the profile to delete.

        Returns:
            True if a profile was deleted, False if not found.
        """
        return self.solar.delete_solar_profile(profile_id)

    def update_solar_profile(
        self, profile_id: int, data: dict[str, Any]
    ) -> SolarProfileModel | None:
        """
        Update solar profile by ID (allows rename).

        Delegates to SolarProfileRepository.update_solar_profile().

        Args:
            profile_id: Primary key ID of the profile.
            data: New field values (partial dict).

        Returns:
            Updated record, or None if the ID does not exist.
        """
        return self.solar.update_solar_profile(profile_id, data)

    # ========================================================================
    # Climate Profile Operations (Delegate to ClimateProfileRepository)
    # ========================================================================

    def list_climate_profiles(self) -> list[ClimateProfileModel]:
        """List all climate profiles ordered by name."""
        return self.climate.list_climate_profiles()

    def get_climate_profile_by_id(self, profile_id: int) -> ClimateProfileModel | None:
        """Fetch a climate profile by primary key."""
        return self.climate.get_climate_profile_by_id(profile_id)

    def delete_climate_profile(self, profile_id: int) -> bool:
        """Delete a climate profile by ID. Returns True if deleted, False if not found."""
        return self.climate.delete_climate_profile(profile_id)

    def update_climate_profile(
        self, profile_id: int, data: dict[str, Any]
    ) -> ClimateProfileModel | None:
        """Update a climate profile by ID (partial). Returns None if not found."""
        return self.climate.update_climate_profile(profile_id, data)

    def load_thermal_model(self, profile_id: int):
        """Convenience: hydrate a runtime ThermalModel (Phase 15) by climate profile id."""
        return self.climate.load_thermal_model(profile_id)

    # ========================================================================
    # Market Profile Operations (Delegate to MarketProfileRepository)
    # ========================================================================

    def upsert_market_profile(self, data: dict[str, Any]) -> MarketProfileModel:
        """Insert or update a market profile by name."""
        return self.market.upsert_market_profile(data)

    def list_market_profiles(self) -> list[MarketProfileModel]:
        """List all market profiles ordered by name."""
        return self.market.list_market_profiles()

    def get_market_profile_by_id(self, profile_id: int) -> MarketProfileModel | None:
        """Fetch a market profile by primary key."""
        return self.market.get_market_profile_by_id(profile_id)

    def get_market_profile_by_name(self, name: str) -> MarketProfileModel | None:
        """Fetch a market profile by unique name."""
        return self.market.get_market_profile_by_name(name)

    def update_market_profile(
        self, profile_id: int, data: dict[str, Any]
    ) -> MarketProfileModel | None:
        """Update a market profile by ID (partial). Returns None if not found."""
        return self.market.update_market_profile(profile_id, data)

    def delete_market_profile(self, profile_id: int) -> bool:
        """Delete a market profile by ID. Returns True if deleted, False if not found."""
        return self.market.delete_market_profile(profile_id)

    def load_market_provider(self, profile_id: int):
        """Convenience: hydrate a runtime MarketPriceProvider by market profile id."""
        return self.market.load_market_provider(profile_id)


__all__ = ["PersistenceService"]
