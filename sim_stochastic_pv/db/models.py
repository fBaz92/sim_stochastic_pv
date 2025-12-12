"""
SQLAlchemy database models for PV system simulation persistence.

Defines the database schema for storing hardware components, simulation
configurations, and analysis results. All models inherit automatic timestamp
tracking via TimestampMixin.
"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import relationship

from .session import Base


class TimestampMixin:
    """
    Mixin adding automatic created_at and updated_at timestamps.

    Provides automatic timestamp tracking for database records:
    - created_at: Set on insert (never changes)
    - updated_at: Set on insert, updated on every modification

    Both timestamps use timezone-aware datetime (UTC recommended).

    Example:
        ```python
        class MyModel(Base, TimestampMixin):
            __tablename__ = "my_table"
            id = Column(Integer, primary_key=True)
            name = Column(String(255))

        # Usage
        record = MyModel(name="Test")
        session.add(record)
        session.commit()
        # record.created_at and record.updated_at are automatically set

        # Update
        record.name = "Updated"
        session.commit()
        # record.updated_at is automatically updated
        ```

    Notes:
        - Timestamps managed by database (server_default, onupdate)
        - Timezone-aware (recommended: configure DB to use UTC)
        - created_at immutable after insert
        - updated_at changes on every UPDATE
    """
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class InverterModel(Base, TimestampMixin):
    """
    Database model for PV inverter hardware specifications.

    Stores technical and commercial data for inverter products used in
    scenario configurations. Supports both standard and hybrid inverters
    with integrated battery storage.

    Attributes:
        id: Primary key (auto-increment).
        name: Unique inverter identifier (e.g., "Fronius Primo 5.0").
        manufacturer: Manufacturer name (e.g., "Fronius").
        model_number: Manufacturer model number.
        nominal_power_kw: AC output power rating (kW).
        datasheet: Additional technical specifications (JSON).
        specs: Complete inverter specs including pricing (JSON).
        scenarios: Related scenario records using this inverter.

    Example:
        ```python
        inverter = InverterModel(
            name="SMA Sunny Boy 5.0",
            manufacturer="SMA",
            model_number="SB5.0-1SP-US-41",
            nominal_power_kw=5.0,
            specs={
                "p_ac_max_kw": 5.0,
                "p_dc_max_kw": 5.5,
                "price_eur": 1600.0,
                "install_cost_eur": 2000.0
            }
        )
        ```

    Notes:
        - name is unique constraint (upsert key)
        - specs JSON contains full hardware configuration
        - Hybrid inverters store integrated_battery_specs in specs
    """
    __tablename__ = "inverters"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    manufacturer = Column(String(255), nullable=True)
    model_number = Column(String(255), nullable=True)
    nominal_power_kw = Column(Float, nullable=True)
    datasheet = Column(JSON, nullable=True)
    specs = Column(JSON, nullable=True)

    scenarios = relationship("ScenarioRecord", back_populates="inverter")


class PanelModel(Base, TimestampMixin):
    """
    Database model for solar panel specifications.

    Stores technical and commercial data for PV panel products used in
    system configurations.

    Attributes:
        id: Primary key (auto-increment).
        name: Unique panel identifier (e.g., "Longi LR5-72HPH-540M").
        manufacturer: Manufacturer name (e.g., "Longi Solar").
        model_number: Manufacturer model number.
        power_w: Nominal peak power per panel (Watts, STC).
        datasheet: Additional technical specifications (JSON).
        specs: Complete panel specs including pricing (JSON).
        scenarios: Related scenario records using this panel.

    Example:
        ```python
        panel = PanelModel(
            name="Canadian Solar CS3W-400P",
            manufacturer="Canadian Solar",
            model_number="CS3W-400P",
            power_w=400.0,
            specs={
                "power_w": 400.0,
                "price_eur": 120.0,
                "efficiency": 0.199,
                "dimensions_mm": [1000, 1980, 40]
            }
        )
        ```

    Notes:
        - name is unique constraint (upsert key)
        - power_w extracted for quick queries
        - specs JSON contains full product data
    """
    __tablename__ = "panels"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    manufacturer = Column(String(255), nullable=True)
    model_number = Column(String(255), nullable=True)
    power_w = Column(Float, nullable=True)
    datasheet = Column(JSON, nullable=True)
    specs = Column(JSON, nullable=True)

    scenarios = relationship("ScenarioRecord", back_populates="panel")


class BatteryModel(Base, TimestampMixin):
    """
    Database model for battery energy storage system specifications.

    Stores technical and commercial data for battery products used in
    energy storage configurations.

    Attributes:
        id: Primary key (auto-increment).
        name: Unique battery identifier (e.g., "Tesla Powerwall 2").
        manufacturer: Manufacturer name (e.g., "Tesla").
        model_number: Manufacturer model number.
        capacity_kwh: Usable capacity per battery module (kWh).
        datasheet: Additional technical specifications (JSON).
        specs: Complete battery specs including cycle life and pricing (JSON).
        scenarios: Related scenario records using this battery.

    Example:
        ```python
        battery = BatteryModel(
            name="BYD Battery-Box Premium LVS 4.0",
            manufacturer="BYD",
            model_number="LVS 4.0",
            capacity_kwh=4.0,
            specs={
                "capacity_kwh": 4.0,
                "cycles_life": 8000,
                "price_eur": 1500.0,
                "chemistry": "LiFePO4"
            }
        )
        ```

    Notes:
        - name is unique constraint (upsert key)
        - capacity_kwh extracted for quick queries
        - specs.cycles_life critical for degradation modeling
    """
    __tablename__ = "batteries"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    manufacturer = Column(String(255), nullable=True)
    model_number = Column(String(255), nullable=True)
    capacity_kwh = Column(Float, nullable=True)
    datasheet = Column(JSON, nullable=True)
    specs = Column(JSON, nullable=True)

    scenarios = relationship("ScenarioRecord", back_populates="battery")


class ScenarioRecord(Base, TimestampMixin):
    """
    Database model for complete PV system scenario configurations.

    Stores a snapshot of a complete system configuration including hardware
    references, simulation parameters, and metadata. Scenarios can be
    replayed or used as templates for new simulations.

    Attributes:
        id: Primary key (auto-increment).
        name: Scenario descriptive name.
        config: Full simulation configuration (JSON).
        extra_metadata: Additional user-defined data (JSON).
        inverter_id: Foreign key to inverter (optional).
        panel_id: Foreign key to panel (optional).
        battery_id: Foreign key to battery (optional).
        load_profile_id: Foreign key to load profile (optional).
        price_profile_id: Foreign key to price profile (optional).
        inverter: Related inverter model.
        panel: Related panel model.
        battery: Related battery model.
        load_profile: Related load profile.
        price_profile: Related price profile.
        runs: Analysis results using this scenario.

    Example:
        ```python
        scenario = ScenarioRecord(
            name="Residential 5.4kWp + 10kWh",
            config={
                "energy": {"pv_kwp": 5.4, "n_batteries": 2},
                "economic": {"investment_eur": 12000, "n_mc": 500}
            },
            extra_metadata={"location": "Milano", "roof_angle": 30},
            inverter_id=1,
            panel_id=2,
            battery_id=3
        )
        ```

    Notes:
        - config JSON contains full simulation parameters
        - Hardware references optional (can be embedded in config)
        - Linked to run results for historical tracking
    """
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    config = Column(JSON, nullable=False)
    extra_metadata = Column("metadata", JSON, nullable=True)

    inverter_id = Column(Integer, ForeignKey("inverters.id"), nullable=True)
    panel_id = Column(Integer, ForeignKey("panels.id"), nullable=True)
    battery_id = Column(Integer, ForeignKey("batteries.id"), nullable=True)
    load_profile_id = Column(Integer, ForeignKey("load_profiles.id"), nullable=True)
    price_profile_id = Column(Integer, ForeignKey("price_profiles.id"), nullable=True)

    inverter = relationship("InverterModel", back_populates="scenarios")
    panel = relationship("PanelModel", back_populates="scenarios")
    battery = relationship("BatteryModel", back_populates="scenarios")
    load_profile = relationship("LoadProfileModel")
    price_profile = relationship("PriceProfileModel")
    runs = relationship("RunResultRecord", back_populates="scenario")


class OptimizationRecord(Base, TimestampMixin):
    """
    Database model for optimization campaign metadata.

    Stores information about multi-scenario optimization runs, tracking
    the search space and execution status.

    Attributes:
        id: Primary key (auto-increment).
        label: Optimization campaign name.
        status: Execution status (e.g., "completed", "running", "failed").
        request: Optimization request configuration (JSON).
        extra_metadata: Additional campaign data (JSON).
        runs: Individual scenario evaluation results.

    Example:
        ```python
        optimization = OptimizationRecord(
            label="Residential PV Sizing Study 2025",
            status="completed",
            request={
                "inverter_options": [...],
                "panel_count_options": [8, 10, 12],
                "battery_count_options": [1, 2]
            },
            extra_metadata={"total_scenarios": 36}
        )
        ```

    Notes:
        - Tracks multi-scenario optimization campaigns
        - request JSON defines hardware search space
        - Linked to multiple run results
    """
    __tablename__ = "optimizations"

    id = Column(Integer, primary_key=True)
    label = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False, default="completed")
    request = Column(JSON, nullable=False)
    extra_metadata = Column("metadata", JSON, nullable=True)

    runs = relationship("RunResultRecord", back_populates="optimization")


class RunResultRecord(Base, TimestampMixin):
    """
    Database model for simulation and optimization results.

    Stores the outcomes of analysis runs, linking to the source scenario
    or optimization campaign. Results are stored as JSON summaries with
    optional file system paths for detailed exports.

    Attributes:
        id: Primary key (auto-increment).
        result_type: Type of result ("analysis", "optimization", etc.).
        summary: Key metrics and statistics (JSON).
        output_dir: Filesystem path to exported result files (optional).
        scenario_id: Foreign key to scenario (for single-scenario runs).
        optimization_id: Foreign key to optimization (for campaign runs).
        scenario: Related scenario record.
        optimization: Related optimization record.

    Example:
        ```python
        result = RunResultRecord(
            result_type="analysis",
            summary={
                "final_gain_eur": 15000.0,
                "irr_mean": 0.082,
                "payback_years": 9.5,
                "break_even_month": 114
            },
            output_dir="/exports/run_2025-01-15_143022",
            scenario_id=42
        )
        ```

    Notes:
        - summary JSON contains aggregated metrics
        - output_dir points to detailed CSV/PNG exports
        - Either scenario_id or optimization_id should be set
    """
    __tablename__ = "run_results"

    id = Column(Integer, primary_key=True)
    result_type = Column(String(50), nullable=False)
    summary = Column(JSON, nullable=False)
    output_dir = Column(Text, nullable=True)

    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=True)
    optimization_id = Column(Integer, ForeignKey("optimizations.id"), nullable=True)

    scenario = relationship("ScenarioRecord", back_populates="runs")
    optimization = relationship("OptimizationRecord", back_populates="runs")


class LoadProfileModel(Base, TimestampMixin):
    """
    Database model for electricity consumption profiles.

    Stores load profile configurations that can be referenced by scenarios.
    Supports multiple profile types (ARERA, custom 24h, custom hourly).

    Attributes:
        id: Primary key (auto-increment).
        name: Unique profile identifier.
        profile_type: Profile type ("arera", "custom", "custom_24h").
        data: Profile configuration data (JSON).

    Example:
        ```python
        profile = LoadProfileModel(
            name="Residential ARERA Standard",
            profile_type="arera",
            data={
                "bl_table": [[110.67, 99.78, 157.05], ...],  # 12x3 array
                "use_stochastic_variation": True,
                "variation_percentiles": [-0.10, 0.10]
            }
        )
        ```

    Notes:
        - name is unique constraint
        - data structure depends on profile_type
        - Linked to scenarios via load_profile_id
    """
    __tablename__ = "load_profiles"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    profile_type = Column(String(50), nullable=False)  # arera, custom, custom_24h
    data = Column(JSON, nullable=False)  # The profile values


class PriceProfileModel(Base, TimestampMixin):
    """
    Database model for electricity price configurations.

    Stores price escalation model parameters that can be referenced by
    scenarios. Supports deterministic and stochastic escalation.

    Attributes:
        id: Primary key (auto-increment).
        name: Unique price profile identifier.
        data: Price model configuration (JSON).

    Example:
        ```python
        price_profile = PriceProfileModel(
            name="Conservative Escalation 2%",
            data={
                "base_price_eur_per_kwh": 0.22,
                "annual_escalation": 0.02,
                "use_stochastic_escalation": False,
                "seasonal_factors": [0.05, 0.04, ..., 0.05]
            }
        )
        ```

    Notes:
        - name is unique constraint
        - data contains EscalatingPriceModel parameters
        - Linked to scenarios via price_profile_id
    """
    __tablename__ = "price_profiles"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    data = Column(JSON, nullable=False)  # base price, escalation, etc.


class SavedConfigurationModel(Base, TimestampMixin):
    """
    Database model for saved simulation configurations.

    Stores complete configuration snapshots that can be loaded and
    reused. Supports both single-scenario and optimization campaign configs.

    Attributes:
        id: Primary key (auto-increment).
        name: Unique configuration identifier.
        config_type: Configuration type ("scenario", "campaign").
        data: Complete configuration data (JSON).

    Example:
        ```python
        config = SavedConfigurationModel(
            name="Standard Residential Template",
            config_type="scenario",
            data={
                "energy": {...},
                "solar": {...},
                "load_profile": {...},
                "price": {...},
                "economic": {...}
            }
        )
        ```

    Notes:
        - name is unique constraint
        - config_type determines data structure
        - "scenario": Single system configuration
        - "campaign": Optimization search space
    """
    __tablename__ = "saved_configurations"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    config_type = Column(String(50), nullable=False)  # scenario, campaign
    data = Column(JSON, nullable=False)
