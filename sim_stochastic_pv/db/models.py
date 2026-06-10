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


class LocationModel(Base, TimestampMixin):
    """
    Database model for installation sites (geographic locations).

    A location is the durable anchor of every site-specific dataset: the
    solar production profile (PVGIS), the calibrated stochastic climate
    profile (Open-Meteo), and — in future — plant designs. Before this
    table existed, the "site" was a free-text ``location_name`` string
    duplicated on each profile, which made the address ↔ profile pairing
    fragile (renaming one side silently orphaned the other) and gave the
    import flow no transactional unit to anchor to.

    Attributes:
        id: Primary key (auto-increment).
        name: Unique short identifier chosen by the user (e.g. ``"Pavullo"``).
            Used as the upsert key and propagated as the default name of the
            linked solar/climate profiles.
        address: Free-text query as typed by the user in the geocoding
            search box (e.g. ``"via Giardini 5, Pavullo"``). Optional —
            map-only picks have no typed address.
        display_name: Canonical human-readable address returned by the
            geocoder (Nominatim ``display_name``). Optional.
        latitude: Decimal latitude in degrees (-90 to +90).
        longitude: Decimal longitude in degrees (-180 to +180).
        elevation_m: Elevation above sea level in metres, when known
            (PVGIS or Open-Meteo gridcell value). Optional.
        notes: Free-text user notes about the site.

    Example:
        ```python
        site = LocationModel(
            name="Pavullo",
            address="Pavullo nel Frignano",
            display_name="Pavullo nel Frignano, Modena, Emilia-Romagna, Italia",
            latitude=44.336,
            longitude=10.831,
            elevation_m=682.0,
        )
        ```

    Notes:
        - ``name`` is the unique upsert key (same convention as profiles).
        - Linked from ``SolarProfileModel.location_id`` and
          ``ClimateProfileModel.location_id`` (nullable FKs): one location
          can own several solar profiles (e.g. different tilt/azimuth
          variants) but typically owns exactly one of each.
        - Deleting a location detaches its profiles by default (FK set to
          NULL); hard-deleting the profiles too is an explicit caller choice.
    """

    __tablename__ = "locations"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)

    address = Column(String(500), nullable=True)
    display_name = Column(String(500), nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation_m = Column(Float, nullable=True)

    notes = Column(Text, nullable=True)


class PlantDesignModel(Base, TimestampMixin):
    """
    Database model for a PV plant design ("Impianto").

    A plant design is the first-class description of one specific system,
    with two fidelity levels sharing the same table and the same economic
    pipeline:

    - ``design_level = "essential"`` — a received commercial offer,
      described only by its nameplate data: AC power, DC power (defaults
      to the AC power when the quote does not state it), optional storage
      capacity, turn-key cost and tax incentive. Enough to answer "is
      this offer worth it?" through the Monte Carlo analysis.
    - ``design_level = "detailed"`` — a full electrical design produced
      by the designer (string layout, cables, protections, loss chain).
      The extra payload lives in ``data`` and is introduced by the
      electrical-designer feature; the resolver maps both levels onto
      the same simulator configuration.

    Attributes:
        id: Primary key (auto-increment).
        name: Unique design identifier chosen by the user
            (e.g. ``"Offerta Rossi 6kW"``). Upsert key.
        design_level: ``"essential"`` or ``"detailed"``.
        description: Optional free-text description.
        data: JSON payload. For ``essential``:
            ``{"p_ac_kw": float, "p_dc_kwp": float | null,
            "storage_kwh": float | null, "total_cost_eur": float,
            "tax_bonus": {"enabled", "fraction_of_investment",
            "duration_years"} | null}``.
            ``p_dc_kwp = null`` means "equal to ``p_ac_kw``";
            ``storage_kwh`` null/0 means no battery.
        inverter_id: Optional FK to a catalogue inverter (detailed designs
            or offers that name the hardware).
        panel_id: Optional FK to a catalogue panel.
        battery_id: Optional FK to a catalogue battery.
        location_id: Optional FK to the installation site. When set, the
            scenario resolver inherits the site's solar profile (and
            climate profile, when present) automatically.

    Example:
        ```python
        design = PlantDesignModel(
            name="Offerta Rossi 6kW",
            design_level="essential",
            data={
                "p_ac_kw": 6.0,
                "p_dc_kwp": 6.6,
                "storage_kwh": 10.0,
                "total_cost_eur": 14500.0,
                "tax_bonus": {
                    "enabled": True,
                    "fraction_of_investment": 0.5,
                    "duration_years": 10,
                },
            },
            location_id=1,
        )
        ```

    Notes:
        - Scenarios reference a design via ``plant_design_id`` in their
          config JSON (resolved during hydration); there is no FK from
          ``scenarios`` so saved scenario snapshots stay self-contained.
        - Deleting a design does not touch the runs already executed
          from it (their config snapshot is frozen in the run record).
    """

    __tablename__ = "plant_designs"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    design_level = Column(String(32), nullable=False, default="essential")
    description = Column(Text, nullable=True)

    data = Column(JSON, nullable=False)

    inverter_id = Column(Integer, ForeignKey("inverters.id"), nullable=True)
    panel_id = Column(Integer, ForeignKey("panels.id"), nullable=True)
    battery_id = Column(Integer, ForeignKey("batteries.id"), nullable=True)
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=True)


class SolarProfileModel(Base, TimestampMixin):
    """
    Database model for solar irradiance profiles by geographic location.

    Stores monthly solar production data for specific locations, enabling
    accurate PV system simulations across different Italian cities or custom
    locations. Includes optimal panel orientation for each location.

    Attributes:
        id: Primary key (auto-increment).
        name: Unique profile identifier (e.g., "Pavullo_nel_Frignano", "Milano").
        location_name: Full location description (e.g., "Pavullo nel Frignano, Modena, Italy").
        latitude: Location latitude in degrees (-90 to +90).
        longitude: Location longitude in degrees (-180 to +180).
        elevation_m: Elevation above sea level in meters (optional).
        optimal_tilt_degrees: Recommended panel tilt angle (typically ≈ latitude).
        optimal_azimuth_degrees: Recommended panel azimuth (180° = south).
        avg_daily_kwh_per_kwp: Monthly average daily production per kWp (12 floats).
        p_sunny: Monthly long-term marginal probability of sunny conditions
            (12 floats, 0-1). When the Markov chain is active this is the
            stationary distribution that the day-by-day simulation will
            preserve for construction.
        weather_persistence: Monthly day-to-day weather persistence factor
            (12 floats, 0-1). Controls the autocorrelation of the sunny/cloudy
            Markov chain:
            - 0.0 = no memory (iid Bernoulli, legacy behaviour)
            - 1.0 = perfect persistence (weather state never flips)
            Typical climatological values for Italy: 0.2–0.5, higher in
            stable summer/winter, lower in changeable shoulder seasons.
            If `None` (legacy records) the simulation falls back to iid.
        sunny_factor: Production multiplier for sunny days (typically 1.2).
        cloudy_factor: Production multiplier for cloudy days (typically 0.3).
        source: Data source attribution (e.g., "PVGIS", "NREL", "measured").
        notes: Additional metadata and comments.

    Example:
        ```python
        profile = SolarProfileModel(
            name="Pavullo_nel_Frignano",
            location_name="Pavullo nel Frignano, Modena, Italy",
            latitude=44.34,
            longitude=10.83,
            elevation_m=682,
            optimal_tilt_degrees=35.0,
            optimal_azimuth_degrees=180.0,
            avg_daily_kwh_per_kwp=[1.46, 2.27, 3.47, 4.42, 5.29, 5.80,
                                    6.28, 5.70, 4.45, 3.06, 1.81, 1.36],
            p_sunny=[0.40, 0.45, 0.50, 0.55, 0.60, 0.70,
                     0.75, 0.70, 0.60, 0.50, 0.40, 0.42],
            sunny_factor=1.2,
            cloudy_factor=0.3,
            source="PVGIS",
            notes="Data extracted from PVGIS for 35° tilt, south-facing"
        )
        ```

    Notes:
        - name is unique constraint (upsert key)
        - avg_daily_kwh_per_kwp and p_sunny must contain exactly 12 values
        - optimal_tilt_degrees typically set to latitude for mid-latitudes
        - Data sourced from PVGIS (https://re.jrc.ec.europa.eu/pvg_tools/en/)
        - Used by SolarModel via SolarProfileRepository
    """
    __tablename__ = "solar_profiles"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)

    # Location metadata
    location_name = Column(String(255), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation_m = Column(Float, nullable=True)
    # Owning site, when the profile was imported through the unified
    # "add location" flow. NULL on legacy/hand-crafted rows.
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=True)

    # Optimal orientation for this location
    optimal_tilt_degrees = Column(Float, nullable=False)
    optimal_azimuth_degrees = Column(Float, nullable=False, default=180.0)

    # Monthly solar production data (JSON columns storing arrays)
    avg_daily_kwh_per_kwp = Column(JSON, nullable=False)  # 12 floats
    p_sunny = Column(JSON, nullable=False)  # 12 floats (0-1)
    # Day-to-day Markov-chain persistence per month (12 floats, 0-1).
    # Nullable to keep retro-compatibility with legacy rows: a NULL value is
    # interpreted as 0.0 (no memory) by the simulator.
    weather_persistence = Column(JSON, nullable=True)
    sunny_factor = Column(Float, nullable=False, default=1.2)
    cloudy_factor = Column(Float, nullable=False, default=0.3)

    # Data source and notes
    source = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)


class ClimateProfileModel(Base, TimestampMixin):
    """
    Database model for stochastic thermal (ambient-temperature) profiles.

    Stores a fully-calibrated :class:`ThermalModel` (Phase 15) for a single
    geographic location, ready to be replayed by the simulator for
    Phase-16 electrical derating and Phase-17 HVAC load coupling.

    Calibration provenance lives in the JSON ``monthly_params`` blob so a
    future change in :class:`ThermalMonthParams` schema can be detected
    and re-fitted instead of silently corrupted.

    Attributes:
        id: Primary key.
        name: Unique short identifier (e.g. ``"Pavullo"``).
        location_name: Human-readable description (Nominatim display_name).
        latitude: Decimal latitude in degrees.
        longitude: Decimal longitude in degrees.
        elevation_m: Elevation in metres (Open-Meteo gridcell), optional.
        source: Provenance string (e.g. ``"OpenMeteo Archive"``).
        harmonic: JSON object with keys ``{"a0", "a1", "a2"}`` carrying
            the deterministic seasonal harmonic coefficients (°C).
        monthly_params: JSON list of 12 objects, one per calendar month
            (index 0 = January). Each entry has keys
            ``{"t_std_residual_c", "persistence_phi", "t_amplitude_c",
            "gpd_upper", "gpd_lower"}`` where the two GPD entries are
            either ``null`` or
            ``{"threshold", "shape", "scale", "exceedance_prob"}``.
        climate_trend_c_per_year: Linear trend (°C/year) applied on top
            of the seasonal mean (0 = stationary climate). Phase 15
            default 0; users opt in by editing the profile.
        lookback_window: JSON object ``{"start_year", "end_year"}`` of
            the archive window used for calibration (audit).
        notes: Free-text metadata (e.g. RMSE of the harmonic fit).

    Notes:
        - The serialization helpers live in
          :mod:`sim_stochastic_pv.persistence.climate_repo` (because
          they import :mod:`simulation.thermal` and we keep DB models
          dependency-light).
        - Linked to :class:`ScenarioRecord` via ``climate_profile_id``
          (introduced as a nullable foreign key in the same Phase-15
          migration). Scenarios without a climate profile preserve the
          pre-Phase-15 behaviour (no thermal model).
    """

    __tablename__ = "climate_profiles"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)

    # Location metadata
    location_name = Column(String(255), nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    elevation_m = Column(Float, nullable=True)
    source = Column(String(255), nullable=True)
    # Owning site, when the profile was calibrated through the unified
    # "add location" flow. NULL on legacy/hand-crafted rows.
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=True)

    # Calibrated model payload
    harmonic = Column(JSON, nullable=False)            # {a0, a1, a2}
    monthly_params = Column(JSON, nullable=False)      # list[12] of dicts
    climate_trend_c_per_year = Column(Float, nullable=False, default=0.0)
    lookback_window = Column(JSON, nullable=True)      # {start_year, end_year}

    notes = Column(Text, nullable=True)


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

    # Phase 12 — soft archive. Default hidden in the Dashboard sidebar;
    # the user can toggle "Mostra archiviati" to bring them back. Set to
    # ``now()`` when archived and back to ``None`` when unarchived.
    archived_at = Column(DateTime, nullable=True)

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


class MarketProfileModel(Base, TimestampMixin):
    """
    Database model for reusable electricity-market profiles.

    Stores a pre-computed wholesale price surface together with the
    dedicated-withdrawal (*ritiro dedicato*) valuation parameters, so that a
    scenario can value its PV export against a simulated market without
    re-running the (expensive) market Monte Carlo on every path. A scenario
    references a profile through a top-level ``market_profile_id`` (or
    ``market_profile_name``) key in its JSON config — mirroring how
    ``climate_profile_id`` wires a :class:`ClimateProfileModel`.

    The whole provider configuration lives in the ``data`` JSON blob in the
    exact shape produced by
    :meth:`sim_stochastic_pv.simulation.market_pricing.MarketPriceProvider.to_config_dict`,
    namely ``{version, pmg_base_eur_per_kwh, retail_markup_fraction,
    retail_fixed_components_eur_per_kwh, surface, build_config}``. Keeping the
    cached surface inside the row (rather than rebuilding it on demand) is a
    deliberate trade-off: the surface costs seconds to compute but only
    kilobytes-to-megabytes to store, and the PV Monte Carlo then performs O(1)
    look-ups against it.

    Attributes:
        id: Primary key (auto-increment).
        name: Unique, human-readable profile identifier (indexed).
        description: Optional free-text description for the editing UI.
        data: Full provider configuration (JSON) — surface + PMG/retail params.

    Example:
        ```python
        profile = MarketProfileModel(
            name="Italia (mercato base)",
            description="Superficie di prezzo all'ingrosso + PMG 0.04 EUR/kWh.",
            data=provider.to_config_dict(build_config={"mix": "italian"}),
        )
        ```

    Notes:
        - ``name`` carries a unique constraint.
        - Hydrated back into a runtime ``MarketPriceProvider`` by
          :class:`sim_stochastic_pv.persistence.market_repo.MarketProfileRepository`.
        - Linked to scenarios via the JSON ``market_profile_id`` key (there is
          no foreign-key column, exactly like ``climate_profile_id``).
    """
    __tablename__ = "market_profiles"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False, index=True)
    description = Column(Text, nullable=True)
    data = Column(JSON, nullable=False)  # serialized surface + pmg/retail config


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
        - "optimization": Multi-scenario optimization search space
    """
    __tablename__ = "saved_configurations"

    id = Column(Integer, primary_key=True)
    name = Column(String(255), unique=True, nullable=False)
    config_type = Column(String(50), nullable=False)  # scenario, optimization
    data = Column(JSON, nullable=False)
