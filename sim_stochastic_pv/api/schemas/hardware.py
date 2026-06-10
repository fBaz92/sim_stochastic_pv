"""
Hardware component schemas for API validation.

This module contains Pydantic models for hardware components used in
photovoltaic energy system simulations:
- Inverters: DC to AC power conversion equipment
- Solar Panels: Photovoltaic modules for energy generation
- Batteries: Energy storage systems

All schemas support both creation (input) and response (output) operations,
with automatic field population from the `specs` JSON blob for flexibility
and backward compatibility as the schema evolves.

The hardware catalog design allows scenarios to reference hardware by ID
rather than embedding values, ensuring hardware updates propagate to all
scenarios using that hardware.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .common import _coerce_to_dict, _merge_specs_defaults


class InverterResponse(BaseModel):
    """
    Inverter hardware component response schema.

    Represents an inverter (DC to AC power converter) in the hardware catalog.
    Used when returning inverter data from API GET endpoints.

    Inverters are the interface between DC power from solar panels/batteries
    and AC power for household consumption and grid connection.

    Attributes:
        id: Unique database identifier for this inverter.
        name: Human-readable inverter name, must be unique in catalog.
        manufacturer: Manufacturer or brand name (e.g., "Huawei", "SMA").
        model_number: Manufacturer's model number for reference.
        p_ac_max_kw: Maximum AC power output in kilowatts (kW).
        p_dc_max_kw: Maximum DC power input in kW (defaults to p_ac_max_kw if not specified).
        nominal_power_kw: Nominal/rated power in kW for sizing calculations.
        price_eur: Purchase price in EUR (excludes installation).
        install_cost_eur: Installation labor and materials cost in EUR.
        integrated_battery_specs: Specifications for integrated battery systems (optional).
            Some inverters have built-in battery systems (e.g., Tesla Powerwall).
        integrated_battery_price_eur: Price of integrated battery in EUR.
        integrated_battery_count_options: List of available battery counts for
            integrated systems (e.g., [1, 2, 3] for systems supporting 1-3 batteries).
        datasheet: URL or metadata for product datasheet/documentation.
        specs: Full specification data as JSON blob for extensibility.
            Allows storing vendor-specific or future fields without schema migration.

    Example:
        ```python
        # Response from GET /api/inverters
        {
            "id": 1,
            "name": "Huawei SUN2000-5KTL-M1",
            "manufacturer": "Huawei",
            "model_number": "SUN2000-5KTL-M1",
            "p_ac_max_kw": 5.0,
            "p_dc_max_kw": 6.5,
            "nominal_power_kw": 5.0,
            "price_eur": 1200.00,
            "install_cost_eur": 800.00,
            "datasheet": "https://example.com/datasheet.pdf",
            "specs": {
                "efficiency": 98.6,
                "warranty_years": 10,
                "mppt_trackers": 2
            }
        }
        ```

    Notes:
        - Price fields are optional to support catalog entries without pricing
        - Integrated battery specs are only relevant for hybrid inverters
        - The specs JSON blob is auto-merged into top-level fields via validator
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    manufacturer: Optional[str] = None
    model_number: Optional[str] = None
    p_ac_max_kw: Optional[float] = None
    p_dc_max_kw: Optional[float] = None
    nominal_power_kw: Optional[float] = None
    price_eur: Optional[float] = None
    install_cost_eur: Optional[float] = None
    integrated_battery_specs: Optional[Dict[str, Any]] = None
    integrated_battery_price_eur: Optional[float] = None
    integrated_battery_count_options: Optional[List[int]] = None
    # Phase 16 — optional electrical datasheet fields. ``None`` is the
    # backward-compat sentinel ("legacy hardware without MPPT detail"),
    # so the front-end can simply hide the "Modello elettrico
    # dettagliato" panel when these come back unset.
    v_dc_min_v: Optional[float] = None
    v_dc_max_v: Optional[float] = None
    v_mppt_min_v: Optional[float] = None
    v_mppt_max_v: Optional[float] = None
    n_mppt_trackers: Optional[int] = None
    i_dc_max_per_mppt_a: Optional[float] = None
    datasheet: Optional[Any] = None
    specs: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def populate_from_specs(cls, values: Any) -> Dict[str, Any]:
        """
        Auto-populate missing fields from the specs JSON blob.

        Allows flexible schema evolution: new fields can be added to specs
        without schema changes, and will be automatically extracted when present.

        Args:
            values: Raw field values from database or API input.

        Returns:
            Enhanced values with specs fields merged into top-level fields.
        """
        values = _coerce_to_dict(values)
        return _merge_specs_defaults(
            values,
            [
                "p_ac_max_kw",
                "p_dc_max_kw",
                "price_eur",
                "install_cost_eur",
                "integrated_battery_specs",
                "integrated_battery_price_eur",
                "integrated_battery_count_options",
                # Phase 16 — surface electrical specs through the same
                # auto-populate mechanism so DB rows that store them
                # only in the ``specs`` JSON blob still show up at the
                # top level for API consumers.
                "v_dc_min_v",
                "v_dc_max_v",
                "v_mppt_min_v",
                "v_mppt_max_v",
                "n_mppt_trackers",
                "i_dc_max_per_mppt_a",
                "datasheet",
            ],
        )


class InverterCreate(BaseModel):
    """
    Schema for creating or updating an inverter.

    Input schema for POST /api/inverters endpoint. Creates a new inverter
    or updates an existing one (upsert operation based on name matching).

    The upsert behavior allows idempotent API calls: submitting the same
    inverter data multiple times won't create duplicates.

    Attributes:
        name: Unique inverter identifier used for upsert matching (required).
        manufacturer: Manufacturer or brand name.
        model_number: Model number for reference and documentation.
        p_ac_max_kw: Maximum AC output power in kW (required, must be > 0).
        p_dc_max_kw: Maximum DC input power in kW (must be > 0 if specified).
        price_eur: Purchase price in EUR (must be >= 0).
        install_cost_eur: Installation cost in EUR (must be >= 0).
        integrated_battery_specs: Integrated battery specifications.
        integrated_battery_price_eur: Integrated battery price (must be >= 0).
        integrated_battery_count_options: Available battery counts (e.g., [1, 2, 3]).
        datasheet: Product datasheet URL or metadata.
        specs: Additional specifications as JSON for extensibility.

    Example:
        ```python
        # POST /api/inverters
        {
            "name": "Huawei SUN2000-5KTL-M1",
            "manufacturer": "Huawei",
            "model_number": "SUN2000-5KTL-M1",
            "p_ac_max_kw": 5.0,
            "p_dc_max_kw": 6.5,
            "price_eur": 1200.00,
            "install_cost_eur": 800.00,
            "specs": {
                "efficiency": 98.6,
                "warranty_years": 10,
                "mppt_trackers": 2,
                "dimensions_mm": {"width": 365, "height": 565, "depth": 156},
                "weight_kg": 15.5
            }
        }
        ```

    Raises:
        ValidationError: If required fields are missing or constraints are violated.
            - name is required and must not be empty
            - p_ac_max_kw is required and must be positive
            - All price fields must be non-negative if provided
    """

    name: str = Field(..., min_length=1, description="Unique inverter name for identification")
    manufacturer: Optional[str] = Field(None, description="Manufacturer or brand name")
    model_number: Optional[str] = Field(None, description="Manufacturer's model number")
    p_ac_max_kw: float = Field(..., gt=0, description="Maximum AC power output in kW (must be > 0)")
    p_dc_max_kw: Optional[float] = Field(None, gt=0, description="Maximum DC power input in kW (must be > 0)")
    price_eur: Optional[float] = Field(None, ge=0, description="Purchase price in EUR (must be >= 0)")
    install_cost_eur: Optional[float] = Field(None, ge=0, description="Installation cost in EUR (must be >= 0)")
    integrated_battery_specs: Optional[Dict[str, Any]] = Field(None, description="Integrated battery specifications")
    integrated_battery_price_eur: Optional[float] = Field(None, ge=0, description="Integrated battery price in EUR")
    integrated_battery_count_options: Optional[List[int]] = Field(None, description="Available battery counts")
    # Phase 16 — optional electrical datasheet fields (all-or-nothing
    # validation belongs to the simulator's ``validation._validate_electrical``;
    # the CRUD endpoint accepts partial fills so the user can refine
    # them later).
    v_dc_min_v: Optional[float] = Field(None, ge=0, description="Inverter DC operating window: lower bound (V)")
    v_dc_max_v: Optional[float] = Field(None, ge=0, description="Inverter DC operating window: upper bound (V)")
    v_mppt_min_v: Optional[float] = Field(None, ge=0, description="MPPT tracking window: lower bound (V)")
    v_mppt_max_v: Optional[float] = Field(None, ge=0, description="MPPT tracking window: upper bound (V)")
    n_mppt_trackers: Optional[int] = Field(None, ge=1, description="Number of independent MPPT inputs (>=1)")
    i_dc_max_per_mppt_a: Optional[float] = Field(None, ge=0, description="Maximum DC current per MPPT input (A)")
    datasheet: Optional[str | Dict[str, Any]] = Field(None, description="Datasheet URL or metadata")
    specs: Optional[Dict[str, Any]] = Field(None, description="Additional specifications (JSON)")


class PanelResponse(BaseModel):
    """
    Solar panel hardware component response schema.

    Represents a photovoltaic panel module in the hardware catalog.
    Used when returning panel data from API GET endpoints.

    Solar panels convert sunlight into DC electrical power. The power_w
    rating represents the panel's output under Standard Test Conditions (STC):
    1000 W/m² irradiance, 25°C cell temperature, AM1.5 spectrum.

    Attributes:
        id: Unique database identifier for this panel.
        name: Human-readable panel name, must be unique in catalog.
        manufacturer: Manufacturer or brand name (e.g., "Canadian Solar", "JA Solar").
        model_number: Manufacturer's model number.
        power_w: Rated power output in watts (W) under STC conditions.
        price_eur: Purchase price per panel in EUR.
        datasheet: URL or metadata for product datasheet.
        specs: Full specification data as JSON blob for extensibility.

    Example:
        ```python
        # Response from GET /api/panels
        {
            "id": 1,
            "name": "Canadian Solar CS3W-410MS",
            "manufacturer": "Canadian Solar",
            "model_number": "CS3W-410MS",
            "power_w": 410.0,
            "price_eur": 150.00,
            "datasheet": "https://example.com/cs3w-410ms.pdf",
            "specs": {
                "efficiency": 20.7,
                "voc": 48.7,
                "isc": 11.03,
                "dimensions_mm": {"length": 1722, "width": 1134, "thickness": 30},
                "weight_kg": 22.5,
                "warranty_years": 25
            }
        }
        ```

    Notes:
        - power_w is the nameplate rating under STC, actual output varies with conditions
        - Price is per panel, total system cost = price_eur * panel_count
        - specs can store electrical characteristics (Voc, Isc, Vmp, Imp)
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    manufacturer: Optional[str] = None
    model_number: Optional[str] = None
    power_w: Optional[float] = None
    price_eur: Optional[float] = None
    # Phase 16 — optional electrical datasheet fields. ``None`` means
    # the legacy "energy-only" panel (no MPPT/thermal detail). The
    # front-end gates the electrical UI on the presence of these.
    v_oc_stc_v: Optional[float] = None
    v_mpp_stc_v: Optional[float] = None
    i_sc_stc_a: Optional[float] = None
    i_mpp_stc_a: Optional[float] = None
    n_cells_series: Optional[int] = None
    beta_voc_pct_per_c: Optional[float] = None
    gamma_pmax_pct_per_c: Optional[float] = None
    noct_c: Optional[float] = None
    datasheet: Optional[Any] = None
    specs: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def populate_from_specs(cls, values: Any) -> Dict[str, Any]:
        """
        Auto-populate missing fields from the specs JSON blob.

        Args:
            values: Raw field values from database or API input.

        Returns:
            Enhanced values with specs fields merged into top-level fields.
        """
        values = _coerce_to_dict(values)
        return _merge_specs_defaults(
            values,
            [
                "price_eur",
                # Phase 16 — pull electrical datasheet fields from the
                # ``specs`` JSON blob into top-level attributes so API
                # consumers can read them without a second roundtrip.
                "v_oc_stc_v",
                "v_mpp_stc_v",
                "i_sc_stc_a",
                "i_mpp_stc_a",
                "n_cells_series",
                "beta_voc_pct_per_c",
                "gamma_pmax_pct_per_c",
                "noct_c",
                "datasheet",
            ],
        )


class PanelCreate(BaseModel):
    """
    Schema for creating or updating a solar panel.

    Input schema for POST /api/panels endpoint. Creates a new panel
    or updates an existing one (upsert operation based on name matching).

    Attributes:
        name: Unique panel identifier used for upsert matching (required).
        manufacturer: Manufacturer or brand name.
        model_number: Model number for reference.
        power_w: Rated power in watts under STC (required, must be > 0).
        price_eur: Purchase price per panel in EUR (must be >= 0).
        datasheet: Product datasheet URL or metadata.
        specs: Additional specifications as JSON for extensibility.

    Example:
        ```python
        # POST /api/panels
        {
            "name": "Canadian Solar CS3W-410MS",
            "manufacturer": "Canadian Solar",
            "model_number": "CS3W-410MS",
            "power_w": 410.0,
            "price_eur": 150.00,
            "specs": {
                "efficiency": 20.7,
                "cell_type": "Monocrystalline PERC",
                "voc": 48.7,
                "isc": 11.03,
                "vmp": 40.6,
                "imp": 10.10,
                "temp_coeff_pmax": -0.37,
                "dimensions_mm": {"length": 1722, "width": 1134, "thickness": 30}
            }
        }
        ```

    Raises:
        ValidationError: If required fields are missing or constraints are violated.
    """

    name: str = Field(..., min_length=1, description="Unique panel name for identification")
    manufacturer: Optional[str] = Field(None, description="Manufacturer or brand name")
    model_number: Optional[str] = Field(None, description="Manufacturer's model number")
    power_w: float = Field(..., gt=0, description="Rated power in watts under STC (must be > 0)")
    price_eur: Optional[float] = Field(None, ge=0, description="Purchase price per panel in EUR (must be >= 0)")
    # Phase 16 — optional electrical datasheet fields. The CRUD endpoint
    # accepts partial fills; the simulator enforces the all-or-nothing
    # contract via ``validation._validate_electrical`` when the scenario
    # opts into ``electrical.mode='mppt_window'``.
    v_oc_stc_v: Optional[float] = Field(None, ge=0, description="Open-circuit voltage at STC (V)")
    v_mpp_stc_v: Optional[float] = Field(None, ge=0, description="MPP voltage at STC (V)")
    i_sc_stc_a: Optional[float] = Field(None, ge=0, description="Short-circuit current at STC (A)")
    i_mpp_stc_a: Optional[float] = Field(None, ge=0, description="MPP current at STC (A)")
    n_cells_series: Optional[int] = Field(None, ge=1, description="Number of cells wired in series")
    beta_voc_pct_per_c: Optional[float] = Field(
        None, description="V_oc temperature coefficient (%/°C, typically negative)"
    )
    gamma_pmax_pct_per_c: Optional[float] = Field(
        None, description="P_max temperature coefficient (%/°C, typically negative)"
    )
    noct_c: Optional[float] = Field(None, description="Nominal Operating Cell Temperature (°C)")
    datasheet: Optional[str | Dict[str, Any]] = Field(None, description="Datasheet URL or metadata")
    specs: Optional[Dict[str, Any]] = Field(None, description="Additional specifications (JSON)")


class BatteryResponse(BaseModel):
    """
    Battery energy storage system response schema.

    Represents a battery unit in the hardware catalog. Used when returning
    battery data from API GET endpoints.

    Batteries store excess solar energy for later use, enabling increased
    self-consumption and backup power capability.

    Attributes:
        id: Unique database identifier for this battery.
        name: Human-readable battery name, must be unique in catalog.
        manufacturer: Manufacturer or brand name (e.g., "Tesla", "LG Chem").
        model_number: Manufacturer's model number.
        capacity_kwh: Usable energy storage capacity in kilowatt-hours (kWh).
        cycles_life: Number of full charge/discharge cycles before end-of-life.
            End-of-life typically defined as 80% of original capacity.
        price_eur: Purchase price in EUR (excludes installation).
        datasheet: URL or metadata for product datasheet.
        specs: Full specification data as JSON blob for extensibility.

    Example:
        ```python
        # Response from GET /api/batteries
        {
            "id": 1,
            "name": "Tesla Powerwall 2",
            "manufacturer": "Tesla",
            "model_number": "Powerwall 2",
            "capacity_kwh": 13.5,
            "cycles_life": 10000,
            "price_eur": 7000.00,
            "datasheet": "https://example.com/powerwall2.pdf",
            "specs": {
                "chemistry": "Li-ion NMC",
                "voltage_nominal": 50.0,
                "max_charge_power_kw": 5.0,
                "max_discharge_power_kw": 5.0,
                "efficiency": 90.0,
                "dimensions_mm": {"width": 753, "height": 1150, "depth": 147},
                "weight_kg": 114,
                "warranty_years": 10,
                "operating_temp_range": {"min": -20, "max": 50}
            }
        }
        ```

    Notes:
        - capacity_kwh is usable capacity, actual nominal capacity may be higher
        - cycles_life assumes full depth-of-discharge, partial cycles scale proportionally
        - specs can include chemistry, voltage, power limits, efficiency
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    manufacturer: Optional[str] = None
    model_number: Optional[str] = None
    capacity_kwh: Optional[float] = None
    cycles_life: Optional[int] = None
    price_eur: Optional[float] = None
    datasheet: Optional[Any] = None
    specs: Optional[Dict[str, Any]] = None

    @model_validator(mode="before")
    @classmethod
    def populate_from_specs(cls, values: Any) -> Dict[str, Any]:
        """
        Auto-populate missing fields from the specs JSON blob.

        Args:
            values: Raw field values from database or API input.

        Returns:
            Enhanced values with specs fields merged into top-level fields.
        """
        values = _coerce_to_dict(values)
        return _merge_specs_defaults(values, ["capacity_kwh", "cycles_life", "price_eur", "datasheet"])


class BatteryCreate(BaseModel):
    """
    Schema for creating or updating a battery.

    Input schema for POST /api/batteries endpoint. Creates a new battery
    or updates an existing one (upsert operation based on name matching).

    Attributes:
        name: Unique battery identifier used for upsert matching (required).
        manufacturer: Manufacturer or brand name.
        model_number: Model number for reference.
        capacity_kwh: Usable capacity in kWh (required, must be > 0).
        cycles_life: Cycle life before end-of-life (must be > 0 if specified).
        price_eur: Purchase price in EUR (must be >= 0).
        datasheet: Product datasheet URL or metadata.
        specs: Additional specifications as JSON for extensibility.

    Example:
        ```python
        # POST /api/batteries
        {
            "name": "Tesla Powerwall 2",
            "manufacturer": "Tesla",
            "model_number": "Powerwall 2",
            "capacity_kwh": 13.5,
            "cycles_life": 10000,
            "price_eur": 7000.00,
            "specs": {
                "chemistry": "Li-ion NMC",
                "voltage_nominal": 50.0,
                "max_charge_power_kw": 5.0,
                "max_discharge_power_kw": 5.0,
                "round_trip_efficiency": 90.0,
                "warranty_years": 10,
                "depth_of_discharge_max": 100.0
            }
        }
        ```

    Raises:
        ValidationError: If required fields are missing or constraints are violated.
    """

    name: str = Field(..., min_length=1, description="Unique battery name for identification")
    manufacturer: Optional[str] = Field(None, description="Manufacturer or brand name")
    model_number: Optional[str] = Field(None, description="Manufacturer's model number")
    capacity_kwh: float = Field(..., gt=0, description="Usable capacity in kWh (must be > 0)")
    cycles_life: Optional[int] = Field(None, gt=0, description="Cycle life before end-of-life (must be > 0)")
    price_eur: Optional[float] = Field(None, ge=0, description="Purchase price in EUR (must be >= 0)")
    datasheet: Optional[str | Dict[str, Any]] = Field(None, description="Datasheet URL or metadata")
    specs: Optional[Dict[str, Any]] = Field(None, description="Additional specifications (JSON)")


class CableResponse(BaseModel):
    """
    DC cable catalogue row (electrical-designer catalogue).

    Attributes:
        id: Primary key.
        name: Unique catalogue identifier (e.g. "Cavo solare H1Z2Z2-K 6mm2").
        manufacturer: Brand, optional.
        section_mm2: Conductor cross-section (mm²).
        material: Conductor material ("copper" default).
        price_eur_per_m: List price per metre (EUR), optional.
        iz_a: Thermal current rating in free air (A), optional — gates the
            designer's recommended-section pick.
        notes: Free text.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    manufacturer: Optional[str] = None
    section_mm2: float
    material: str = "copper"
    price_eur_per_m: Optional[float] = None
    iz_a: Optional[float] = None
    notes: Optional[str] = None


class CableCreate(BaseModel):
    """
    Create/upsert payload for a DC cable catalogue row.

    POST /api/cables upserts by unique ``name``.
    """

    name: str = Field(..., min_length=1, description="Unique cable name (upsert key)")
    manufacturer: Optional[str] = Field(None, description="Brand")
    section_mm2: float = Field(..., gt=0, description="Conductor cross-section (mm²)")
    material: str = Field("copper", description="Conductor material")
    price_eur_per_m: Optional[float] = Field(None, ge=0, description="Price per metre (EUR)")
    iz_a: Optional[float] = Field(None, gt=0, description="Thermal current rating Iz (A)")
    notes: Optional[str] = Field(None, description="Free text")


class ProtectionResponse(BaseModel):
    """
    DC protection catalogue row (fuse / breaker / disconnector / SPD).

    Attributes:
        id: Primary key.
        name: Unique catalogue identifier.
        manufacturer: Brand, optional.
        kind: Device family ("fuse", "breaker", "disconnector", "spd").
        rated_current_a: Nominal current I_n (A); None for SPDs.
        rated_voltage_v: Rated DC voltage (V).
        price_eur: List price (EUR), optional.
        specs: Family-specific JSON blob, optional.
        notes: Free text.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    manufacturer: Optional[str] = None
    kind: str = "fuse"
    rated_current_a: Optional[float] = None
    rated_voltage_v: Optional[float] = None
    price_eur: Optional[float] = None
    specs: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class ProtectionCreate(BaseModel):
    """
    Create/upsert payload for a DC protection catalogue row.

    POST /api/protections upserts by unique ``name``.
    """

    name: str = Field(..., min_length=1, description="Unique protection name (upsert key)")
    manufacturer: Optional[str] = Field(None, description="Brand")
    kind: str = Field("fuse", description="fuse | breaker | disconnector | spd")
    rated_current_a: Optional[float] = Field(None, gt=0, description="Nominal current (A)")
    rated_voltage_v: Optional[float] = Field(None, gt=0, description="Rated DC voltage (V)")
    price_eur: Optional[float] = Field(None, ge=0, description="List price (EUR)")
    specs: Optional[Dict[str, Any]] = Field(None, description="Family-specific data (JSON)")
    notes: Optional[str] = Field(None, description="Free text")
