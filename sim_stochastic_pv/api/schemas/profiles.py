"""
Load, price, and solar profile schemas for API validation.

This module contains Pydantic models for electricity and solar profiles:
- Load Profiles: Electricity consumption patterns for homes
- Price Profiles: Electricity pricing models and escalation
- Solar Profiles: Geographic solar irradiance data for PV simulation

Profiles can be saved and reused across multiple scenarios, enabling
consistent modeling and easy comparison of different configurations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class LoadProfileResponse(BaseModel):
    """
    Electricity load profile response schema.

    Represents a stored electricity consumption profile pattern.
    Used when returning load profile data from API GET endpoints.

    Load profiles define how electricity consumption varies over time,
    including daily patterns, seasonal variations, and occupancy effects.

    Attributes:
        id: Unique database identifier.
        name: Human-readable profile name (must be unique).
        profile_type: Type of load profile.
            - "arera": Italian ARERA standard residential profile
            - "custom": Custom monthly average profile (12 values)
            - "custom_24h": Custom hourly profile (12 months × 24 hours)
        data: Profile configuration data as JSON.
            Structure depends on profile_type:
            - arera: Empty dict (uses built-in profile)
            - custom: {"home_profiles_w": [w1, w2, ..., w12]}
            - custom_24h: {"home_profiles_w": [[h1, h2, ..., h24], ...] × 12}

    Example:
        ```python
        # ARERA profile (Italian standard)
        {
            "id": 1,
            "name": "Standard Italian Home",
            "profile_type": "arera",
            "data": {}
        }

        # Custom monthly profile
        {
            "id": 2,
            "name": "Low Consumption Home",
            "profile_type": "custom",
            "data": {
                "home_profiles_w": [200, 180, 190, 210, 230, 250,
                                   270, 260, 240, 220, 200, 190]
            }
        }

        # Custom hourly profile
        {
            "id": 3,
            "name": "Night Worker Profile",
            "profile_type": "custom_24h",
            "data": {
                "home_profiles_w": [
                    [100, 90, 80, ..., 120],  # January (24 hours)
                    [110, 95, 85, ..., 130],  # February (24 hours)
                    ...  # 12 months total
                ],
                "min_days_home": [25, 25, 25, ..., 25]  # Days at home per month
            }
        }
        ```

    Notes:
        - home_profiles_w values are in watts (W)
        - custom profiles can be monthly averages or full hourly patterns
        - Profiles include both "home" and "away" consumption patterns
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str = Field(..., description="Unique profile name")
    profile_type: str = Field(..., description="Profile type: 'arera', 'custom', or 'custom_24h'")
    data: Dict[str, Any] = Field(..., description="Profile configuration (structure varies by type)")


class LoadProfileCreate(BaseModel):
    """
    Schema for creating or updating a load profile.

    Input schema for POST /api/profiles/load endpoint. Creates a new profile
    or updates an existing one (upsert by name).

    Attributes:
        name: Unique profile identifier (required).
        profile_type: Type of profile (required).
            Valid values: "arera", "custom", "custom_24h"
        data: Profile configuration matching the profile_type (required).

    Example:
        ```python
        # POST /api/profiles/load
        {
            "name": "Summer Peak Reducer",
            "profile_type": "custom",
            "data": {
                "home_profiles_w": [
                    220, 200, 210, 230, 250, 300,  # Jan-Jun
                    280, 270, 260, 240, 220, 210   # Jul-Dec
                ],
                "away_profiles_w": [
                    100, 100, 100, 100, 120, 150,
                    140, 130, 120, 110, 100, 100
                ],
                "min_days_home": [26, 24, 26, 25, 26, 24,
                                 28, 28, 26, 26, 24, 26],
                "home_variation_percentiles": [-0.1, 0.1],
                "away_variation_percentiles": [-0.05, 0.05]
            }
        }
        ```

    Raises:
        ValidationError: If profile_type is invalid or data structure doesn't match type.
    """

    name: str = Field(..., min_length=1, description="Unique profile name")
    profile_type: str = Field(..., description="Profile type: 'arera', 'custom', or 'custom_24h'")
    data: Dict[str, Any] = Field(..., description="Profile configuration data")


class LoadProfilePreviewParams(BaseModel):
    """
    Tunable parameters of a load-profile representative-week preview.

    Shared by the saved-profile preview (``/{id}/preview``, where the profile
    shape comes from the DB) and the inline preview
    (:class:`LoadProfilePreviewRequest`, where the shape is in the body).

    Attributes:
        month: Calendar month to preview (0=January … 11=December).
        regime: ``"home"`` (full personality: variability + appliances + HVAC)
            or ``"away"`` (semi-constant + optional variability only).
        climate_profile_id: Optional climate profile id. When set, the weekly
            outdoor temperature is returned and (if the profile enables HVAC)
            the heat-pump load is simulated against that climate.
        n_paths: Monte Carlo paths for the bands. Kept modest for interactivity.
        seed: Master seed for reproducibility.
    """

    model_config = ConfigDict(extra="forbid")

    month: int = Field(0, ge=0, le=11, description="Month 0=Jan … 11=Dec")
    regime: str = Field("home", description="'home' or 'away'")
    climate_profile_id: Optional[int] = Field(
        None, description="Climate profile id for temperature / HVAC overlay"
    )
    n_paths: int = Field(80, ge=1, le=500, description="Monte Carlo paths")
    seed: int = Field(42, description="Master RNG seed")


class LoadProfilePreviewRequest(LoadProfilePreviewParams):
    """
    Inline load-profile preview request (for unsaved edits in the editor).

    Carries the full profile shape in the body so the detail page can preview
    edits live before they are persisted. The ``data`` may include the optional
    ``stochastic`` / ``appliances`` / ``thermal`` blocks alongside the home/away
    patterns.
    """

    profile_type: str = Field(..., description="home_away / custom / custom_24h / weekly / arera")
    data: Dict[str, Any] = Field(..., description="Profile configuration data")


class LoadProfilePreviewResponse(BaseModel):
    """
    Aggregated representative-week preview of one load-profile regime.

    Mirrors :class:`sim_stochastic_pv.simulation.load_preview.LoadPreviewResult`.
    Weekly arrays are length 168 (7 days × 24 h), weekday-major; power in kW,
    energy in kWh, temperature in °C. Temperature fields are ``null`` when no
    climate was supplied; ``temp_in_c_mean`` is ``null`` outside dynamic HVAC.
    """

    regime: str
    month: int
    n_paths: int
    week_hours: List[int]
    total_kw_mean: List[float]
    total_kw_p05: List[float]
    total_kw_p95: List[float]
    baseline_kw_mean: List[float]
    appliance_kw_mean: List[float]
    hvac_kw_mean: List[float]
    annual_kwh_mean: float
    baseline_kwh_annual: float
    appliance_kwh_annual: float
    hvac_kwh_annual_mean: float
    appliance_kwh_annual_by_name: Dict[str, float]
    has_appliances: bool
    has_hvac: bool
    has_thermal: bool
    temp_out_c_mean: Optional[List[float]] = None
    temp_out_c_p05: Optional[List[float]] = None
    temp_out_c_p95: Optional[List[float]] = None
    temp_in_c_mean: Optional[List[float]] = None


class PriceProfileResponse(BaseModel):
    """
    Electricity price profile response schema.

    Represents a stored electricity pricing model including base price
    and escalation parameters. Used when returning price profile data.

    Price profiles model electricity costs over the system lifetime,
    including inflation and market dynamics.

    Attributes:
        id: Unique database identifier.
        name: Human-readable profile name (must be unique).
        data: Price model configuration as JSON.
            Expected fields:
            - base_price_eur_per_kwh: Initial electricity price (EUR/kWh)
            - annual_escalation: Average annual price increase rate (e.g., 0.02 = 2%/year)
            - use_stochastic_escalation: Whether to use stochastic escalation
            - escalation_variation_percentiles: Range of escalation uncertainty (e.g., [-0.05, 0.05])

    Example:
        ```python
        {
            "id": 1,
            "name": "Italian Residential 2025",
            "data": {
                "base_price_eur_per_kwh": 0.22,
                "annual_escalation": 0.025,
                "use_stochastic_escalation": true,
                "escalation_variation_percentiles": [-0.05, 0.05]
            }
        }
        ```

    Notes:
        - base_price_eur_per_kwh should match current market rates
        - annual_escalation typically ranges from 0% to 5% (0.00 to 0.05)
        - Stochastic escalation adds Monte Carlo uncertainty to price evolution
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str = Field(..., description="Unique profile name")
    data: Dict[str, Any] = Field(..., description="Price model configuration")


class PriceProfileCreate(BaseModel):
    """
    Schema for creating or updating a price profile.

    Input schema for POST /api/profiles/price endpoint. Creates a new profile
    or updates an existing one (upsert by name).

    Attributes:
        name: Unique profile identifier (required).
        data: Price model configuration (required).
            Should include all price model parameters.

    Example:
        ```python
        # POST /api/profiles/price
        {
            "name": "Conservative Escalation Model",
            "data": {
                "base_price_eur_per_kwh": 0.20,
                "annual_escalation": 0.015,
                "use_stochastic_escalation": true,
                "escalation_variation_percentiles": [-0.03, 0.03]
            }
        }
        ```

    Raises:
        ValidationError: If name is empty or data is missing required fields.
    """

    name: str = Field(..., min_length=1, description="Unique profile name")
    data: Dict[str, Any] = Field(..., description="Price model configuration")


class SolarProfileUpdate(BaseModel):
    """
    Schema for updating an existing :class:`SolarProfileModel` by ID.

    All fields are optional: only the keys present in the payload are
    written to the underlying record. This lets the UI rename a profile
    (or correct metadata like ``location_name`` or ``notes``) without
    having to resubmit the full PVGIS-derived monthly arrays.

    Attributes:
        name: Short identifier (must remain unique).
        location_name: Human-readable description.
        latitude: Decimal latitude.
        longitude: Decimal longitude.
        elevation_m: Elevation in metres.
        optimal_tilt_degrees: Optimal tilt angle (°).
        optimal_azimuth_degrees: Optimal azimuth (°, 180 = south).
        avg_daily_kwh_per_kwp: Monthly avg daily production (12 floats).
        p_sunny: Monthly probability of sunny day (12 floats, 0-1).
        weather_persistence: Monthly weather persistence factor (12 floats).
        sunny_factor: Production multiplier on sunny days.
        cloudy_factor: Production multiplier on cloudy days.
        source: Provenance attribution.
        notes: Free-text metadata.

    Notes:
        Validation of monthly array lengths is intentionally permissive —
        the wizard step rejects malformed input upstream, and the editor
        in the Database UI presents inline guards.
    """

    name: Optional[str] = Field(None, min_length=1)
    location_name: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90.0, le=90.0)
    longitude: Optional[float] = Field(None, ge=-180.0, le=180.0)
    elevation_m: Optional[float] = None
    optimal_tilt_degrees: Optional[float] = Field(None, ge=0.0, le=90.0)
    optimal_azimuth_degrees: Optional[float] = Field(None, ge=0.0, le=360.0)
    avg_daily_kwh_per_kwp: Optional[List[float]] = None
    p_sunny: Optional[List[float]] = None
    weather_persistence: Optional[List[float]] = None
    sunny_factor: Optional[float] = Field(None, gt=0.0)
    cloudy_factor: Optional[float] = Field(None, ge=0.0)
    source: Optional[str] = None
    notes: Optional[str] = None


class SolarProfileResponse(BaseModel):
    """
    Solar irradiance profile response schema.

    Represents a stored geographic solar profile used to parameterise the
    PV production model for a specific installation location.  The frontend
    wizard reads this endpoint at Step 1 ("Luogo di installazione") to
    populate the location dropdown and to render the read-only monthly
    weather preview.

    Attributes:
        id: Unique database identifier.
        name: Short machine-friendly identifier (e.g. "Milano", "Palermo").
        location_name: Full human-readable description
            (e.g. "Milano, Lombardia, Italy").
        latitude: Decimal latitude in degrees (−90 to +90).
        longitude: Decimal longitude in degrees (−180 to +180).
        elevation_m: Elevation above sea level in metres (optional).
        optimal_tilt_degrees: Recommended panel tilt angle for this location
            (typically close to latitude).  Pre-fills the wizard tilt field.
        optimal_azimuth_degrees: Recommended panel azimuth (180° = south).
        avg_daily_kwh_per_kwp: Monthly average daily production per kWp
            (12 floats, kWh/kWp/day).
        p_sunny: Monthly long-term probability of a sunny day (12 floats, 0–1).
        weather_persistence: Monthly weather-state persistence factor
            (12 floats or None for legacy records).
        sunny_factor: Production multiplier on sunny days (typically 1.2).
        cloudy_factor: Production multiplier on cloudy days (typically 0.3).
        source: Data source attribution (e.g. "PVGIS").
        notes: Free-text metadata.

    Example:
        ```python
        # GET /api/profiles/solar
        [
            {
                "id": 1,
                "name": "Milano",
                "location_name": "Milano, Lombardia, Italy",
                "latitude": 45.46,
                "longitude": 9.19,
                "elevation_m": 120.0,
                "optimal_tilt_degrees": 35.0,
                "optimal_azimuth_degrees": 180.0,
                "avg_daily_kwh_per_kwp": [1.3, 2.1, 3.4, 4.5, 5.3, 5.9,
                                          6.1, 5.6, 4.3, 2.9, 1.6, 1.1],
                "p_sunny": [0.38, 0.42, 0.48, 0.53, 0.58, 0.68,
                            0.73, 0.68, 0.58, 0.48, 0.38, 0.40],
                "weather_persistence": [0.3, 0.25, 0.2, 0.2, 0.2, 0.25,
                                        0.3, 0.3, 0.25, 0.2, 0.25, 0.3],
                "sunny_factor": 1.2,
                "cloudy_factor": 0.3,
                "source": "PVGIS",
                "notes": "35° tilt, south-facing"
            },
            ...
        ]
        ```

    Notes:
        - This endpoint is read-only from the wizard's perspective (profiles
          are seeded or managed via CLI/admin tools).
        - ``avg_daily_kwh_per_kwp`` and ``p_sunny`` always contain exactly 12
          values (one per calendar month, January = index 0).
        - ``weather_persistence`` may be ``None`` for legacy records created
          before Phase 1; the simulator treats ``None`` identically to ``0.0``
          (iid Bernoulli, no weather memory).
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Primary key")
    name: str = Field(..., description="Short identifier (e.g. 'Milano')")
    location_name: str = Field(..., description="Full human-readable location")
    latitude: float = Field(..., description="Decimal latitude (°)")
    longitude: float = Field(..., description="Decimal longitude (°)")
    elevation_m: Optional[float] = Field(None, description="Elevation (m a.s.l.)")
    location_id: Optional[int] = Field(None, description="Owning site (locations.id), None for legacy rows")
    optimal_tilt_degrees: float = Field(..., description="Optimal panel tilt (°)")
    optimal_azimuth_degrees: float = Field(180.0, description="Optimal panel azimuth (°, 180=south)")
    avg_daily_kwh_per_kwp: List[float] = Field(..., description="Monthly avg daily production (kWh/kWp/day, 12 values)")
    p_sunny: List[float] = Field(..., description="Monthly probability of sunny day (0–1, 12 values)")
    weather_persistence: Optional[List[float]] = Field(None, description="Monthly weather persistence factor (0–1, 12 values or None)")
    sunny_factor: float = Field(1.2, description="Production multiplier on sunny days")
    cloudy_factor: float = Field(0.3, description="Production multiplier on cloudy days")
    source: Optional[str] = Field(None, description="Data source attribution")
    notes: Optional[str] = Field(None, description="Free-text metadata")
