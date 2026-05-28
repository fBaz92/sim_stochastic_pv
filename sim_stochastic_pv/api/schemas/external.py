"""
Pydantic schemas for the external-data endpoints (Phase 14).

These schemas cover the API surface introduced to support the wizard
"Luogo" step:

- ``POST /api/external/geocode``: forward-geocoding via Nominatim;
- ``GET  /api/external/climate-normals``: read-only monthly climate
  normals via Open-Meteo (preview for the Luogo step);
- ``POST /api/profiles/solar/from_location``: orchestrates a PVGIS
  fetch + an Open-Meteo cloud-cover lookup to create a
  ``SolarProfileModel`` in one shot.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------


class GeocodeRequest(BaseModel):
    """
    Free-text geocoding query forwarded to Nominatim.

    The request runs server-side so that:

    1. The Nominatim usage policy's ``User-Agent`` requirement is honoured
       consistently regardless of which browser the user is on;
    2. Future caching/rate-limiting layers can live in one place.

    Attributes:
        query: User-typed location string (place name, address, postcode).
            Whitespace-only or empty strings return an empty result list
            without hitting the upstream.
        limit: Max number of candidates to return. Clamped to [1, 10].
        accept_language: Comma-separated BCP-47 language tags used to
            localise ``display_name``. Defaults to Italian then English to
            match the project's primary audience.
    """

    model_config = ConfigDict(json_schema_extra={
        "examples": [{"query": "Pavullo nel Frignano", "limit": 3, "accept_language": "it,en"}]
    })

    query: Annotated[str, Field(min_length=1, max_length=200, description="Free-text place query")]
    limit: Annotated[int, Field(ge=1, le=10, description="Max number of candidates")]= 5
    accept_language: Annotated[str, Field(max_length=64)] = "it,en"


class GeocodeResultResponse(BaseModel):
    """
    One candidate location returned by ``/api/external/geocode``.

    Attributes:
        display_name: Human-readable description (localised per request).
        latitude: Decimal latitude in degrees.
        longitude: Decimal longitude in degrees.
        place_class: Nominatim coarse category (``"place"``, ``"boundary"``).
            Optional — useful to filter address-level matches in the UI.
        place_type: Nominatim fine category (``"city"``, ``"town"``,
            ``"administrative"``).
        importance: Relevance score in [0, 1] (higher is better). Used by
            the UI to sort candidates.
    """

    display_name: str
    latitude: float
    longitude: float
    place_class: str | None = None
    place_type: str | None = None
    importance: float | None = None


# ---------------------------------------------------------------------------
# Climate normals (preview)
# ---------------------------------------------------------------------------


class ClimateNormalsResponse(BaseModel):
    """
    Read-only monthly climate normals for a location.

    Returned by ``GET /api/external/climate-normals?lat=&lon=`` to power
    the "Clima locale" preview in the Luogo step of the wizard. Each array
    has exactly 12 entries (January → December).

    Attributes:
        latitude: Latitude actually queried (Open-Meteo gridcell).
        longitude: Longitude actually queried.
        elevation_m: Reported elevation of the gridcell (metres). Optional.
        start_year: First year (inclusive) of the aggregation window.
        end_year: Last year (inclusive) of the aggregation window.
        avg_tmax_c: 12 monthly means of the daily max temperature (°C).
        avg_tmin_c: 12 monthly means of the daily min temperature (°C).
        avg_tmean_c: 12 monthly means of the daily mean temperature (°C).
        p_sunny: 12 monthly probabilities of a sunny day, approximated as
            ``1 − cloud_cover_fraction`` from the same archive window.
            Useful for the user to gauge expected solar yield variability.
    """

    latitude: float
    longitude: float
    elevation_m: float | None = None
    start_year: int
    end_year: int
    avg_tmax_c: list[float] = Field(..., min_length=12, max_length=12)
    avg_tmin_c: list[float] = Field(..., min_length=12, max_length=12)
    avg_tmean_c: list[float] = Field(..., min_length=12, max_length=12)
    p_sunny: list[float] = Field(..., min_length=12, max_length=12)


# ---------------------------------------------------------------------------
# Solar profile from location
# ---------------------------------------------------------------------------


class SolarProfileFromLocationRequest(BaseModel):
    """
    Request to build a ``SolarProfileModel`` from PVGIS + Open-Meteo data.

    The backend resolves PVGIS' monthly PV energy yields for the
    ``(latitude, longitude, tilt, azimuth, loss)`` combination and uses
    Open-Meteo cloud-cover normals to seed ``p_sunny[m]``. The other
    weather-Markov parameters (``sunny_factor``, ``cloudy_factor``,
    ``weather_persistence``) get sensible defaults that the user can later
    edit through the standard profile editor.

    Attributes:
        name: Unique short identifier under which the profile is stored
            (e.g. ``"Pavullo"``). If a profile with the same name exists,
            the endpoint refuses to overwrite by default and returns 409
            Conflict; use the ``overwrite`` flag to upsert.
        location_name: Human-readable description, typically populated
            from the Nominatim ``display_name`` of the chosen candidate.
        latitude: Decimal latitude.
        longitude: Decimal longitude.
        tilt_degrees: Panel tilt to be used by PVGIS (0–90).
        azimuth_degrees: Panel azimuth in compass convention
            (0=N, 90=E, 180=S, 270=W). Internally converted to PVGIS' aspect.
        loss_pct: System loss percentage assumed by PVGIS. Defaults to 14.
        lookback_years: Open-Meteo aggregation window in years. Defaults
            to 10.
        overwrite: When ``True``, upsert the profile by ``name`` if it
            already exists. Default ``False`` → conflict on collision.
    """

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "name": "Pavullo",
            "location_name": "Pavullo nel Frignano, Modena, Italia",
            "latitude": 44.336,
            "longitude": 10.831,
            "tilt_degrees": 35.0,
            "azimuth_degrees": 180.0,
            "loss_pct": 14.0,
            "lookback_years": 10,
            "overwrite": False,
        }]
    })

    name: Annotated[str, Field(min_length=1, max_length=100)]
    location_name: Annotated[str, Field(min_length=1, max_length=255)]
    latitude: Annotated[float, Field(ge=-90.0, le=90.0)]
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)]
    tilt_degrees: Annotated[float, Field(ge=0.0, le=90.0)] = 35.0
    azimuth_degrees: Annotated[float, Field(ge=0.0, le=360.0)] = 180.0
    loss_pct: Annotated[float, Field(ge=0.0, le=100.0)] = 14.0
    lookback_years: Annotated[int, Field(ge=1, le=30)] = 10
    overwrite: bool = False


# ---------------------------------------------------------------------------
# Climate profile (Phase 15 — thermal model)
# ---------------------------------------------------------------------------


class ClimateProfileFromLocationRequest(BaseModel):
    """
    Request to build a :class:`ClimateProfileModel` from Open-Meteo data
    by fitting a :class:`ThermalModel` to the last ``lookback_years`` of
    daily archive at ``(latitude, longitude)``.

    Attributes:
        name: Unique short identifier.
        location_name: Human-readable description.
        latitude: Decimal latitude.
        longitude: Decimal longitude.
        lookback_years: Archive window used for calibration (default 10).
        climate_trend_c_per_year: Linear trend to bake into the resulting
            model (°C/year, default 0). Phase 15 does not auto-detect a
            trend from the data — the user opts in explicitly.
        overwrite: Upsert if a profile with the same name exists; default
            ``False`` returns 409 instead.
    """

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "name": "Pavullo_climate",
            "location_name": "Pavullo nel Frignano, Modena, Italia",
            "latitude": 44.336,
            "longitude": 10.831,
            "lookback_years": 10,
            "climate_trend_c_per_year": 0.0,
            "overwrite": False,
        }]
    })

    name: Annotated[str, Field(min_length=1, max_length=100)]
    location_name: Annotated[str, Field(min_length=1, max_length=255)]
    latitude: Annotated[float, Field(ge=-90.0, le=90.0)]
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)]
    lookback_years: Annotated[int, Field(ge=1, le=30)] = 10
    climate_trend_c_per_year: Annotated[float, Field(ge=-0.5, le=0.5)] = 0.0
    overwrite: bool = False


class ClimateProfileResponse(BaseModel):
    """
    Climate profile record. Includes a compact summary of the calibrated
    model (no raw daily series — those live upstream at Open-Meteo).

    The full ``monthly_params`` JSON is included so the frontend can
    render per-month diagnostic tables; raw stats (rmse, n_samples) live
    in ``notes`` for now.
    """

    id: int
    name: str
    location_name: str
    latitude: float
    longitude: float
    elevation_m: float | None = None
    source: str | None = None
    harmonic: dict
    monthly_params: list[dict]
    climate_trend_c_per_year: float
    lookback_window: dict | None = None
    notes: str | None = None


class ClimateProfileUpdate(BaseModel):
    """
    Partial-update schema for an existing :class:`ClimateProfileModel`.

    Only fields included in the payload are written. The calibrated
    payload (``harmonic`` / ``monthly_params`` / ``climate_trend_c_per_year``)
    is editable so power users can tweak the model, but the normal flow
    is to recalibrate via the ``from_location`` endpoint.

    Attributes:
        name: Unique short identifier.
        location_name: Human-readable display name.
        latitude: Decimal latitude.
        longitude: Decimal longitude.
        elevation_m: Elevation (m a.s.l.).
        source: Provenance string.
        notes: Free-text metadata.
        climate_trend_c_per_year: Linear trend (°C/year).
        harmonic: Seasonal harmonic coefficients {a0, a1, a2}.
        monthly_params: List of 12 month-level parameter dicts.
        lookback_window: Audit info on the archive window used.
    """

    name: str | None = Field(None, min_length=1)
    location_name: str | None = None
    latitude: float | None = Field(None, ge=-90.0, le=90.0)
    longitude: float | None = Field(None, ge=-180.0, le=180.0)
    elevation_m: float | None = None
    source: str | None = None
    notes: str | None = None
    climate_trend_c_per_year: float | None = None
    harmonic: dict | None = None
    monthly_params: list[dict] | None = None
    lookback_window: dict | None = None


class ClimateProfilePreviewResponse(BaseModel):
    """
    Fan-chart preview of the simulated temperature paths for a profile.

    Two views are returned in the same response so the frontend can
    render them together:

    - **Daily-mean fan chart** (``days``, ``mean_c``, ``p05_c``,
      ``p95_c``, ``sample_paths_c``) — long-term seasonal evolution of
      the *daily mean*.
    - **Per-month hourly distributions** (``monthly_*``) — for each of
      the 12 calendar months, percentiles of the hourly temperatures
      across all paths and all hours. Captures both the diurnal swing
      and the inter-path / extreme-event variability.

    Attributes:
        days: Day-of-horizon indices (0..n_days-1).
        mean_c: Cross-path mean °C per day.
        p05_c: 5th percentile °C per day.
        p95_c: 95th percentile °C per day.
        sample_paths_c: Subset of individual paths (≤ 50) for plotting
            as light grey strokes under the band.
        monthly_p05_c: 12 floats, 5th percentile of hourly temperatures
            per month across all paths.
        monthly_p25_c: 12 floats, 25th percentile.
        monthly_p50_c: 12 floats, median.
        monthly_p75_c: 12 floats, 75th percentile.
        monthly_p95_c: 12 floats, 95th percentile.
        monthly_min_c: 12 floats, hourly min per month (across all paths
            and hours).
        monthly_max_c: 12 floats, hourly max per month.
    """

    days: list[int]
    mean_c: list[float]
    p05_c: list[float]
    p95_c: list[float]
    sample_paths_c: list[list[float]]
    monthly_p05_c: list[float] = Field(..., min_length=12, max_length=12)
    monthly_p25_c: list[float] = Field(..., min_length=12, max_length=12)
    monthly_p50_c: list[float] = Field(..., min_length=12, max_length=12)
    monthly_p75_c: list[float] = Field(..., min_length=12, max_length=12)
    monthly_p95_c: list[float] = Field(..., min_length=12, max_length=12)
    monthly_min_c: list[float] = Field(..., min_length=12, max_length=12)
    monthly_max_c: list[float] = Field(..., min_length=12, max_length=12)
