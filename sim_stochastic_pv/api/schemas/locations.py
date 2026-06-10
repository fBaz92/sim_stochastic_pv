"""
Pydantic schemas for the installation-site endpoints (``/api/locations``).

These schemas cover the unified "add location" flow: a single request
carries the geocoded site plus the download options, and a single response
reports the saved location together with the per-component outcome of the
PVGIS / Open-Meteo imports (explicit errors instead of silent halves).
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from .external import ClimateProfileResponse
from .profiles import SolarProfileResponse


class SolarProfileSummary(BaseModel):
    """
    Compact view of a solar profile linked to a location.

    Used inside :class:`LocationResponse` so the Database page can show
    the download status ("scaricato il …") without shipping the full
    12-month payload for every site in the list.

    Attributes:
        id: Primary key of the solar profile.
        name: Unique profile name.
        source: Provenance string (e.g. ``"PVGIS+OpenMeteo"``).
        optimal_tilt_degrees: Tilt used for the PVGIS import (°).
        optimal_azimuth_degrees: Azimuth used for the import (°, 180=south).
        updated_at: Timestamp of the last write (download/refresh), used
            by the UI as "aggiornato al".
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    source: str | None = None
    optimal_tilt_degrees: float
    optimal_azimuth_degrees: float
    updated_at: datetime | None = None


class ClimateProfileSummary(BaseModel):
    """
    Compact view of a climate profile linked to a location.

    Attributes:
        id: Primary key of the climate profile.
        name: Unique profile name.
        source: Provenance string (e.g. ``"OpenMeteo Archive"``).
        lookback_window: Audit dict ``{"start_year", "end_year"}`` of the
            calibration archive, or ``None``.
        updated_at: Timestamp of the last calibration.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    source: str | None = None
    lookback_window: dict | None = None
    updated_at: datetime | None = None


class LocationResponse(BaseModel):
    """
    Installation site with the summaries of its linked profiles.

    Returned by ``GET /api/locations`` (list) and embedded in
    :class:`LocationImportResponse`. The profile lists let the UI render
    a per-site status: present (with freshness date) vs missing (with a
    "download" call to action).

    Attributes:
        id: Primary key.
        name: Unique short identifier (upsert key).
        address: Free-text query typed by the user, when available.
        display_name: Canonical geocoder address, when available.
        latitude: Decimal latitude (°).
        longitude: Decimal longitude (°).
        elevation_m: Elevation (m a.s.l.), when known.
        notes: Free-text user notes.
        solar_profiles: Linked solar profiles (possibly empty).
        climate_profiles: Linked climate profiles (possibly empty).
    """

    id: int
    name: str
    address: str | None = None
    display_name: str | None = None
    latitude: float
    longitude: float
    elevation_m: float | None = None
    notes: str | None = None
    solar_profiles: list[SolarProfileSummary] = Field(default_factory=list)
    climate_profiles: list[ClimateProfileSummary] = Field(default_factory=list)


class LocationUpdate(BaseModel):
    """
    Partial-update schema for an existing location (rename, notes, …).

    Only fields explicitly present in the payload are written. Moving a
    site (lat/lon) is allowed but does NOT re-download the profiles —
    the user re-imports explicitly to refresh the data.
    """

    name: str | None = Field(None, min_length=1, max_length=100)
    address: str | None = None
    display_name: str | None = None
    latitude: float | None = Field(None, ge=-90.0, le=90.0)
    longitude: float | None = Field(None, ge=-180.0, le=180.0)
    notes: str | None = None


class LocationImportRequest(BaseModel):
    """
    One-shot request to save a site and download its data profiles.

    This is the single entry point of the "Aggiungi posizione" flow (both
    the Database manager and the wizard's Luogo step). The backend fetches
    every requested external dataset *before* writing anything, then
    persists location + profiles in one transaction, so a failure can
    never leave the address saved without its data silently missing —
    every skipped component is reported in the response.

    Attributes:
        name: Unique short identifier of the site; also used as the name
            of the linked solar/climate profiles (upsert key for all
            three). Re-importing with the same name refreshes the data.
        address: Free-text query typed by the user (optional, audit).
        display_name: Geocoder display name (optional).
        latitude: Decimal latitude.
        longitude: Decimal longitude.
        include_solar: Download the PVGIS solar profile (default True).
        include_climate: Calibrate the stochastic climate profile from
            the Open-Meteo daily archive (default True).
        tilt_degrees: Panel tilt for PVGIS (0–90, default 35).
        azimuth_degrees: Panel azimuth, compass convention (180 = south).
        loss_pct: PVGIS system-loss assumption in percent (default 14).
        lookback_years: Archive window for Open-Meteo aggregations and
            climate calibration (default 10).
        climate_trend_c_per_year: Linear warming trend baked into the
            calibrated climate model (°C/year, default 0).
    """

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "name": "Pavullo",
            "display_name": "Pavullo nel Frignano, Modena, Italia",
            "latitude": 44.336,
            "longitude": 10.831,
            "include_solar": True,
            "include_climate": True,
            "tilt_degrees": 35.0,
            "azimuth_degrees": 180.0,
        }]
    })

    name: Annotated[str, Field(min_length=1, max_length=100)]
    address: Annotated[str | None, Field(max_length=500)] = None
    display_name: Annotated[str | None, Field(max_length=500)] = None
    latitude: Annotated[float, Field(ge=-90.0, le=90.0)]
    longitude: Annotated[float, Field(ge=-180.0, le=180.0)]
    include_solar: bool = True
    include_climate: bool = True
    tilt_degrees: Annotated[float, Field(ge=0.0, le=90.0)] = 35.0
    azimuth_degrees: Annotated[float, Field(ge=0.0, le=360.0)] = 180.0
    loss_pct: Annotated[float, Field(ge=0.0, le=100.0)] = 14.0
    lookback_years: Annotated[int, Field(ge=1, le=30)] = 10
    climate_trend_c_per_year: Annotated[float, Field(ge=-0.5, le=0.5)] = 0.0


class LocationImportResponse(BaseModel):
    """
    Outcome of a location import, with explicit per-component status.

    The location itself is always saved when this response is returned
    (a hard failure raises 4xx/5xx instead). Each requested profile is
    either present (full record) or absent with the corresponding error
    message populated — never silently missing.

    Attributes:
        location: The saved site, including linked-profile summaries.
        solar_profile: Full solar profile record, or ``None`` when the
            download was skipped (``include_solar=False``) or failed.
        climate_profile: Full climate profile record, or ``None`` when
            skipped or failed.
        solar_error: Human-readable reason why the solar download failed
            (``None`` on success or skip).
        climate_error: Human-readable reason why the climate calibration
            failed (``None`` on success or skip).
    """

    location: LocationResponse
    solar_profile: SolarProfileResponse | None = None
    climate_profile: ClimateProfileResponse | None = None
    solar_error: str | None = None
    climate_error: str | None = None
