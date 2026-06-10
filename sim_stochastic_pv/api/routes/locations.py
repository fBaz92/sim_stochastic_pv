"""
Installation-site API endpoints (``/api/locations``).

A location is the durable anchor of every site-specific dataset: the PVGIS
solar profile and the calibrated Open-Meteo climate profile hang off it via
``location_id`` foreign keys. These endpoints power the unified "Aggiungi
posizione" flow used by both the Database manager and the wizard's Luogo
step:

- ``GET    /api/locations`` — list sites with linked-profile summaries.
- ``POST   /api/locations/import`` — save a site and download its profiles
  in one shot (network first, then a single write transaction; explicit
  per-component errors instead of silent halves).
- ``PUT    /api/locations/{id}`` — rename / edit metadata.
- ``DELETE /api/locations/{id}`` — delete, detaching or deleting profiles.

External-source failures never produce half-written profiles: each profile
is persisted only when its full upstream fetch succeeded, and every skipped
component is reported in the response payload.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ...db.models import ClimateProfileModel, LocationModel, SolarProfileModel
from ...external import (
    ExternalAPIError,
    OpenMeteoClient,
    PVGISClient,
)
from ...persistence import PersistenceService
from ...persistence.climate_repo import serialize_thermal_model
from ...simulation.thermal_calibration import (
    calibrate_thermal_model,
    samples_from_daily_arrays,
)
from .. import dependencies
from ..schemas.external import ClimateProfileResponse
from ..schemas.locations import (
    ClimateProfileSummary,
    LocationImportRequest,
    LocationImportResponse,
    LocationResponse,
    LocationUpdate,
    SolarProfileSummary,
)
from ..schemas.profiles import SolarProfileResponse


router = APIRouter(prefix="/api/locations", tags=["locations"])


# Defaults used when the upstream PVGIS / Open-Meteo data do not provide a
# direct equivalent. They mirror the seed-data assumptions for Italian
# residential PV (see SolarMonthParams docstring) and are conservative.
DEFAULT_SUNNY_FACTOR = 1.2
DEFAULT_CLOUDY_FACTOR = 0.3
DEFAULT_WEATHER_PERSISTENCE = [0.3] * 12


def _location_response(
    record: LocationModel,
    solar: list[SolarProfileModel],
    climate: list[ClimateProfileModel],
) -> LocationResponse:
    """Project a location row + its linked profiles into the response schema."""
    return LocationResponse(
        id=record.id,
        name=record.name,
        address=record.address,
        display_name=record.display_name,
        latitude=record.latitude,
        longitude=record.longitude,
        elevation_m=record.elevation_m,
        notes=record.notes,
        solar_profiles=[SolarProfileSummary.model_validate(r) for r in solar],
        climate_profiles=[ClimateProfileSummary.model_validate(r) for r in climate],
    )


@router.get("", response_model=list[LocationResponse])
def list_locations(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[LocationResponse]:
    """
    List all installation sites with the summaries of their linked profiles.

    The per-site profile summaries (name, source, ``updated_at``) let the
    frontend render the download status of each site — present with a
    freshness date, or missing with a "scarica" call to action — without
    shipping the full monthly payloads.

    Returns:
        Sites ordered by name. Empty list when none exist.
    """
    records = persistence.locations.list_locations()
    out: list[LocationResponse] = []
    for record in records:
        solar, climate = persistence.locations.linked_profiles(record.id)
        out.append(_location_response(record, solar, climate))
    return out


@router.put("/{location_id}", response_model=LocationResponse)
def update_location(
    location_id: int,
    payload: LocationUpdate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> LocationResponse:
    """
    Update a location by primary key (partial — allows rename).

    Editing latitude/longitude does NOT re-download the linked profiles;
    re-import explicitly (``POST /api/locations/import``) to refresh data
    after moving a site.

    Raises:
        HTTPException 404: location not found.
        HTTPException 409: new ``name`` already used by another location.
    """
    data = payload.model_dump(exclude_unset=True)
    try:
        record = persistence.locations.update_location(location_id, data)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"Location {location_id} not found"
        )
    solar, climate = persistence.locations.linked_profiles(record.id)
    return _location_response(record, solar, climate)


@router.delete("/{location_id}", status_code=204)
def delete_location(
    location_id: int,
    delete_profiles: bool = Query(
        False,
        description=(
            "When true, also hard-delete the linked solar/climate profiles. "
            "Default false: profiles survive detached (location_id → NULL)."
        ),
    ),
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> None:
    """
    Delete a location, detaching (default) or deleting its profiles.

    Raises:
        HTTPException 404: location not found.
    """
    if not persistence.locations.delete_location(
        location_id, delete_profiles=delete_profiles
    ):
        raise HTTPException(
            status_code=404, detail=f"Location {location_id} not found"
        )


@router.post("/import", response_model=LocationImportResponse)
def import_location(
    payload: LocationImportRequest,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
    pvgis: PVGISClient = Depends(dependencies.get_pvgis_client),
    meteo: OpenMeteoClient = Depends(dependencies.get_openmeteo_client),
) -> LocationImportResponse:
    """
    Save an installation site and download its data profiles in one shot.

    Pipeline (network first, write last):

    1. When ``include_solar``: fetch PVGIS monthly yields and Open-Meteo
       climate normals (for the ``p_sunny`` seed). A failure here marks
       the solar component as failed — no solar profile is written.
    2. When ``include_climate``: fetch the Open-Meteo daily archive and
       calibrate the stochastic thermal model. A failure marks the
       climate component as failed — no climate profile is written.
    3. Persist location + successfully-fetched profiles in ONE database
       transaction (:meth:`LocationRepository.persist_import`). Profiles
       are upserted by name (= site name) and linked via ``location_id``;
       a same-named legacy profile is adopted by the site.

    The location row is always saved (even when both downloads fail or
    are skipped) so the address itself is never lost; the response makes
    every failure explicit via ``solar_error`` / ``climate_error``.

    Args:
        payload: Site identity + download options.
        persistence: Persistence service.
        pvgis: PVGIS client (injected, stubbed in tests).
        meteo: Open-Meteo client (injected, stubbed in tests).

    Returns:
        :class:`LocationImportResponse` with the saved location, the
        full profile records that were written, and the per-component
        error messages for the ones that were not.
    """
    solar_data: dict | None = None
    climate_data: dict | None = None
    solar_error: str | None = None
    climate_error: str | None = None
    elevation_m: float | None = None

    # --- 1. Solar: PVGIS yields + Open-Meteo normals (no DB writes yet) ----
    if payload.include_solar:
        try:
            yld = pvgis.fetch_monthly_yield(
                latitude=payload.latitude,
                longitude=payload.longitude,
                tilt_degrees=payload.tilt_degrees,
                azimuth_degrees=payload.azimuth_degrees,
                loss_pct=payload.loss_pct,
            )
            normals = meteo.fetch_climate_normals(
                latitude=payload.latitude,
                longitude=payload.longitude,
                lookback_years=payload.lookback_years,
            )
        except ExternalAPIError as exc:
            solar_error = f"Download PVGIS/Open-Meteo fallito: {exc}"
        else:
            elevation_m = (
                yld.elevation_m if yld.elevation_m is not None else normals.elevation_m
            )
            solar_data = {
                "name": payload.name,
                "location_name": payload.display_name or payload.name,
                "latitude": float(payload.latitude),
                "longitude": float(payload.longitude),
                "elevation_m": elevation_m,
                "optimal_tilt_degrees": float(payload.tilt_degrees),
                "optimal_azimuth_degrees": float(payload.azimuth_degrees),
                "avg_daily_kwh_per_kwp": yld.avg_daily_kwh_per_kwp(),
                "p_sunny": list(normals.p_sunny),
                "weather_persistence": list(DEFAULT_WEATHER_PERSISTENCE),
                "sunny_factor": DEFAULT_SUNNY_FACTOR,
                "cloudy_factor": DEFAULT_CLOUDY_FACTOR,
                "source": "PVGIS+OpenMeteo",
                "notes": (
                    f"Auto-created from PVGIS (loss={payload.loss_pct}%) + "
                    f"Open-Meteo Archive "
                    f"({normals.years_window[0]}–{normals.years_window[1]}). "
                    f"Edit p_sunny / weather_persistence / factors via the "
                    f"standard profile editor."
                ),
            }

    # --- 2. Climate: Open-Meteo archive + calibration (no DB writes yet) ---
    if payload.include_climate:
        try:
            archive = meteo.fetch_daily_archive(
                latitude=payload.latitude,
                longitude=payload.longitude,
                lookback_years=payload.lookback_years,
            )
        except ExternalAPIError as exc:
            climate_error = f"Download archivio Open-Meteo fallito: {exc}"
        else:
            samples = samples_from_daily_arrays(
                dates=list(archive.dates),
                t_mean_c=list(archive.t_mean_c),
                t_max_c=list(archive.t_max_c),
                t_min_c=list(archive.t_min_c),
            )
            if not samples:
                climate_error = (
                    "Open-Meteo non ha restituito campioni giornalieri "
                    "utilizzabili per questo luogo."
                )
            else:
                model, report = calibrate_thermal_model(
                    samples,
                    climate_trend_c_per_year=payload.climate_trend_c_per_year,
                )
                blob = serialize_thermal_model(model)
                if elevation_m is None:
                    elevation_m = archive.elevation_m
                climate_data = {
                    "name": payload.name,
                    "location_name": payload.display_name or payload.name,
                    "latitude": float(payload.latitude),
                    "longitude": float(payload.longitude),
                    "elevation_m": archive.elevation_m,
                    "source": "OpenMeteo Archive",
                    "harmonic": blob["harmonic"],
                    "monthly_params": blob["monthly_params"],
                    "climate_trend_c_per_year": blob["climate_trend_c_per_year"],
                    "lookback_window": {
                        "start_year": archive.years_window[0],
                        "end_year": archive.years_window[1],
                    },
                    "notes": (
                        f"Calibrated from Open-Meteo Archive "
                        f"({archive.years_window[0]}–{archive.years_window[1]}, "
                        f"n={report.n_samples} days). "
                        f"Harmonic RMSE = {report.rmse_harmonic_c:.2f} °C. "
                        f"Upper-tail GPD fitted for "
                        f"{sum(report.per_month_gpd_upper_fitted)}/12 months, "
                        f"lower-tail for {sum(report.per_month_gpd_lower_fitted)}/12."
                    ),
                }

    # --- 3. Single write transaction ----------------------------------------
    # Optional fields are only written when known, so a re-import that omits
    # the address (e.g. map-only pick) never erases previously saved values.
    location_data: dict = {
        "name": payload.name,
        "latitude": float(payload.latitude),
        "longitude": float(payload.longitude),
    }
    if payload.address is not None:
        location_data["address"] = payload.address
    if payload.display_name is not None:
        location_data["display_name"] = payload.display_name
    if elevation_m is not None:
        location_data["elevation_m"] = elevation_m
    location, solar_record, climate_record = persistence.locations.persist_import(
        location_data, solar_data=solar_data, climate_data=climate_data
    )

    solar_list, climate_list = persistence.locations.linked_profiles(location.id)
    return LocationImportResponse(
        location=_location_response(location, solar_list, climate_list),
        solar_profile=(
            SolarProfileResponse.model_validate(solar_record)
            if solar_record is not None
            else None
        ),
        climate_profile=(
            ClimateProfileResponse.model_validate(climate_record)
            if climate_record is not None
            else None
        ),
        solar_error=solar_error,
        climate_error=climate_error,
    )
