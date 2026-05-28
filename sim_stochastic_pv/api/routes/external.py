"""
External-data API endpoints (Phase 14).

These endpoints delegate to the :mod:`sim_stochastic_pv.external` clients
and surface the responses through clean Pydantic schemas. They power the
wizard "Luogo" step:

- ``POST /api/external/geocode`` — forward-geocode a free-text query.
- ``GET  /api/external/climate-normals`` — monthly climate normals for
  the read-only "Clima locale" preview.
- ``POST /api/profiles/solar/from_location`` — fetch PVGIS' monthly PV
  energy yields for ``(lat, lon, tilt, azimuth, loss)``, derive ``p_sunny``
  from Open-Meteo cloud cover, and persist a :class:`SolarProfileModel`.

Errors raised by the external clients are mapped to ``HTTPException`` 502
so the frontend can show a sane "data source unavailable" notice.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ...external import (
    ExternalAPIError,
    NominatimClient,
    OpenMeteoClient,
    PVGISClient,
)
from ...persistence import PersistenceService
from ...persistence.climate_repo import (
    deserialize_thermal_model,
    serialize_thermal_model,
)
from ...simulation.thermal import simulate_temperature_preview
from ...simulation.thermal_calibration import (
    calibrate_thermal_model,
    samples_from_daily_arrays,
)
from .. import dependencies
from ..schemas.external import (
    ClimateNormalsResponse,
    ClimateProfileFromLocationRequest,
    ClimateProfilePreviewResponse,
    ClimateProfileResponse,
    ClimateProfileUpdate,
    GeocodeRequest,
    GeocodeResultResponse,
    SolarProfileFromLocationRequest,
)
from ..schemas.profiles import SolarProfileResponse


router = APIRouter(prefix="/api", tags=["external"])


# ---------------------------------------------------------------------------
# Geocoding
# ---------------------------------------------------------------------------


@router.post("/external/geocode", response_model=list[GeocodeResultResponse])
def geocode(
    payload: GeocodeRequest,
    client: NominatimClient = Depends(dependencies.get_nominatim_client),
) -> list[GeocodeResultResponse]:
    """
    Forward-geocode a free-text query into up to ``payload.limit`` candidates.

    The endpoint exists primarily to insulate the frontend from the OSMF
    ``User-Agent`` requirement and to centralise potential future caching.

    Args:
        payload: Validated :class:`GeocodeRequest` (query, limit, language).
        client: Nominatim client injected via FastAPI ``Depends``.

    Returns:
        List of :class:`GeocodeResultResponse` ordered by Nominatim's
        ``importance`` desc. May be empty for queries with no match.

    Raises:
        HTTPException 502: Upstream Nominatim returned a non-2xx status or
            an unparseable payload (mapped from :class:`ExternalAPIError`).
    """
    try:
        results = client.search(
            query=payload.query,
            limit=payload.limit,
            accept_language=payload.accept_language,
        )
    except ExternalAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return [
        GeocodeResultResponse(
            display_name=r.display_name,
            latitude=r.latitude,
            longitude=r.longitude,
            place_class=r.place_class,
            place_type=r.place_type,
            importance=r.importance,
        )
        for r in results
    ]


# ---------------------------------------------------------------------------
# Climate normals (preview only — full ClimateProfileModel arrives in Fase 15)
# ---------------------------------------------------------------------------


@router.get(
    "/external/climate-normals",
    response_model=ClimateNormalsResponse,
)
def climate_normals(
    lat: float = Query(..., ge=-90.0, le=90.0, description="Decimal latitude"),
    lon: float = Query(..., ge=-180.0, le=180.0, description="Decimal longitude"),
    lookback_years: int = Query(
        10, ge=1, le=30, description="Archive window length in years"
    ),
    client: OpenMeteoClient = Depends(dependencies.get_openmeteo_client),
) -> ClimateNormalsResponse:
    """
    Return monthly climate normals (tmax/tmin/tmean + sunny probability)
    for a single location, aggregated from the Open-Meteo Archive API.

    Args:
        lat: Decimal latitude.
        lon: Decimal longitude.
        lookback_years: Number of full calendar years to aggregate
            (default 10, capped at 30 to keep payloads reasonable).
        client: Open-Meteo client injected via FastAPI ``Depends``.

    Returns:
        :class:`ClimateNormalsResponse` with 12 monthly normals.

    Raises:
        HTTPException 502: Upstream Open-Meteo returned an error or
            malformed payload.
    """
    try:
        normals = client.fetch_climate_normals(
            latitude=lat,
            longitude=lon,
            lookback_years=lookback_years,
        )
    except ExternalAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return ClimateNormalsResponse(
        latitude=normals.latitude,
        longitude=normals.longitude,
        elevation_m=normals.elevation_m,
        start_year=normals.years_window[0],
        end_year=normals.years_window[1],
        avg_tmax_c=list(normals.avg_tmax_c),
        avg_tmin_c=list(normals.avg_tmin_c),
        avg_tmean_c=list(normals.avg_tmean_c),
        p_sunny=list(normals.p_sunny),
    )


# ---------------------------------------------------------------------------
# Solar profile from location (PVGIS + Open-Meteo orchestration)
# ---------------------------------------------------------------------------


# Defaults used when the upstream PVGIS / Open-Meteo data do not provide a
# direct equivalent. They mirror the seed-data assumptions for Italian
# residential PV (see SolarMonthParams docstring) and are conservative.
DEFAULT_SUNNY_FACTOR = 1.2
DEFAULT_CLOUDY_FACTOR = 0.3
DEFAULT_WEATHER_PERSISTENCE = [0.3] * 12


@router.post(
    "/profiles/solar/from_location",
    response_model=SolarProfileResponse,
)
def create_solar_profile_from_location(
    payload: SolarProfileFromLocationRequest,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
    pvgis: PVGISClient = Depends(dependencies.get_pvgis_client),
    meteo: OpenMeteoClient = Depends(dependencies.get_openmeteo_client),
) -> SolarProfileResponse:
    """
    Build a :class:`SolarProfileModel` from PVGIS + Open-Meteo data.

    Pipeline:

    1. PVGIS PVcalc → 12 monthly PV energy yields for the
       ``(lat, lon, tilt, azimuth, loss)`` combination. Converted to
       ``avg_daily_kwh_per_kwp[m]`` via :meth:`PVGISMonthlyYield.avg_daily_kwh_per_kwp`.
    2. Open-Meteo Archive → 12 monthly cloud-cover means over
       ``lookback_years`` years. Converted to ``p_sunny[m]`` via the
       standard ``1 − cloud_cover_fraction`` approximation.
    3. Persist a new ``SolarProfileModel`` (or upsert if ``payload.overwrite``
       is ``True`` and a profile with the same name already exists). Defaults
       for the weather-Markov parameters are conservative seed values that
       the user can edit afterwards.

    Args:
        payload: Location + geometry + window parameters.
        persistence: Persistence service.
        pvgis: PVGIS client.
        meteo: Open-Meteo client.

    Returns:
        :class:`SolarProfileResponse` of the newly created/updated profile.

    Raises:
        HTTPException 409: A profile with ``payload.name`` already exists
            and ``overwrite`` is ``False``.
        HTTPException 502: Either PVGIS or Open-Meteo returned an error
            or malformed payload.

    Example:
        ```bash
        curl -X POST http://localhost:8000/api/profiles/solar/from_location \\
             -H 'Content-Type: application/json' \\
             -d '{
                   "name": "Pavullo",
                   "location_name": "Pavullo nel Frignano, Modena, Italia",
                   "latitude": 44.336,
                   "longitude": 10.831,
                   "tilt_degrees": 35.0,
                   "azimuth_degrees": 180.0
                 }'
        ```
    """
    # --- Conflict check -----------------------------------------------------
    existing = persistence.solar.get_solar_profile_by_name(payload.name)
    if existing is not None and not payload.overwrite:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Solar profile '{payload.name}' already exists. "
                f"Pass overwrite=true to upsert."
            ),
        )

    # --- PVGIS fetch --------------------------------------------------------
    try:
        yld = pvgis.fetch_monthly_yield(
            latitude=payload.latitude,
            longitude=payload.longitude,
            tilt_degrees=payload.tilt_degrees,
            azimuth_degrees=payload.azimuth_degrees,
            loss_pct=payload.loss_pct,
        )
    except ExternalAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    avg_daily = yld.avg_daily_kwh_per_kwp()

    # --- Open-Meteo fetch for p_sunny seed ---------------------------------
    try:
        normals = meteo.fetch_climate_normals(
            latitude=payload.latitude,
            longitude=payload.longitude,
            lookback_years=payload.lookback_years,
        )
    except ExternalAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    p_sunny = list(normals.p_sunny)

    # --- Persist ------------------------------------------------------------
    notes = (
        f"Auto-created from PVGIS (loss={payload.loss_pct}%) + "
        f"Open-Meteo Archive ({normals.years_window[0]}–{normals.years_window[1]}). "
        f"Edit p_sunny / weather_persistence / factors via the standard profile editor."
    )

    data = {
        "name": payload.name,
        "location_name": payload.location_name,
        "latitude": float(payload.latitude),
        "longitude": float(payload.longitude),
        "elevation_m": yld.elevation_m if yld.elevation_m is not None else normals.elevation_m,
        "optimal_tilt_degrees": float(payload.tilt_degrees),
        "optimal_azimuth_degrees": float(payload.azimuth_degrees),
        "avg_daily_kwh_per_kwp": avg_daily,
        "p_sunny": p_sunny,
        "weather_persistence": list(DEFAULT_WEATHER_PERSISTENCE),
        "sunny_factor": DEFAULT_SUNNY_FACTOR,
        "cloudy_factor": DEFAULT_CLOUDY_FACTOR,
        "source": "PVGIS+OpenMeteo",
        "notes": notes,
    }

    record = persistence.upsert_solar_profile(data)

    return SolarProfileResponse(
        id=record.id,
        name=record.name,
        location_name=record.location_name,
        latitude=record.latitude,
        longitude=record.longitude,
        elevation_m=record.elevation_m,
        optimal_tilt_degrees=record.optimal_tilt_degrees,
        optimal_azimuth_degrees=record.optimal_azimuth_degrees,
        avg_daily_kwh_per_kwp=list(record.avg_daily_kwh_per_kwp),
        p_sunny=list(record.p_sunny),
        weather_persistence=(
            list(record.weather_persistence)
            if record.weather_persistence is not None
            else None
        ),
        sunny_factor=record.sunny_factor,
        cloudy_factor=record.cloudy_factor,
        source=record.source,
        notes=record.notes,
    )


# ---------------------------------------------------------------------------
# Climate profile endpoints (Phase 15 — thermal model)
# ---------------------------------------------------------------------------


def _record_to_climate_response(record) -> ClimateProfileResponse:
    """Project a ``ClimateProfileModel`` row into the response schema."""
    return ClimateProfileResponse(
        id=record.id,
        name=record.name,
        location_name=record.location_name,
        latitude=record.latitude,
        longitude=record.longitude,
        elevation_m=record.elevation_m,
        source=record.source,
        harmonic=dict(record.harmonic),
        monthly_params=list(record.monthly_params),
        climate_trend_c_per_year=record.climate_trend_c_per_year,
        lookback_window=(
            dict(record.lookback_window)
            if record.lookback_window is not None
            else None
        ),
        notes=record.notes,
    )


@router.get(
    "/profiles/climate",
    response_model=list[ClimateProfileResponse],
)
def list_climate_profiles(
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> list[ClimateProfileResponse]:
    """List all climate profiles in the DB ordered by name."""
    records = persistence.climate.list_climate_profiles()
    return [_record_to_climate_response(r) for r in records]


@router.delete("/profiles/climate/{profile_id}", status_code=204)
def delete_climate_profile(
    profile_id: int,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> None:
    """Hard-delete a climate profile. Returns 404 if not found."""
    if not persistence.climate.delete_climate_profile(profile_id):
        raise HTTPException(
            status_code=404,
            detail=f"Climate profile {profile_id} not found",
        )


@router.put(
    "/profiles/climate/{profile_id}",
    response_model=ClimateProfileResponse,
)
def update_climate_profile(
    profile_id: int,
    payload: ClimateProfileUpdate,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> ClimateProfileResponse:
    """
    Update a climate profile by primary key (partial — allows rename).

    Only fields explicitly set in ``payload`` are written. The calibrated
    payload (``harmonic`` / ``monthly_params`` / trend) is editable but
    the normal flow is to recalibrate via ``from_location``.

    Raises:
        HTTPException 404: profile not found.
        HTTPException 409: new ``name`` already used by another profile.
    """
    data = payload.model_dump(exclude_unset=True)
    try:
        record = persistence.update_climate_profile(profile_id, data)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Climate profile {profile_id} not found",
        )
    return _record_to_climate_response(record)


@router.post(
    "/profiles/climate/from_location",
    response_model=ClimateProfileResponse,
)
def create_climate_profile_from_location(
    payload: ClimateProfileFromLocationRequest,
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
    meteo: OpenMeteoClient = Depends(dependencies.get_openmeteo_client),
) -> ClimateProfileResponse:
    """
    Build a :class:`ClimateProfileModel` from Open-Meteo data.

    Pipeline:

    1. Fetch ``lookback_years`` of daily archive (tmean, tmax, tmin) at
       ``(lat, lon)``.
    2. Calibrate a :class:`ThermalModel` via
       :func:`calibrate_thermal_model` — fits the seasonal harmonic, per
       month AR(1) and GPD tails.
    3. Persist the model + audit info in :class:`ClimateProfileModel`.

    Errors:
        409 if ``payload.name`` exists and ``overwrite`` is ``False``.
        502 if Open-Meteo returns non-2xx / malformed data.
    """
    existing = persistence.climate.get_climate_profile_by_name(payload.name)
    if existing is not None and not payload.overwrite:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Climate profile '{payload.name}' already exists. "
                f"Pass overwrite=true to upsert."
            ),
        )

    # 1. Fetch raw daily archive
    try:
        archive = meteo.fetch_daily_archive(
            latitude=payload.latitude,
            longitude=payload.longitude,
            lookback_years=payload.lookback_years,
        )
    except ExternalAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # 2. Calibrate
    samples = samples_from_daily_arrays(
        dates=list(archive.dates),
        t_mean_c=list(archive.t_mean_c),
        t_max_c=list(archive.t_max_c),
        t_min_c=list(archive.t_min_c),
    )
    if not samples:
        raise HTTPException(
            status_code=502,
            detail="Open-Meteo returned no usable daily samples for this location.",
        )

    model, report = calibrate_thermal_model(
        samples,
        climate_trend_c_per_year=payload.climate_trend_c_per_year,
    )
    blob = serialize_thermal_model(model)

    notes = (
        f"Calibrated from Open-Meteo Archive "
        f"({archive.years_window[0]}–{archive.years_window[1]}, "
        f"n={report.n_samples} days). "
        f"Harmonic RMSE = {report.rmse_harmonic_c:.2f} °C. "
        f"Upper-tail GPD fitted for "
        f"{sum(report.per_month_gpd_upper_fitted)}/12 months, "
        f"lower-tail for {sum(report.per_month_gpd_lower_fitted)}/12."
    )

    record = persistence.climate.upsert_climate_profile({
        "name": payload.name,
        "location_name": payload.location_name,
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
        "notes": notes,
    })

    return _record_to_climate_response(record)


@router.get(
    "/profiles/climate/{profile_id}/preview",
    response_model=ClimateProfilePreviewResponse,
)
def preview_climate_profile(
    profile_id: int,
    n_paths: int = Query(50, ge=1, le=200, description="MC paths"),
    n_years: int = Query(1, ge=1, le=20, description="Years per path"),
    seed: int = Query(42, description="Master RNG seed"),
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
) -> ClimateProfilePreviewResponse:
    """
    Fan-chart preview of simulated temperature paths for a saved profile.

    Returns the same payload shape used by the frontend's
    :class:`TemperaturePreview` Svelte component (mean + p05/p95 band +
    sample paths). Cap on ``n_paths`` is 200 to keep payloads light.

    Errors:
        404 if the profile does not exist.
    """
    record = persistence.climate.get_climate_profile_by_id(profile_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Climate profile {profile_id} not found",
        )

    model = deserialize_thermal_model({
        "harmonic": record.harmonic,
        "monthly_params": record.monthly_params,
        "climate_trend_c_per_year": record.climate_trend_c_per_year,
    })
    result = simulate_temperature_preview(
        model,
        n_paths=n_paths,
        n_years=n_years,
        seed=seed,
    )
    return ClimateProfilePreviewResponse(
        days=result.days.tolist(),
        mean_c=result.mean_c.tolist(),
        p05_c=result.p05_c.tolist(),
        p95_c=result.p95_c.tolist(),
        sample_paths_c=result.sample_paths_c.tolist(),
        monthly_p05_c=result.monthly_p05_c.tolist(),
        monthly_p25_c=result.monthly_p25_c.tolist(),
        monthly_p50_c=result.monthly_p50_c.tolist(),
        monthly_p75_c=result.monthly_p75_c.tolist(),
        monthly_p95_c=result.monthly_p95_c.tolist(),
        monthly_min_c=result.monthly_min_c.tolist(),
        monthly_max_c=result.monthly_max_c.tolist(),
    )
