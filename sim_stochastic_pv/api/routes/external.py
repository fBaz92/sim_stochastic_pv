"""
External-data API endpoints (Phase 14).

These endpoints delegate to the :mod:`sim_stochastic_pv.external` clients
and surface the responses through clean Pydantic schemas. They power the
wizard "Luogo" step:

- ``POST /api/external/geocode`` — forward-geocode a free-text query.
- ``GET  /api/external/climate-normals`` — monthly climate normals for
  the read-only "Clima locale" preview.
- Climate-profile management (list / update / delete / preview).

Profile *creation* from external data lives in the unified location-import
flow: ``POST /api/locations/import`` (see routes/locations.py).

Errors raised by the external clients are mapped to ``HTTPException`` 502
so the frontend can show a sane "data source unavailable" notice.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from ...external import (
    ExternalAPIError,
    NominatimClient,
    OpenMeteoClient,
)
from ...persistence import PersistenceService
from ...persistence.climate_repo import deserialize_thermal_model
from ...simulation.thermal import simulate_temperature_preview
from ...simulation.thermal_validation import backtest_annual_extremes
from .. import dependencies
from ..schemas.external import (
    ClimateExtremesCheckResponse,
    ClimateNormalsResponse,
    ClimateProfilePreviewResponse,
    ClimateProfileResponse,
    ClimateProfileUpdate,
    GeocodeRequest,
    GeocodeResultResponse,
)


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
# Climate profile endpoints (Phase 15 — thermal model)
#
# Profile *creation* lives in the unified location-import flow
# (``POST /api/locations/import`` in routes/locations.py); here we keep the
# management endpoints (list / update / delete / preview).
# ---------------------------------------------------------------------------


def _record_to_climate_response(record) -> ClimateProfileResponse:
    """Project a ``ClimateProfileModel`` row into the response schema."""
    return ClimateProfileResponse.model_validate(record)


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


# Verdict thresholds for the extremes backtest. A p05–p95 band should
# contain ~90% of the observations; we flag below 60% to leave room for
# small-sample noise on 10-year windows. The median-bias gate catches a
# systematically hot/cold model even when the band still covers.
EXTREMES_MIN_COVERAGE = 0.6
EXTREMES_MAX_MEDIAN_BIAS_C = 1.5


def _extremes_verdict(result) -> str:
    """Build the Italian verdict string for an extremes backtest."""
    if len(result.observed_years) < 2:
        return (
            "Archivio osservato insufficiente per il confronto "
            "(servono almeno 2 anni completi)."
        )
    issues: list[str] = []
    if result.tmax_coverage < EXTREMES_MIN_COVERAGE:
        issues.append(
            f"i massimi osservati cadono nella banda simulata solo nel "
            f"{result.tmax_coverage:.0%} degli anni"
        )
    if abs(result.tmax_median_bias_c) > EXTREMES_MAX_MEDIAN_BIAS_C:
        direction = "caldo" if result.tmax_median_bias_c > 0 else "freddo"
        issues.append(
            f"il modello è sistematicamente {direction} sui massimi "
            f"({result.tmax_median_bias_c:+.1f} °C sulla mediana)"
        )
    if result.tmin_coverage < EXTREMES_MIN_COVERAGE:
        issues.append(
            f"i minimi osservati cadono nella banda simulata solo nel "
            f"{result.tmin_coverage:.0%} degli anni"
        )
    if abs(result.tmin_median_bias_c) > EXTREMES_MAX_MEDIAN_BIAS_C:
        direction = "mite" if result.tmin_median_bias_c > 0 else "rigido"
        issues.append(
            f"il modello è sistematicamente {direction} sui minimi "
            f"({result.tmin_median_bias_c:+.1f} °C sulla mediana)"
        )
    if not issues:
        return (
            "OK: il modello riproduce gli estremi annui osservati "
            f"(copertura massimi {result.tmax_coverage:.0%}, "
            f"minimi {result.tmin_coverage:.0%})."
        )
    return "ATTENZIONE: " + "; ".join(issues) + ". Valuta una ricalibrazione."


@router.get(
    "/profiles/climate/{profile_id}/extremes-check",
    response_model=ClimateExtremesCheckResponse,
)
def check_climate_profile_extremes(
    profile_id: int,
    n_paths: int = Query(50, ge=10, le=200, description="MC paths"),
    lookback_years: int = Query(
        10, ge=3, le=30, description="Observed archive window (years)"
    ),
    seed: int = Query(42, description="Master RNG seed"),
    persistence: PersistenceService = Depends(dependencies.get_persistence_service),
    meteo: OpenMeteoClient = Depends(dependencies.get_openmeteo_client),
) -> ClimateExtremesCheckResponse:
    """
    Backtest a saved climate profile against the observed annual extremes.

    Re-fetches the Open-Meteo daily archive at the profile's coordinates
    (so the comparison includes any years recorded *after* calibration),
    simulates the saved model over the same window length with the epoch
    re-anchored at the window start, and returns observed-vs-simulated
    annual extremes plus a human-readable verdict.

    Args:
        profile_id: Primary key of the climate profile.
        n_paths: Simulation paths (default 50, capped at 200 to keep the
            response latency in the seconds range).
        lookback_years: Length of the observed window to fetch.
        seed: Master RNG seed for reproducible bands.
        persistence: Persistence service.
        meteo: Open-Meteo client (stubbed in tests).

    Raises:
        HTTPException 404: profile not found.
        HTTPException 502: Open-Meteo archive unavailable.
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

    try:
        archive = meteo.fetch_daily_archive(
            latitude=record.latitude,
            longitude=record.longitude,
            lookback_years=lookback_years,
        )
    except ExternalAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    result = backtest_annual_extremes(
        model,
        dates=list(archive.dates),
        t_max_c=list(archive.t_max_c),
        t_min_c=list(archive.t_min_c),
        n_paths=n_paths,
        seed=seed,
    )

    return ClimateExtremesCheckResponse(
        observed_years=result.observed_years,
        observed_annual_tmax=result.observed_annual_tmax,
        observed_annual_tmin=result.observed_annual_tmin,
        sim_tmax_p05=result.sim_tmax_p05,
        sim_tmax_p50=result.sim_tmax_p50,
        sim_tmax_p95=result.sim_tmax_p95,
        sim_tmin_p05=result.sim_tmin_p05,
        sim_tmin_p50=result.sim_tmin_p50,
        sim_tmin_p95=result.sim_tmin_p95,
        tmax_coverage=result.tmax_coverage,
        tmin_coverage=result.tmin_coverage,
        tmax_median_bias_c=result.tmax_median_bias_c,
        tmin_median_bias_c=result.tmin_median_bias_c,
        n_paths=result.n_paths,
        n_years=result.n_years,
        climate_trend_c_per_year=record.climate_trend_c_per_year,
        elevation_m=record.elevation_m,
        verdict=_extremes_verdict(result),
    )
