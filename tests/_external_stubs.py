"""
Shared no-network stubs for the external clients (Nominatim, PVGIS,
Open-Meteo).

Used by ``test_api_external.py`` and ``test_api_locations.py`` through
FastAPI ``dependency_overrides`` so the endpoint tests never hit the public
Internet. Each stub returns deterministic data shaped like the real client
responses; failure variants raise :class:`ExternalAPIError` to exercise the
explicit-error paths of the import flow.
"""

from __future__ import annotations

import datetime as dt
import math

from sim_stochastic_pv.external import (
    ClimateNormals,
    DailyArchive,
    ExternalAPIError,
    GeocodeResult,
    PVGISMonthlyYield,
)


class StubNominatim:
    """Returns a single canned :class:`GeocodeResult` regardless of input."""

    def search(self, query: str, limit: int = 5, accept_language: str = "it,en"):
        return [
            GeocodeResult(
                display_name="Pavullo nel Frignano, Modena, Italia",
                latitude=44.336,
                longitude=10.831,
                place_class="place",
                place_type="town",
                importance=0.71,
            )
        ]


class StubPVGIS:
    """Returns a deterministic :class:`PVGISMonthlyYield`."""

    def fetch_monthly_yield(
        self,
        latitude: float,
        longitude: float,
        tilt_degrees: float,
        azimuth_degrees: float,
        peakpower_kwp: float = 1.0,
        loss_pct: float = 14.0,
        pv_tech: str = "crystSi",
        mounting_place: str = "free",
    ) -> PVGISMonthlyYield:
        e_m = (45, 65, 100, 125, 155, 165, 170, 145, 110, 85, 55, 40)
        h_i = tuple(int(e * 1.05) for e in e_m)
        return PVGISMonthlyYield(
            latitude=latitude,
            longitude=longitude,
            elevation_m=682.0,
            tilt_degrees=tilt_degrees,
            azimuth_degrees=azimuth_degrees,
            loss_pct=loss_pct,
            peakpower_kwp=peakpower_kwp,
            monthly_e_kwh=tuple(float(v) for v in e_m),
            monthly_h_i_kwh_per_m2=tuple(float(v) for v in h_i),
        )


class FailingPVGIS:
    """Raises :class:`ExternalAPIError` on any fetch (upstream outage)."""

    def fetch_monthly_yield(self, *args, **kwargs):
        raise ExternalAPIError(
            "pvgis", "https://stub/pvgis", "unreachable (stubbed outage)"
        )


class StubOpenMeteo:
    """
    Returns 12 constant monthly normals (40% cloud → p_sunny=0.6) for
    :meth:`fetch_climate_normals`, and a synthetic 10-year daily archive
    with a clean seasonal sinusoid for :meth:`fetch_daily_archive`.
    """

    def fetch_climate_normals(
        self,
        latitude: float,
        longitude: float,
        lookback_years: int = 10,
        end_year: int | None = None,
    ) -> ClimateNormals:
        return ClimateNormals(
            latitude=latitude,
            longitude=longitude,
            elevation_m=682.0,
            years_window=(2014, 2023),
            avg_tmax_c=tuple([20.0] * 12),
            avg_tmin_c=tuple([8.0] * 12),
            avg_tmean_c=tuple([14.0] * 12),
            p_sunny=tuple([0.6] * 12),
        )

    def fetch_daily_archive(
        self,
        latitude: float,
        longitude: float,
        lookback_years: int = 10,
        end_year: int | None = None,
    ) -> DailyArchive:
        dates: list[str] = []
        tmean: list[float] = []
        tmax: list[float] = []
        tmin: list[float] = []
        start = dt.date(2014, 1, 1)
        end = dt.date(2023, 12, 31)
        d = start
        while d <= end:
            doy = d.timetuple().tm_yday - 1
            seasonal = 12.0 - 10.0 * math.cos(2 * math.pi * doy / 365.25)
            dates.append(d.isoformat())
            tmean.append(seasonal)
            tmax.append(seasonal + 5.0)
            tmin.append(seasonal - 5.0)
            d += dt.timedelta(days=1)
        return DailyArchive(
            latitude=latitude,
            longitude=longitude,
            elevation_m=682.0,
            years_window=(2014, 2023),
            dates=tuple(dates),
            t_mean_c=tuple(tmean),
            t_max_c=tuple(tmax),
            t_min_c=tuple(tmin),
        )


class ArchiveFailingOpenMeteo(StubOpenMeteo):
    """Normals succeed but the daily archive fetch fails (partial outage)."""

    def fetch_daily_archive(self, *args, **kwargs):
        raise ExternalAPIError(
            "openmeteo",
            "https://stub/openmeteo/archive",
            "archive unreachable (stubbed outage)",
        )
