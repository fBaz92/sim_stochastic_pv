"""
Open-Meteo Archive API client for monthly climate normals.

`Open-Meteo <https://open-meteo.com/>`_ exposes a free, no-API-key archive
of ERA5 reanalysis data covering 1940→present at hourly resolution. We use
it to derive **monthly climatological normals** for a chosen location:
average daily max / min / mean temperature and average cloud-cover fraction,
aggregated over a multi-year window (typically the last 10–30 years).

Outputs of this client:

- :class:`ClimateNormals` (12 entries) — used by:
  - the wizard "Luogo" step as a read-only preview of the local climate;
  - the ``solar/from_location`` flow to derive a sensible ``p_sunny[m]``
    from the average cloud-cover fraction (we use the standard
    threshold-free approximation ``p_sunny ≈ 1 − cloud_cover``);
  - Phase 15 (future) as input for the calibration of ``ThermalModel``
    (mean, std, persistence, GPD tails will be fit on the same archive
    window).

API endpoint:
    ``https://archive-api.open-meteo.com/v1/archive``

References:
    https://open-meteo.com/en/docs/historical-weather-api
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import httpx

from .errors import ExternalAPIError

OPENMETEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Default look-back window: 10 calendar years ending at the end of last year.
# 10 years is a reasonable compromise between sampling noise (a single year is
# too noisy) and computational cost (30 years is a few MB of JSON).
DEFAULT_LOOKBACK_YEARS = 10


@dataclass(frozen=True)
class DailyArchive:
    """
    Raw daily-resolution archive series returned by
    :meth:`OpenMeteoClient.fetch_daily_archive`.

    Unlike :class:`ClimateNormals` (which aggregates the series into 12
    monthly normals), this struct preserves the day-by-day data so that
    downstream consumers can compute their own statistics — most notably
    the Phase-15 thermal calibration (:func:`calibrate_thermal_model`
    fits seasonal harmonics, AR(1) residuals and GPD tails on the raw
    daily means).

    Attributes:
        latitude: Gridcell latitude used by Open-Meteo (may differ
            slightly from the request).
        longitude: Gridcell longitude.
        elevation_m: Reported gridcell elevation (m), optional.
        years_window: ``(start_year, end_year)`` of the requested archive
            window. Inclusive.
        dates: ISO-format daily date strings (``"YYYY-MM-DD"``), length
            ``N`` = number of days in the window.
        t_mean_c: Daily mean temperatures (°C). Same length as
            ``dates``. ``None`` entries flag missing days.
        t_max_c: Daily maximum temperatures (°C). Same length, ``None``
            for missing days.
        t_min_c: Daily minimum temperatures (°C). Same length.

    Notes:
        - Missing days surface as ``None`` rather than ``NaN`` so the
          adapter in :func:`samples_from_daily_arrays` can drop them
          cleanly without numpy NaN gymnastics.
        - Length is *not* exactly ``years × 365`` because of leap years;
          downstream code should rely on ``dates`` length, not multiply
          out.
    """

    latitude: float
    longitude: float
    elevation_m: float | None
    years_window: tuple[int, int]
    dates: tuple[str, ...]
    t_mean_c: tuple[float | None, ...]
    t_max_c: tuple[float | None, ...]
    t_min_c: tuple[float | None, ...]


@dataclass(frozen=True)
class ClimateNormals:
    """
    Monthly climatological normals for a single location.

    All arrays have length 12; index 0 = January, index 11 = December.

    Attributes:
        latitude: Latitude actually used (Open-Meteo grid).
        longitude: Longitude actually used.
        elevation_m: Elevation reported for the gridcell (metres). May be
            ``None`` when Open-Meteo does not return it for the response.
        years_window: Tuple ``(start_year, end_year)`` of the archive window
            aggregated (inclusive). Useful for auditability.
        avg_tmax_c: Mean of the **daily maximum** temperature across all
            days in the month (°C).
        avg_tmin_c: Mean of the **daily minimum** temperature (°C).
        avg_tmean_c: Mean of the **daily mean** temperature (°C).
        p_sunny: Approximate marginal probability of a "sunny day" per
            month, derived as ``1 - cloud_cover_fraction`` from the mean
            cloud-cover. Useful as default seed for the weather Markov
            chain in :class:`SolarMonthParams`.

    Example:
        ```python
        with OpenMeteoClient() as client:
            normals = client.fetch_climate_normals(
                latitude=44.34, longitude=10.83,
                lookback_years=10,
            )
        normals.avg_tmax_c[6]  # ~28.5 (Pavullo, July)
        normals.p_sunny[6]     # ~0.7 (clear summer days)
        ```
    """

    latitude: float
    longitude: float
    elevation_m: float | None
    years_window: tuple[int, int]
    avg_tmax_c: tuple[float, ...]
    avg_tmin_c: tuple[float, ...]
    avg_tmean_c: tuple[float, ...]
    p_sunny: tuple[float, ...]


class OpenMeteoClient:
    """
    Synchronous client for Open-Meteo Archive API.

    The client requests *daily* archive data (tmax, tmin, mean cloud-cover)
    for a multi-year window and aggregates locally into 12 monthly normals.
    All HTTP failures are wrapped in :class:`ExternalAPIError`.

    Example:
        ```python
        with OpenMeteoClient() as client:
            normals = client.fetch_climate_normals(
                latitude=45.46, longitude=9.19,  # Milano
                lookback_years=10,
            )
        normals.avg_tmean_c    # 12 floats, °C
        normals.p_sunny        # 12 floats in [0, 1]
        ```
    """

    def __init__(
        self,
        base_url: str = OPENMETEO_ARCHIVE_URL,
        timeout_s: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        """
        Initialize the Open-Meteo Archive client.

        Args:
            base_url: Override the upstream URL (useful for tests).
            timeout_s: Per-request timeout. 60 s default because the
                archive can be slow for long windows.
            client: Optional ``httpx.Client`` to reuse across calls.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    def __enter__(self) -> "OpenMeteoClient":
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _http_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def fetch_daily_archive(
        self,
        latitude: float,
        longitude: float,
        lookback_years: int = DEFAULT_LOOKBACK_YEARS,
        end_year: int | None = None,
    ) -> DailyArchive:
        """
        Fetch raw daily archive data (no aggregation).

        Used by the Phase-15 thermal calibration which needs the full
        day-by-day series to fit harmonics, AR(1) and GPD tails. The
        existing :meth:`fetch_climate_normals` aggregates the same
        upstream call into 12 monthly normals and is kept for
        backward-compatibility with the Fase-14 callers.

        Args:
            latitude: Decimal latitude (-90 to +90).
            longitude: Decimal longitude (-180 to +180).
            lookback_years: Number of full calendar years to include
                (default 10, must be ≥ 1).
            end_year: Last calendar year (inclusive). Defaults to
                ``today.year - 1`` so partial years are excluded.

        Returns:
            :class:`DailyArchive` with parallel date / tmean / tmax / tmin
            arrays.

        Raises:
            ValueError: ``lookback_years`` < 1.
            ExternalAPIError: Upstream non-2xx or malformed JSON.
        """
        if lookback_years < 1:
            raise ValueError(
                f"lookback_years must be >= 1 (got {lookback_years})"
            )
        if end_year is None:
            end_year = dt.date.today().year - 1
        start_year = end_year - lookback_years + 1
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean",
            "timezone": "UTC",
        }

        try:
            response = self._http_client().get(self.base_url, params=params)
        except httpx.HTTPError as exc:
            raise ExternalAPIError(
                provider="openmeteo",
                url=self.base_url,
                message=f"network error: {exc}",
            ) from exc

        if response.status_code != 200:
            raise ExternalAPIError(
                provider="openmeteo",
                url=str(response.url),
                message=f"unexpected status (body: {response.text[:200]})",
                status_code=response.status_code,
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ExternalAPIError(
                provider="openmeteo",
                url=str(response.url),
                message=f"non-JSON response: {exc}",
                status_code=response.status_code,
            ) from exc

        try:
            daily = payload["daily"]
            times = list(daily["time"])
            tmax = list(daily["temperature_2m_max"])
            tmin = list(daily["temperature_2m_min"])
            tmean = list(daily["temperature_2m_mean"])
        except (KeyError, TypeError) as exc:
            raise ExternalAPIError(
                provider="openmeteo",
                url=str(response.url),
                message=f"missing daily.* in payload: {exc}",
            ) from exc

        if not (len(times) == len(tmax) == len(tmin) == len(tmean)):
            raise ExternalAPIError(
                provider="openmeteo",
                url=str(response.url),
                message="daily arrays have mismatched lengths",
            )

        elevation = payload.get("elevation")
        if elevation is not None:
            try:
                elevation = float(elevation)
            except (TypeError, ValueError):
                elevation = None

        def _maybe_float(v: Any) -> float | None:
            if v is None:
                return None
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            # NaN policy: surface as None
            if f != f:  # noqa: PLR0124 - cheap NaN check, no import needed
                return None
            return f

        return DailyArchive(
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation,
            years_window=(start_year, end_year),
            dates=tuple(str(t) for t in times),
            t_mean_c=tuple(_maybe_float(v) for v in tmean),
            t_max_c=tuple(_maybe_float(v) for v in tmax),
            t_min_c=tuple(_maybe_float(v) for v in tmin),
        )

    def fetch_climate_normals(
        self,
        latitude: float,
        longitude: float,
        lookback_years: int = DEFAULT_LOOKBACK_YEARS,
        end_year: int | None = None,
    ) -> ClimateNormals:
        """
        Fetch and aggregate monthly climate normals for a location.

        Args:
            latitude: Decimal latitude (-90 to +90).
            longitude: Decimal longitude (-180 to +180).
            lookback_years: Number of full calendar years to aggregate.
                Must be ≥ 1. Default 10.
            end_year: Last calendar year (inclusive) in the window. Defaults
                to ``today.year - 1`` so we never include a partial current
                year.

        Returns:
            :class:`ClimateNormals` with 12 monthly normals.

        Raises:
            ValueError: ``lookback_years`` < 1.
            ExternalAPIError: Upstream returned non-2xx or malformed JSON.
        """
        if lookback_years < 1:
            raise ValueError(
                f"lookback_years must be >= 1 (got {lookback_years})"
            )

        if end_year is None:
            end_year = dt.date.today().year - 1
        start_year = end_year - lookback_years + 1
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,cloud_cover_mean",
            "timezone": "UTC",
        }

        try:
            response = self._http_client().get(self.base_url, params=params)
        except httpx.HTTPError as exc:
            raise ExternalAPIError(
                provider="openmeteo",
                url=self.base_url,
                message=f"network error: {exc}",
            ) from exc

        if response.status_code != 200:
            raise ExternalAPIError(
                provider="openmeteo",
                url=str(response.url),
                message=f"unexpected status (body: {response.text[:200]})",
                status_code=response.status_code,
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ExternalAPIError(
                provider="openmeteo",
                url=str(response.url),
                message=f"non-JSON response: {exc}",
                status_code=response.status_code,
            ) from exc

        return self._aggregate_monthly(
            payload=payload,
            latitude=latitude,
            longitude=longitude,
            years_window=(start_year, end_year),
        )

    @staticmethod
    def _aggregate_monthly(
        payload: dict[str, Any],
        latitude: float,
        longitude: float,
        years_window: tuple[int, int],
    ) -> ClimateNormals:
        """
        Group the daily archive series into 12 monthly normals.

        The Open-Meteo response shape we care about:

            {
              "latitude": ..., "longitude": ..., "elevation": ...,
              "daily": {
                "time": ["1994-01-01", "1994-01-02", ...],
                "temperature_2m_max":  [...],
                "temperature_2m_min":  [...],
                "temperature_2m_mean": [...],
                "cloud_cover_mean":    [...]
              }
            }

        Raises:
            ExternalAPIError: ``daily`` block missing or arrays mismatched.
        """
        try:
            daily = payload["daily"]
            times: list[str] = daily["time"]
            tmax: list[float | None] = daily["temperature_2m_max"]
            tmin: list[float | None] = daily["temperature_2m_min"]
            tmean: list[float | None] = daily["temperature_2m_mean"]
            ccov: list[float | None] = daily["cloud_cover_mean"]
        except (KeyError, TypeError) as exc:
            raise ExternalAPIError(
                provider="openmeteo",
                url="<parse>",
                message=f"missing daily.* in payload: {exc}",
            ) from exc

        if not (len(times) == len(tmax) == len(tmin) == len(tmean) == len(ccov)):
            raise ExternalAPIError(
                provider="openmeteo",
                url="<parse>",
                message="daily arrays have mismatched lengths",
            )

        elevation = payload.get("elevation")
        if elevation is not None:
            try:
                elevation = float(elevation)
            except (TypeError, ValueError):
                elevation = None

        # Accumulators per month (1..12 → index 0..11)
        sums_tmax: list[float] = [0.0] * 12
        sums_tmin: list[float] = [0.0] * 12
        sums_tmean: list[float] = [0.0] * 12
        sums_ccov: list[float] = [0.0] * 12
        counts: list[int] = [0] * 12

        for ts, tx, tn, tm, cc in zip(times, tmax, tmin, tmean, ccov):
            if tx is None or tn is None or tm is None or cc is None:
                # Skip days where any variable is missing — keeps the mean
                # consistent across variables.
                continue
            try:
                month_idx = int(ts[5:7]) - 1
            except (ValueError, IndexError):
                continue
            if not 0 <= month_idx < 12:
                continue
            sums_tmax[month_idx] += float(tx)
            sums_tmin[month_idx] += float(tn)
            sums_tmean[month_idx] += float(tm)
            sums_ccov[month_idx] += float(cc)
            counts[month_idx] += 1

        # Guard against months with zero valid days (should never happen for
        # archive windows ≥ 1 year, but defensive against partial data).
        avg_tmax = [(s / c) if c > 0 else float("nan") for s, c in zip(sums_tmax, counts)]
        avg_tmin = [(s / c) if c > 0 else float("nan") for s, c in zip(sums_tmin, counts)]
        avg_tmean = [(s / c) if c > 0 else float("nan") for s, c in zip(sums_tmean, counts)]
        avg_ccov_pct = [(s / c) if c > 0 else float("nan") for s, c in zip(sums_ccov, counts)]

        # Cloud cover is reported in percent. Convert to "sunny probability".
        # Bound to [0, 1] to avoid downstream NaN propagation if Open-Meteo
        # ever returns slightly out-of-range values (we have seen +100.0001
        # in some edge cases).
        p_sunny = tuple(
            max(0.0, min(1.0, 1.0 - (cc / 100.0))) for cc in avg_ccov_pct
        )

        return ClimateNormals(
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation,
            years_window=years_window,
            avg_tmax_c=tuple(avg_tmax),
            avg_tmin_c=tuple(avg_tmin),
            avg_tmean_c=tuple(avg_tmean),
            p_sunny=p_sunny,
        )
