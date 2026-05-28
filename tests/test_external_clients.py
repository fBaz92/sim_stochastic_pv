"""
Unit tests for the Phase-14 external API clients.

Tests run with ``httpx.MockTransport`` so they never reach the public
Internet: each ``MockTransport`` callable replays a recorded JSON response
(or asserts the request shape) without leaving the process.
"""

from __future__ import annotations

import httpx
import pytest

from sim_stochastic_pv.external import (
    ClimateNormals,
    ExternalAPIError,
    GeocodeResult,
    NominatimClient,
    OpenMeteoClient,
    PVGISClient,
    PVGISMonthlyYield,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(handler) -> httpx.Client:
    """Build an httpx.Client whose transport calls ``handler(request)``."""
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---------------------------------------------------------------------------
# NominatimClient
# ---------------------------------------------------------------------------


class TestNominatimClient:
    """Forward geocoding via Nominatim."""

    def test_search_returns_parsed_results(self) -> None:
        """A 200 JSON response is parsed into ``GeocodeResult``s."""
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = request.url
            captured["headers"] = dict(request.headers)
            return httpx.Response(
                200,
                json=[
                    {
                        "lat": "44.336",
                        "lon": "10.831",
                        "display_name": "Pavullo nel Frignano, Modena, Italia",
                        "class": "place",
                        "type": "town",
                        "importance": 0.71,
                    },
                ],
            )

        with NominatimClient(client=_make_client(handler)) as client:
            results = client.search("Pavullo nel Frignano", limit=3)

        # Result shape
        assert len(results) == 1
        r = results[0]
        assert isinstance(r, GeocodeResult)
        assert r.latitude == pytest.approx(44.336)
        assert r.longitude == pytest.approx(10.831)
        assert r.place_class == "place"
        assert r.place_type == "town"
        assert r.importance == pytest.approx(0.71)

        # Required headers + query params
        assert captured["headers"].get("user-agent", "").startswith("sim-stochastic-pv/")
        q = dict(captured["url"].params)
        assert q["q"] == "Pavullo nel Frignano"
        assert q["limit"] == "3"
        assert q["format"] == "jsonv2"

    def test_search_returns_empty_for_blank_query(self) -> None:
        """Whitespace-only queries short-circuit and do not hit the network."""
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(200, json=[])

        with NominatimClient(client=_make_client(handler)) as client:
            assert client.search("   ") == []
            assert client.search("") == []

        assert calls == 0, "blank queries must not touch the transport"

    def test_search_wraps_http_error(self) -> None:
        """A non-2xx response is wrapped in ExternalAPIError with the status."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="Service Unavailable")

        with NominatimClient(client=_make_client(handler)) as client:
            with pytest.raises(ExternalAPIError) as exc_info:
                client.search("Roma")

        assert exc_info.value.status_code == 503
        assert exc_info.value.provider == "nominatim"


# ---------------------------------------------------------------------------
# PVGISClient
# ---------------------------------------------------------------------------


class TestPVGISClient:
    """PVGIS PVcalc monthly yield retrieval."""

    @staticmethod
    def _fake_pvgis_payload() -> dict:
        """Minimal PVGIS payload with 12 monthly entries (Jan→Dec)."""
        # Hand-crafted to be realistic for a 1 kWp installation in Northern Italy:
        # sum(E_m) ≈ 1200 kWh/year (PR≈0.86).
        e_m = [45, 65, 100, 125, 155, 165, 170, 145, 110, 85, 55, 40]
        h_i = [int(e * 1.05) for e in e_m]
        return {
            "inputs": {"location": {"elevation": 682.0}},
            "outputs": {
                "monthly": {
                    "fixed": [
                        {"month": m, "E_m": float(e_m[m - 1]), "H(i)_m": float(h_i[m - 1])}
                        for m in range(1, 13)
                    ]
                }
            },
        }

    def test_fetch_returns_monthly_yield_for_south_facing(self) -> None:
        """A normal south-facing request returns 12 monthly values."""
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = request.url
            return httpx.Response(200, json=self._fake_pvgis_payload())

        with PVGISClient(client=_make_client(handler)) as client:
            yld = client.fetch_monthly_yield(
                latitude=44.34,
                longitude=10.83,
                tilt_degrees=35.0,
                azimuth_degrees=180.0,
                peakpower_kwp=1.0,
                loss_pct=14.0,
            )

        # 12 monthly values, monotonically increasing from Jan to summer
        assert isinstance(yld, PVGISMonthlyYield)
        assert len(yld.monthly_e_kwh) == 12
        assert yld.elevation_m == pytest.approx(682.0)
        assert yld.monthly_e_kwh[5] > yld.monthly_e_kwh[0]  # June > January

        # avg_daily_kwh_per_kwp is positive and ranges roughly 1.3 .. 5.5
        daily = yld.avg_daily_kwh_per_kwp()
        assert len(daily) == 12
        assert all(d >= 0 for d in daily)
        assert daily[5] > daily[0]

        # PVGIS azimuth conversion: south compass (180°) → PVGIS aspect 0°
        q = dict(captured["url"].params)
        assert q["aspect"] == "0.0"
        assert q["angle"] == "35.0"

    def test_aspect_converts_compass_to_pvgis_convention(self) -> None:
        """Azimuth 90° (East) maps to PVGIS aspect -90; 270° (West) → +90."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=self._fake_pvgis_payload())

        with PVGISClient(client=_make_client(handler)) as client:
            captured_aspects: list[str] = []

            def capture(request: httpx.Request) -> httpx.Response:
                captured_aspects.append(dict(request.url.params)["aspect"])
                return httpx.Response(200, json=self._fake_pvgis_payload())

            client._client = httpx.Client(transport=httpx.MockTransport(capture))

            client.fetch_monthly_yield(
                latitude=44.0, longitude=10.0,
                tilt_degrees=30.0, azimuth_degrees=90.0,  # East
            )
            client.fetch_monthly_yield(
                latitude=44.0, longitude=10.0,
                tilt_degrees=30.0, azimuth_degrees=270.0,  # West
            )

        assert captured_aspects == ["-90.0", "90.0"]

    def test_invalid_peakpower_raises(self) -> None:
        """A non-positive peakpower trips the input validator before any HTTP."""

        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail("should not hit the network for invalid inputs")

        with PVGISClient(client=_make_client(handler)) as client:
            with pytest.raises(ValueError):
                client.fetch_monthly_yield(
                    latitude=44.0, longitude=10.0,
                    tilt_degrees=30.0, azimuth_degrees=180.0,
                    peakpower_kwp=0.0,
                )

    def test_malformed_payload_raises_external_api_error(self) -> None:
        """Missing outputs.monthly.fixed → ExternalAPIError, not silent fallback."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"inputs": {}, "outputs": {}})

        with PVGISClient(client=_make_client(handler)) as client:
            with pytest.raises(ExternalAPIError) as exc_info:
                client.fetch_monthly_yield(
                    latitude=44.0, longitude=10.0,
                    tilt_degrees=30.0, azimuth_degrees=180.0,
                )

        assert exc_info.value.provider == "pvgis"


# ---------------------------------------------------------------------------
# OpenMeteoClient
# ---------------------------------------------------------------------------


class TestOpenMeteoClient:
    """Climate-normals aggregation from the Open-Meteo Archive API."""

    @staticmethod
    def _fake_archive_payload() -> dict:
        """
        Build a synthetic 2-year daily archive (2020–2021) for a single
        location, with simple seasonal pattern + constant cloud cover.

        We don't try to be physically realistic — only the *aggregation*
        path is under test.
        """
        import datetime as dt

        times: list[str] = []
        tmax: list[float] = []
        tmin: list[float] = []
        tmean: list[float] = []
        ccov: list[float] = []

        start = dt.date(2020, 1, 1)
        end = dt.date(2021, 12, 31)
        d = start
        while d <= end:
            # Simple sinusoidal seasonal pattern
            month = d.month
            base = 10.0 + 12.0 * (1 - abs(month - 7) / 6)  # peak in July
            tmax.append(base + 6.0)
            tmin.append(base - 6.0)
            tmean.append(base)
            ccov.append(40.0)  # 40% cloud → p_sunny ≈ 0.6
            times.append(d.isoformat())
            d += dt.timedelta(days=1)

        return {
            "latitude": 44.34,
            "longitude": 10.83,
            "elevation": 682.0,
            "daily": {
                "time": times,
                "temperature_2m_max": tmax,
                "temperature_2m_min": tmin,
                "temperature_2m_mean": tmean,
                "cloud_cover_mean": ccov,
            },
        }

    def test_fetch_aggregates_into_monthly_normals(self) -> None:
        """A 2-year daily archive collapses to 12 monthly means."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=self._fake_archive_payload())

        with OpenMeteoClient(client=_make_client(handler)) as client:
            normals = client.fetch_climate_normals(
                latitude=44.34, longitude=10.83, lookback_years=2,
            )

        assert isinstance(normals, ClimateNormals)
        assert len(normals.avg_tmax_c) == 12
        assert len(normals.avg_tmin_c) == 12
        assert len(normals.avg_tmean_c) == 12
        assert len(normals.p_sunny) == 12

        # July (index 6) is the warmest month in our synthetic series
        assert normals.avg_tmax_c[6] == max(normals.avg_tmax_c)
        # Cloud cover is uniform → p_sunny is uniform around 0.6
        assert all(abs(p - 0.6) < 1e-6 for p in normals.p_sunny)

    def test_invalid_lookback_years_raises(self) -> None:
        """lookback_years < 1 is rejected before any HTTP."""

        def handler(request: httpx.Request) -> httpx.Response:
            pytest.fail("should not hit the network")

        with OpenMeteoClient(client=_make_client(handler)) as client:
            with pytest.raises(ValueError):
                client.fetch_climate_normals(
                    latitude=44.0, longitude=10.0, lookback_years=0,
                )

    def test_missing_daily_block_raises_external_api_error(self) -> None:
        """A response without ``daily`` → ExternalAPIError."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"latitude": 44.0, "longitude": 10.0})

        with OpenMeteoClient(client=_make_client(handler)) as client:
            with pytest.raises(ExternalAPIError) as exc_info:
                client.fetch_climate_normals(
                    latitude=44.0, longitude=10.0, lookback_years=1,
                )

        assert exc_info.value.provider == "openmeteo"

    def test_fetch_daily_archive_returns_raw_arrays(self) -> None:
        """Phase 15 — daily archive helper used by ThermalModel calibration."""
        # Synthetic 2-year window with simple sinusoidal seasonal pattern.
        payload = self._fake_archive_payload()

        def handler(request: httpx.Request) -> httpx.Response:
            # The daily-archive endpoint doesn't request cloud_cover.
            assert "cloud_cover_mean" not in dict(request.url.params).get("daily", "")
            return httpx.Response(200, json=payload)

        from sim_stochastic_pv.external import DailyArchive  # noqa: PLC0415

        with OpenMeteoClient(client=_make_client(handler)) as client:
            archive = client.fetch_daily_archive(
                latitude=44.34, longitude=10.83, lookback_years=2,
            )

        assert isinstance(archive, DailyArchive)
        n = len(payload["daily"]["time"])
        assert len(archive.dates) == n
        assert len(archive.t_mean_c) == n
        assert len(archive.t_max_c) == n
        assert len(archive.t_min_c) == n
        # First date matches input
        assert archive.dates[0] == payload["daily"]["time"][0]
