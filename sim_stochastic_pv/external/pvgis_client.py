"""
European Commission JRC PVGIS v5.2 PVcalc client.

`PVGIS <https://re.jrc.ec.europa.eu/pvg_tools/en/>`_ is the European
Commission Joint Research Centre tool for PV resource assessment. The
``PVcalc`` endpoint returns the *expected monthly energy output* of a
grid-connected PV system at the requested location, including all the
geometry corrections (tilt, azimuth, near-horizon shading, beam +
diffuse + reflected components), the temperature-dependent module
efficiency, and a user-supplied system loss percentage.

We use it as the **physical baseline** for the solar profile in the
scenario: for each (latitude, longitude, tilt, azimuth, PR) we receive
12 monthly energy yields (kWh) that get translated into
``avg_daily_kwh_per_kwp[m]`` values used by :class:`SolarMonthParams`.

References:
    - API reference: https://joint-research-centre.ec.europa.eu/pvgis-online-tool/getting-started-pvgis/api-non-interactive-service_en
    - Endpoint: ``https://re.jrc.ec.europa.eu/api/v5_2/PVcalc``

Important: PVGIS uses an **azimuth convention different from this project**.

- In the project, azimuth is in [0°, 360°]:
  ``0° = North, 90° = East, 180° = South, 270° = West``
  (compass bearing).
- PVGIS' ``aspect`` parameter is centred on south:
  ``-90° = East, 0° = South, +90° = West``.

The conversion happens in :meth:`PVGISClient.fetch_monthly_yield`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .errors import ExternalAPIError

PVGIS_BASE_URL = "https://re.jrc.ec.europa.eu/api/v5_2"

# Default Performance Ratio assumed in PVGIS PVcalc.
# We expose it as the ``loss`` parameter (percent) of PVGIS itself, so for
# convenience we also expose a "PR" wrapper: PR = 1 - loss/100.
DEFAULT_PVGIS_LOSS_PCT = 14.0  # equivalent to PR ~ 0.86 (industry baseline)

# Reference PV technology used by PVGIS for the efficiency curves.
# "crystSi" covers mono- and polycrystalline silicon (>95% of residential
# installations). Other options: "CIS", "CdTe", "Unknown".
DEFAULT_PV_TECH = "crystSi"


@dataclass(frozen=True)
class PVGISMonthlyYield:
    """
    Monthly PV energy yield returned by PVGIS PVcalc.

    Attributes:
        latitude: Latitude actually used (PVGIS rounds to grid).
        longitude: Longitude actually used.
        elevation_m: Elevation reported by PVGIS for the gridcell (metres).
        tilt_degrees: Panel tilt that was queried.
        azimuth_degrees: Panel azimuth in the project's convention
            (0=N, 180=S) that was queried.
        loss_pct: System loss percentage assumed (the ``loss`` PVGIS param).
        peakpower_kwp: Nameplate DC power used in the request (kWp).
            Useful for back-converting the energy yields to per-kWp values.
        monthly_e_kwh: 12 monthly PV energy yields (kWh). Index 0 = January,
            index 11 = December.
        monthly_h_i_kwh_per_m2: 12 monthly in-plane irradiation values
            (kWh/m²) — useful for QA and downstream physics.

    Notes:
        - ``monthly_e_kwh`` already includes panel/inverter losses; the
          consumer should NOT further derate.
        - To obtain ``kWh/kWp/day`` (the unit used by
          :class:`SolarMonthParams`):

              avg_daily_kwh_per_kwp[m] = monthly_e_kwh[m] / days_in_month[m] / peakpower_kwp

          See :meth:`avg_daily_kwh_per_kwp`.
    """

    latitude: float
    longitude: float
    elevation_m: float | None
    tilt_degrees: float
    azimuth_degrees: float
    loss_pct: float
    peakpower_kwp: float
    monthly_e_kwh: tuple[float, ...]
    monthly_h_i_kwh_per_m2: tuple[float, ...]

    # Average number of days per calendar month used for kWh→kWh/day
    # conversion. Using 30.4375 across the board (365.25/12) is good enough
    # for an economic simulation and avoids quirks at the year boundary.
    _DAYS_IN_MONTH: tuple[float, ...] = (
        31.0, 28.25, 31.0, 30.0, 31.0, 30.0,
        31.0, 31.0, 30.0, 31.0, 30.0, 31.0,
    )

    def avg_daily_kwh_per_kwp(self) -> list[float]:
        """
        Translate the 12 monthly energies into the
        ``avg_daily_kwh_per_kwp[m]`` list consumed by
        :class:`SolarMonthParams`.

        Returns:
            list[float]: 12 values, units ``kWh/kWp/day``.

        Example:
            ```python
            yld = client.fetch_monthly_yield(lat=44.34, lon=10.83,
                                             tilt=35, azimuth=180)
            daily = yld.avg_daily_kwh_per_kwp()
            # daily[5] ≈ 5.8 (June, kWh/kWp/day for Pavullo nel Frignano)
            ```
        """
        if self.peakpower_kwp <= 0:
            raise ValueError("peakpower_kwp must be > 0 for normalization")
        return [
            e_kwh / days / self.peakpower_kwp
            for e_kwh, days in zip(self.monthly_e_kwh, self._DAYS_IN_MONTH)
        ]


class PVGISClient:
    """
    Synchronous client for the PVGIS v5.2 ``PVcalc`` endpoint.

    The client converts the project's azimuth convention to PVGIS' convention
    transparently. PR is configurable via the ``loss_pct`` parameter (which
    PVGIS internally applies to the energy output).

    Example:
        ```python
        with PVGISClient() as client:
            yld = client.fetch_monthly_yield(
                latitude=44.34, longitude=10.83,
                tilt_degrees=35.0, azimuth_degrees=180.0,
                loss_pct=14.0,
            )
            daily = yld.avg_daily_kwh_per_kwp()
            print(daily)  # 12 floats, kWh/kWp/day
        ```
    """

    def __init__(
        self,
        base_url: str = PVGIS_BASE_URL,
        timeout_s: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        """
        Initialize the PVGIS client.

        Args:
            base_url: Override the upstream URL (useful for tests).
            timeout_s: Per-request timeout (PVGIS is slow under load: 30 s
                default).
            client: Optional ``httpx.Client`` to reuse across calls.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    def __enter__(self) -> "PVGISClient":
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

    def fetch_monthly_yield(
        self,
        latitude: float,
        longitude: float,
        tilt_degrees: float,
        azimuth_degrees: float,
        peakpower_kwp: float = 1.0,
        loss_pct: float = DEFAULT_PVGIS_LOSS_PCT,
        pv_tech: str = DEFAULT_PV_TECH,
        mounting_place: str = "free",
    ) -> PVGISMonthlyYield:
        """
        Fetch the 12 monthly PV energy yields for a specific location/geometry.

        Args:
            latitude: Decimal latitude (-90 to +90).
            longitude: Decimal longitude (-180 to +180).
            tilt_degrees: Panel tilt from horizontal (0–90).
            azimuth_degrees: Panel azimuth in compass convention
                (0=North, 90=East, 180=South, 270=West). Internally
                converted to PVGIS' aspect (0=S, ±90=E/W).
            peakpower_kwp: Nameplate DC peak power in kWp used for the
                request. The returned ``monthly_e_kwh`` scales linearly with
                this parameter, so the conversion to ``kWh/kWp/day`` is the
                same regardless of its value. Default 1.0 keeps the math
                trivial.
            loss_pct: System loss percentage (default 14 ≈ PR 0.86), which
                PVGIS subtracts from the gross yield. Accepts 0 to 100.
            pv_tech: PV technology (``"crystSi"``, ``"CIS"``, ``"CdTe"``,
                ``"Unknown"``). Defaults to crystalline silicon (residential
                norm).
            mounting_place: ``"free"`` (free-standing) or ``"building"``
                (BIPV / rooftop with reduced backside cooling).

        Returns:
            :class:`PVGISMonthlyYield` with the 12 monthly energies and
            in-plane irradiations.

        Raises:
            ValueError: If ``peakpower_kwp`` ≤ 0 or ``loss_pct`` outside
                [0, 100].
            ExternalAPIError: Upstream returned non-2xx or malformed JSON.

        Notes:
            PVGIS' ``aspect`` convention has south = 0°, west = +90°,
            east = −90°. We accept the project's compass convention as input
            and convert via:

                pvgis_aspect = azimuth_degrees - 180

            Example: ``azimuth_degrees=90`` (East) becomes
            ``pvgis_aspect=-90``. ``azimuth_degrees=270`` (West) becomes
            ``pvgis_aspect=+90``. ``azimuth_degrees=0`` (North) becomes
            ``pvgis_aspect=-180`` (PVGIS accepts the full ±180 range).
        """
        if peakpower_kwp <= 0:
            raise ValueError(f"peakpower_kwp must be > 0 (got {peakpower_kwp})")
        if not 0.0 <= loss_pct <= 100.0:
            raise ValueError(
                f"loss_pct must be in [0, 100] (got {loss_pct})"
            )

        pvgis_aspect = float(azimuth_degrees) - 180.0
        url = f"{self.base_url}/PVcalc"
        params = {
            "lat": latitude,
            "lon": longitude,
            "peakpower": peakpower_kwp,
            "loss": loss_pct,
            "angle": tilt_degrees,
            "aspect": pvgis_aspect,
            "pvtechchoice": pv_tech,
            "mountingplace": mounting_place,
            "outputformat": "json",
        }

        try:
            response = self._http_client().get(url, params=params)
        except httpx.HTTPError as exc:
            raise ExternalAPIError(
                provider="pvgis",
                url=url,
                message=f"network error: {exc}",
            ) from exc

        if response.status_code != 200:
            raise ExternalAPIError(
                provider="pvgis",
                url=str(response.url),
                message=f"unexpected status (body: {response.text[:200]})",
                status_code=response.status_code,
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise ExternalAPIError(
                provider="pvgis",
                url=str(response.url),
                message=f"non-JSON response: {exc}",
                status_code=response.status_code,
            ) from exc

        return self._parse_payload(
            payload=payload,
            latitude=latitude,
            longitude=longitude,
            tilt_degrees=tilt_degrees,
            azimuth_degrees=azimuth_degrees,
            loss_pct=loss_pct,
            peakpower_kwp=peakpower_kwp,
        )

    @staticmethod
    def _parse_payload(
        payload: dict[str, Any],
        latitude: float,
        longitude: float,
        tilt_degrees: float,
        azimuth_degrees: float,
        loss_pct: float,
        peakpower_kwp: float,
    ) -> PVGISMonthlyYield:
        """
        Convert the PVGIS JSON payload into a :class:`PVGISMonthlyYield`.

        Raises:
            ExternalAPIError: If the expected ``outputs.monthly.fixed`` block
                is missing or has fewer than 12 entries.

        Notes:
            PVGIS payload shape (simplified):

                {
                  "inputs": {"location": {"elevation": ...}, ...},
                  "outputs": {
                    "monthly": {
                      "fixed": [
                        {"month": 1, "E_m": 38.2, "H(i)_m": 35.1, ...},
                        ...
                      ]
                    }
                  }
                }
        """
        try:
            monthly = payload["outputs"]["monthly"]["fixed"]
        except (KeyError, TypeError) as exc:
            raise ExternalAPIError(
                provider="pvgis",
                url="<parse>",
                message=f"missing outputs.monthly.fixed in payload: {exc}",
            ) from exc

        if not isinstance(monthly, list) or len(monthly) < 12:
            raise ExternalAPIError(
                provider="pvgis",
                url="<parse>",
                message=(
                    f"expected 12 monthly entries, got "
                    f"{len(monthly) if isinstance(monthly, list) else type(monthly).__name__}"
                ),
            )

        # Sort by month to guarantee Jan→Dec ordering even if PVGIS reorders.
        by_month = sorted(monthly, key=lambda m: int(m.get("month", 0)))[:12]

        try:
            e_kwh = tuple(float(m["E_m"]) for m in by_month)
            h_i = tuple(float(m["H(i)_m"]) for m in by_month)
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalAPIError(
                provider="pvgis",
                url="<parse>",
                message=f"malformed monthly entry: {exc}",
            ) from exc

        elevation: float | None = None
        try:
            elevation = float(payload["inputs"]["location"]["elevation"])
        except (KeyError, TypeError, ValueError):
            elevation = None

        return PVGISMonthlyYield(
            latitude=latitude,
            longitude=longitude,
            elevation_m=elevation,
            tilt_degrees=tilt_degrees,
            azimuth_degrees=azimuth_degrees,
            loss_pct=loss_pct,
            peakpower_kwp=peakpower_kwp,
            monthly_e_kwh=e_kwh,
            monthly_h_i_kwh_per_m2=h_i,
        )
