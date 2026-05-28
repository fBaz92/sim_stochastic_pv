"""
OpenStreetMap Nominatim geocoding client.

Nominatim is a free reverse/forward geocoder maintained by the OpenStreetMap
foundation. It is used in this project to turn a free-text location name
(typed in the wizard "Luogo" step) into a pair (latitude, longitude) +
a display string. No API key required, but the OSM usage policy enforces:

- max 1 request per second from any given client;
- a meaningful ``User-Agent`` header identifying the application.

References:
    https://nominatim.org/release-docs/develop/api/Search/
    https://operations.osmfoundation.org/policies/nominatim/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .errors import ExternalAPIError

# Nominatim's policy requires a UA that identifies the consuming application.
# Keep it stable across sessions so admins can throttle/ban this app if it
# misbehaves — anonymous UAs are the first to be blocked.
DEFAULT_USER_AGENT = "sim-stochastic-pv/1.0 (https://github.com/fBaz92)"

# Public Nominatim instance run by the OSMF. Self-hosting Nominatim is overkill
# for a single-user app; switch to a self-hosted URL only if the rate limit
# becomes a real bottleneck.
NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org"


@dataclass(frozen=True)
class GeocodeResult:
    """
    A single geocoding match returned by Nominatim.

    Attributes:
        display_name: Human-readable description, e.g.
            ``"Milano, Lombardia, Italia"``.
        latitude: Decimal latitude in degrees (-90 to +90).
        longitude: Decimal longitude in degrees (-180 to +180).
        place_class: Nominatim's coarse category (e.g. ``"place"``,
            ``"boundary"``). Used by the UI to filter out address-level matches
            when looking for towns/regions.
        place_type: Nominatim's fine category (e.g. ``"city"``, ``"town"``,
            ``"administrative"``).
        importance: Nominatim's relevance score (0–1). Higher is more
            relevant. The frontend uses this to sort the autocomplete list.

    Example:
        ```python
        client = NominatimClient()
        results = client.search("Pavullo nel Frignano", limit=3)
        results[0].display_name  # "Pavullo nel Frignano, Modena, ..."
        results[0].latitude      # ~44.336
        ```
    """

    display_name: str
    latitude: float
    longitude: float
    place_class: str | None = None
    place_type: str | None = None
    importance: float | None = None


class NominatimClient:
    """
    Synchronous client for the public Nominatim ``/search`` endpoint.

    Notes:
        - No API key required.
        - Subject to OSMF usage policy: max 1 req/s + identifying UA.
        - The class does not enforce the 1 req/s itself; the wrapping endpoint
          should debounce/throttle on the server side if multiple calls are
          forwarded per second.
        - All HTTP failures are wrapped in :class:`ExternalAPIError` for
          consistent surfacing.

    Example:
        ```python
        with NominatimClient() as client:
            results = client.search("Roma", limit=5)
            for r in results:
                print(r.display_name, r.latitude, r.longitude)
        ```
    """

    def __init__(
        self,
        base_url: str = NOMINATIM_BASE_URL,
        user_agent: str = DEFAULT_USER_AGENT,
        timeout_s: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        """
        Initialize the Nominatim client.

        Args:
            base_url: Override the upstream URL. Useful for tests (mock
                server) or for pointing to a self-hosted instance.
            user_agent: Identifying ``User-Agent`` header value. Required
                by the OSMF policy.
            timeout_s: Per-request timeout in seconds.
            client: Optional ``httpx.Client`` to reuse across calls. If
                ``None`` a private client is created per :meth:`__enter__`.
        """
        self.base_url = base_url.rstrip("/")
        self.user_agent = user_agent
        self.timeout_s = timeout_s
        self._client = client
        self._owns_client = client is None

    def __enter__(self) -> "NominatimClient":
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

    def _http_client(self) -> httpx.Client:
        """Lazy-init the underlying httpx client when used outside ``with``."""
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout_s)
        return self._client

    def search(
        self,
        query: str,
        limit: int = 5,
        accept_language: str = "it,en",
    ) -> list[GeocodeResult]:
        """
        Forward-geocode a free-text query into up to ``limit`` candidates.

        Args:
            query: Place name, address, or partial location string.
                Empty / whitespace-only inputs return an empty list without
                hitting the network.
            limit: Max number of candidates Nominatim should return
                (clamped to [1, 10] — the public instance discourages more).
            accept_language: Preferred languages for ``display_name``,
                forwarded as ``Accept-Language``. Defaults to Italian then
                English to suit the project's primary audience.

        Returns:
            List of :class:`GeocodeResult` ordered by Nominatim's
            ``importance`` desc. May be empty if the query has no match.

        Raises:
            ExternalAPIError: Upstream returned non-2xx or non-JSON.

        Example:
            ```python
            client = NominatimClient()
            results = client.search("Pavullo nel Frignano", limit=3)
            # results[0] ≈ GeocodeResult(display_name="Pavullo nel Frignano...",
            #                            latitude=44.336, longitude=10.831, ...)
            ```
        """
        if not query or not query.strip():
            return []

        clamped_limit = max(1, min(10, int(limit)))
        url = f"{self.base_url}/search"
        params = {
            "q": query.strip(),
            "format": "jsonv2",
            "limit": clamped_limit,
            "addressdetails": 0,
        }
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": accept_language,
        }

        try:
            response = self._http_client().get(url, params=params, headers=headers)
        except httpx.HTTPError as exc:
            raise ExternalAPIError(
                provider="nominatim",
                url=url,
                message=f"network error: {exc}",
            ) from exc

        if response.status_code != 200:
            raise ExternalAPIError(
                provider="nominatim",
                url=str(response.url),
                message=f"unexpected status code (body: {response.text[:200]})",
                status_code=response.status_code,
            )

        try:
            payload: list[dict[str, Any]] = response.json()
        except ValueError as exc:
            raise ExternalAPIError(
                provider="nominatim",
                url=str(response.url),
                message=f"non-JSON response: {exc}",
                status_code=response.status_code,
            ) from exc

        return [self._parse_one(item) for item in payload]

    @staticmethod
    def _parse_one(item: dict[str, Any]) -> GeocodeResult:
        """
        Convert one Nominatim JSON entry into a :class:`GeocodeResult`.

        Raises:
            ExternalAPIError: Required fields missing or malformed.
        """
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
            display = str(item["display_name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExternalAPIError(
                provider="nominatim",
                url="<parse>",
                message=f"malformed entry {item!r}: {exc}",
            ) from exc

        return GeocodeResult(
            display_name=display,
            latitude=lat,
            longitude=lon,
            place_class=item.get("class"),
            place_type=item.get("type"),
            importance=(
                float(item["importance"]) if "importance" in item else None
            ),
        )
