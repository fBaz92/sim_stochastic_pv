"""
Errors raised by the external API clients.
"""

from __future__ import annotations


class ExternalAPIError(RuntimeError):
    """
    Raised when an upstream public API returns an error or an unexpected payload.

    The message always includes the provider name, the URL that was hit
    (sans secrets), and a hint about what went wrong. This is meant to surface
    to the API caller (via FastAPI's 502 Bad Gateway) so the frontend can show
    a meaningful "data source unavailable" notice instead of failing silently.

    Attributes:
        provider: Short tag identifying the upstream (e.g. ``"nominatim"``,
            ``"pvgis"``, ``"openmeteo"``).
        url: The URL whose response triggered the error. Useful for debugging
            without re-running the request.
        status_code: HTTP status code if the request reached the server,
            otherwise ``None`` (network-level failure).
    """

    def __init__(
        self,
        provider: str,
        url: str,
        message: str,
        status_code: int | None = None,
    ) -> None:
        self.provider = provider
        self.url = url
        self.status_code = status_code
        prefix = f"[{provider}]"
        if status_code is not None:
            prefix += f" HTTP {status_code}"
        super().__init__(f"{prefix} {url}: {message}")
