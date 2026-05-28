"""
Phase 17 — Stochastic decorator that adds intra-day variability to any LoadProfile.

The existing load profiles (`MonthlyAverageLoadProfile`, `AreraLoadProfile`,
`HomeAwayLoadProfile`, `WeeklyPatternLoadProfile`) are deterministic patterns
scaled by the scenario JSON: two days in the same month-hour bucket produce
the identical kW, which is unrealistic — real residential consumption sees
±20-30% intra-day swings driven by sequencing choices the model can't
capture (shower order, washing machine timing, oven minutes).

:class:`StochasticLoadProfile` wraps any existing :class:`LoadProfile` and
multiplies its hourly output by a stationary log-normal multiplicative
noise with AR(1) intra-day correlation:

    z[h] = phi * z[h-1] + sigma_innov * w[h]      with  w[h] ~ N(0, 1)
    eps[h] = exp(z[h] - sigma_log**2 / 2)         (Itō-corrected so E[eps] = 1)

where ``sigma_innov = sigma_log * sqrt(1 - phi**2)`` so that the *unconditional*
variance of ``z[h]`` matches the user-facing ``sigma_log**2`` (the standard
deviation of the log-multiplier). The Itō correction ``- sigma_log**2 / 2``
keeps the long-run mean of the multipliers at exactly 1 — i.e. the wrapped
profile retains its monthly/yearly energy budget. ``phi`` controls the
*persistence*: 0 = white noise (every hour independent), 0.5 = default
realistic level (lag-1 autocorrelation 0.5), 1 = degenerate random walk
(disallowed; we cap it just below 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import LoadProfile


#: Default standard deviation of the log-multiplier (≈ ±20% 1-σ swing).
DEFAULT_SIGMA_LOG: float = 0.20

#: Default AR(1) lag-1 autocorrelation of the log-multiplier.
DEFAULT_PHI_INTRA_DAY: float = 0.5


@dataclass(frozen=True)
class StochasticLoadConfig:
    """
    Hyperparameters of the :class:`StochasticLoadProfile` decorator.

    Attributes:
        enabled: Toggle. When False the decorator should not even be
            built — the scenario_builder is responsible for skipping the
            wrap. Kept on the dataclass so the JSON round-trip stays
            symmetric.
        sigma_log: Standard deviation of the log-multiplier. Default
            0.20 → ±20% 1-σ. Must be ≥ 0. Set to 0 to recover the
            deterministic baseline (the decorator becomes a no-op).
        phi_intra_day: AR(1) lag-1 autocorrelation of the log-multiplier.
            Must be in (-1, 1). Default 0.5 — moderate persistence;
            0.0 = white noise; 0.95 = high persistence (long blocks of
            "above-mean" days).

    Example:
        ```python
        StochasticLoadConfig(enabled=True, sigma_log=0.15, phi_intra_day=0.4)
        ```
    """

    enabled: bool = False
    sigma_log: float = DEFAULT_SIGMA_LOG
    phi_intra_day: float = DEFAULT_PHI_INTRA_DAY

    def __post_init__(self) -> None:
        if self.sigma_log < 0:
            raise ValueError(
                f"sigma_log must be >= 0, got {self.sigma_log}"
            )
        if not (-1.0 < self.phi_intra_day < 1.0):
            raise ValueError(
                "phi_intra_day must be strictly within (-1, 1), "
                f"got {self.phi_intra_day}"
            )


class StochasticLoadProfile(LoadProfile):
    """
    Wrap a :class:`LoadProfile` with stationary LogN multiplicative noise.

    The decorator delegates the *shape* of the load (hour-of-day,
    month-of-year, weekday) to the wrapped profile and applies a
    stochastic multiplier to its kW output. The wrapper is faithful to
    the contract of :class:`LoadProfile`:

    - :meth:`reset_for_run` pre-computes the full multiplier path so
      :meth:`get_hourly_load_kw` is O(1) per call.
    - The wrapped profile's ``reset_for_run`` is also invoked so any
      inner stochastic state (e.g. home/away day picks) is refreshed
      with the same RNG.

    The multiplier sequence is **path-level deterministic for the same
    RNG state**: re-seeding the same generator and re-calling
    :meth:`reset_for_run` reproduces the same eps array byte-for-byte.

    Attributes:
        base: The wrapped :class:`LoadProfile`.
        config: Hyperparameters (sigma_log, phi).
        _eps_hourly: Pre-computed multiplier path of shape
            ``(n_years * 365 * 24,)``. ``None`` until
            :meth:`reset_for_run` is called for the first time.
    """

    def __init__(
        self,
        base: LoadProfile,
        config: StochasticLoadConfig,
    ) -> None:
        """
        Args:
            base: Any concrete LoadProfile to be wrapped. The wrapper does
                NOT mutate it.
            config: :class:`StochasticLoadConfig` with sigma_log + phi.
        """
        self.base = base
        self.config = config
        self._eps_hourly: Optional[np.ndarray] = None
        self._n_hours: int = 0

    # ------------------------------------------------------------------
    # LoadProfile interface
    # ------------------------------------------------------------------

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Reset the wrapped profile AND pre-compute the multiplier path.

        The simulator always passes ``rng`` and ``n_years`` (the
        downstream :class:`EnergySystemSimulator.run_one_path` guarantees
        both). Defensive defaults are kept for the rare case of unit
        tests that exercise the decorator outside a path.
        """
        if rng is None:
            rng = np.random.default_rng()
        self.base.reset_for_run(rng=rng, n_years=n_years)
        years = n_years if n_years is not None else 1
        self._n_hours = int(years * 365 * 24)
        self._eps_hourly = _sample_lognormal_ar1_path(
            n_hours=self._n_hours,
            sigma_log=self.config.sigma_log,
            phi=self.config.phi_intra_day,
            rng=rng,
        )

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Return the wrapped profile's kW multiplied by the hour's epsilon.

        The hour index ``h_global`` is reconstructed from the calendar
        coordinates the simulator already maintains; this keeps the
        wrapper stateless with respect to "which hour am I on".

        Notes:
            - When the wrapper is invoked before
              :meth:`reset_for_run`, the multiplier defaults to 1.0 so
              the call still returns a meaningful value (the deterministic
              baseline).
            - Day-of-year is approximated as ``month_in_year * 31 +
              day_in_month`` — this is **not** a real calendar (Feb has 31
              slots, etc.) but it is consistent with the rest of the
              simulator's ``build_calendar`` mock-calendar contract: all
              months have 31 fictitious days, so ``year_index *
              (12 * 31 * 24)`` is the right multiplier for the global
              hour index. See :func:`sim_stochastic_pv.calendar_utils.build_calendar`.
        """
        base_kw = self.base.get_hourly_load_kw(
            year_index=year_index,
            month_in_year=month_in_year,
            day_in_month=day_in_month,
            hour_in_day=hour_in_day,
            weekday=weekday,
        )
        if self._eps_hourly is None or self.config.sigma_log == 0:
            return base_kw
        # Mock-calendar global hour index: matches the offsets that the
        # simulator's `build_calendar` produces (all months treated as
        # 31 days). The multiplier path length is n_years * 365 * 24
        # however, so wrap around to stay inside bounds — at worst we
        # repeat the last day's multipliers a few hours per year.
        global_hour = (
            year_index * 365 * 24
            + month_in_year * 30 * 24
            + day_in_month * 24
            + hour_in_day
        )
        idx = min(global_hour, self._n_hours - 1)
        eps = float(self._eps_hourly[idx])
        return base_kw * eps


def _sample_lognormal_ar1_path(
    n_hours: int,
    sigma_log: float,
    phi: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample a stationary AR(1) log-multiplier path with mean-1 LogN exp.

    The unconditional variance of ``z`` matches ``sigma_log**2`` so the
    user-facing knob is the *standard deviation of the log-multiplier*,
    not the innovation variance.

    Args:
        n_hours: Length of the path.
        sigma_log: Marginal standard deviation of ``z``.
        phi: AR(1) coefficient (|phi| < 1).
        rng: NumPy generator.

    Returns:
        np.ndarray of shape ``(n_hours,)`` with values ≈ LogN(0, sigma_log²)
        and ``E[eps] == 1`` by construction. When ``sigma_log == 0`` the
        function returns an all-ones array — the legacy deterministic case.
    """
    if sigma_log == 0 or n_hours <= 0:
        return np.ones(max(n_hours, 0), dtype=float)
    sigma_innov = sigma_log * np.sqrt(max(0.0, 1.0 - phi * phi))
    z = np.empty(n_hours, dtype=float)
    # Stationary initial draw so the path starts in the steady state.
    z[0] = rng.normal(loc=0.0, scale=sigma_log)
    if n_hours > 1:
        innov = rng.normal(loc=0.0, scale=sigma_innov, size=n_hours - 1)
        # Vectorised AR(1) is not natively supported by NumPy; this loop
        # runs once per simulation path so the O(n_hours) cost stays
        # manageable (≈ 175k hours at 20 years).
        for t in range(1, n_hours):
            z[t] = phi * z[t - 1] + innov[t - 1]
    # Itō correction so E[eps] = 1.
    return np.exp(z - sigma_log * sigma_log / 2.0)
