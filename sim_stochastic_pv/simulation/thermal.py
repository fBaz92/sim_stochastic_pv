"""
Stochastic ambient-temperature model (Phase 15).

Implements a per-location, monthly-calibrated thermal generator built on
three components:

1. **Seasonal mean** — annual sinusoid fit on archive data, optionally
   superposed with a linear *climate trend* (°C/year). Captures the
   deterministic seasonal swing common to all years.
2. **AR(1) residuals** — daily residuals (observed minus seasonal mean)
   modelled as a lag-1 autoregressive process with per-month parameters
   (φ, σ_innov). Captures the day-to-day persistence that makes "warm
   spells" longer than i.i.d. noise would predict.
3. **Heavy-tailed extreme events** — a peaks-over-threshold (POT) model
   using the Generalized Pareto Distribution (GPD) for both upper
   (heatwaves) and lower (cold snaps) excursions. Each month carries its
   own ``(threshold, ξ shape, σ scale, exceedance_prob)`` per tail. The
   AR(1) state naturally propagates the extreme value for several days,
   reproducing realistic multi-day events.

The model produces *daily mean* temperatures and exposes a deterministic
sinusoidal *diurnal profile* (peak at ~14:00, trough at ~02:00) to lift
the daily series into an hourly series used downstream by the (future)
electrical model (Phase 16) and HVAC load model (Phase 17).

Calibration of the per-month parameters from raw daily archive data lives
in :mod:`sim_stochastic_pv.simulation.thermal_calibration`.

References
----------
- Coles, S. *An Introduction to Statistical Modeling of Extreme Values*,
  Springer, 2001 — chapters 4 (POT) and 7.
- Brockwell & Davis *Time Series: Theory and Methods*, §3.6 (AR(1)).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import stats


# A standard year length for the seasonal harmonic. 365.25 averages leap
# years correctly without introducing per-year branching logic.
DAYS_IN_YEAR = 365.25


# ---------------------------------------------------------------------------
# Day-of-year ↔ month helpers
# ---------------------------------------------------------------------------


# Cumulative day-of-year boundaries (0-indexed) for a non-leap year.
# Index m gives the first day-of-year of month m+1.
_MONTH_BOUNDARIES = np.array(
    [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365],
    dtype=np.int64,
)


def month_of_year(day_of_year: int | np.ndarray) -> int | np.ndarray:
    """
    Map a 0-indexed day-of-year (0..364) to a 0-indexed month (0..11).

    Args:
        day_of_year: Day index within the year. Scalar or array.
            Values are clamped to [0, 364] to handle edge cases at the
            year boundary.

    Returns:
        Month index in [0, 11] of the same shape as ``day_of_year``.

    Example:
        ```python
        month_of_year(0)    # 0 (January 1)
        month_of_year(31)   # 1 (February 1)
        month_of_year(364)  # 11 (December 31)
        ```
    """
    doy = np.clip(np.asarray(day_of_year, dtype=np.int64), 0, 364)
    # np.searchsorted with side='right' returns the index of the first
    # boundary strictly greater than doy, which is exactly month+1.
    months = np.searchsorted(_MONTH_BOUNDARIES, doy, side="right") - 1
    months = np.clip(months, 0, 11)
    if np.ndim(day_of_year) == 0:
        return int(months)
    return months


# ---------------------------------------------------------------------------
# Seasonal harmonic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HarmonicSeasonalMean:
    """
    Deterministic annual seasonal mean as a single first-order sinusoid.

    The temperature at day-of-year ``d`` is

        T_season(d) = a0 + a1·cos(2π·d/365.25) + a2·sin(2π·d/365.25)

    A single sinusoid is sufficient for mid-latitude climates where the
    seasonal cycle is dominated by the annual fundamental (the second
    harmonic captures < 5% of the variance in northern Italy archives).
    A linear climate trend is added on top by the consuming model
    (:class:`ThermalModel.simulate_daily_means`) so this struct stays
    year-agnostic.

    Attributes:
        a0: Annual mean temperature (°C).
        a1: Cosine coefficient (°C). Negative ⇒ trough at January 1.
        a2: Sine coefficient (°C). Together with ``a1`` defines the
            amplitude ``√(a1² + a2²)`` and the phase of the cycle.

    Example:
        ```python
        # Pavullo-like: 12 °C mean, ~10 °C peak-to-mean amplitude,
        # trough around Jan, peak around Jul (a1 ≈ -10, a2 ≈ 0).
        h = HarmonicSeasonalMean(a0=12.0, a1=-10.0, a2=0.0)
        h.evaluate(0)    # ≈ 2.0 (winter)
        h.evaluate(182)  # ≈ 22.0 (summer)
        ```
    """

    a0: float
    a1: float
    a2: float

    def evaluate(self, day_of_year: int | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the seasonal mean at the given day(s) of year.

        Args:
            day_of_year: 0-indexed day-of-year (0..364), scalar or array.

        Returns:
            Seasonal mean temperature (°C), same shape as input.
        """
        omega = 2.0 * np.pi / DAYS_IN_YEAR
        doy = np.asarray(day_of_year, dtype=float)
        return self.a0 + self.a1 * np.cos(omega * doy) + self.a2 * np.sin(omega * doy)


# ---------------------------------------------------------------------------
# Generalized Pareto tails
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPDTail:
    """
    Generalized Pareto Distribution parameters for one tail of a residual
    distribution, calibrated via Peaks-Over-Threshold (POT).

    The model assumes that residuals (observed daily mean minus the
    deterministic seasonal mean) that exceed ``threshold`` are
    GPD-distributed:

        P(R - threshold ≤ x | R > threshold) = 1 − (1 + ξ·x/σ)^(−1/ξ)

    where ``ξ = shape`` and ``σ = scale > 0``. The unconditional probability
    of an exceedance on any given day in this month is
    ``exceedance_prob``.

    For the **upper tail** (heatwaves), ``threshold`` is the positive value
    above which excesses are GPD. For the **lower tail** (cold snaps),
    ``threshold`` is the positive *magnitude*: an event is realised when
    ``residual < −threshold`` and the excess (positive) follows the GPD.
    The :class:`ThermalModel` consumes both shapes consistently — see
    :meth:`ThermalModel.simulate_daily_means`.

    Attributes:
        threshold: Positive value (°C) above which the GPD is fitted.
            For the lower tail this is the absolute magnitude of the
            (negative) threshold.
        shape: GPD shape parameter ξ. ξ > 0 → heavy (Pareto) tail,
            ξ = 0 → exponential tail, ξ < 0 → bounded (Beta) tail.
            Typical values for daily temperature residuals are
            ξ ∈ [−0.2, 0.2].
        scale: GPD scale parameter σ > 0 (°C). The expected excess
            (when ξ < 1) is ``σ / (1 − ξ)``.
        exceedance_prob: Marginal probability of an exceedance on any
            day in this month. Equal to (n_exceedances / n_days_in_month)
            during calibration.

    Example:
        ```python
        # A heatwave tail for July: threshold +6 °C above the seasonal mean,
        # roughly 5% of July days, mildly heavy tail.
        tail = GPDTail(threshold=6.0, shape=0.1, scale=2.5, exceedance_prob=0.05)
        ```

    Notes:
        - ``shape ≥ 1`` would imply infinite mean and is forbidden by
          ``__post_init__`` to keep the calibration outputs safe to use.
        - When ``exceedance_prob = 0`` the tail is effectively disabled
          (no draw will ever happen) — useful to deactivate one side.
    """

    threshold: float
    shape: float
    scale: float
    exceedance_prob: float

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError(f"GPD scale must be > 0 (got {self.scale})")
        if self.shape >= 1.0:
            raise ValueError(
                f"GPD shape >= 1 implies infinite mean; refusing (got {self.shape})"
            )
        if not 0.0 <= self.exceedance_prob <= 1.0:
            raise ValueError(
                f"exceedance_prob must be in [0, 1] (got {self.exceedance_prob})"
            )

    def sample_excess(self, rng: np.random.Generator) -> float:
        """
        Draw a single excess from the GPD (always positive).

        The returned value is the *excess over the threshold*, not the
        absolute residual. The caller combines it with the threshold and
        the sign convention.

        Args:
            rng: NumPy random generator (uses ``stats.genpareto.rvs``
                with ``random_state=rng`` for reproducibility).

        Returns:
            Single excess in °C, > 0.
        """
        return float(stats.genpareto.rvs(c=self.shape, scale=self.scale, random_state=rng))


# ---------------------------------------------------------------------------
# Per-month parameter bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThermalMonthParams:
    """
    Stochastic parameters for a single calendar month.

    Bundles the four blocks the simulator needs to advance day-to-day:

    1. Marginal moments of the residual:
        - ``t_std_residual_c`` — standard deviation of the detrended
          daily-mean residuals.
        - ``persistence_phi`` — lag-1 autocorrelation of the residuals
          (the AR(1) ``φ``). Together with the std it determines the
          innovation std: ``σ_innov = std · √(1 − φ²)``.
    2. Upper tail extreme-event model (heatwaves), :class:`GPDTail` or
       ``None`` to disable.
    3. Lower tail extreme-event model (cold snaps), :class:`GPDTail` or
       ``None`` to disable.
    4. Diurnal swing amplitude ``t_amplitude_c`` (°C). Half of the average
       (Tmax − Tmin) across the calibration window for this month. Used by
       :meth:`ThermalModel.to_hourly` to lift daily means into hourly
       series via a sinusoidal diurnal profile.

    Attributes:
        t_std_residual_c: σ of the daily-mean residuals (°C).
            Typical EU range: 1.5–3.5 °C, larger in shoulder seasons.
        persistence_phi: AR(1) lag-1 autocorrelation. Empirical range for
            daily mean residuals at mid-latitudes is roughly 0.6–0.9.
        gpd_upper: Heatwave tail (positive excursions). ``None`` to
            disable.
        gpd_lower: Cold-snap tail (negative excursions, threshold stored
            as positive magnitude). ``None`` to disable.
        t_amplitude_c: Average diurnal half-amplitude (°C), i.e.
            ``(Tmax_avg − Tmin_avg) / 2`` over the calibration window.

    Notes:
        - All fields are immutable (``frozen=True``) so a model can be
          shared safely across MC paths without accidental mutation.
        - ``persistence_phi`` is clamped at the call site to ``[0, 0.99]``
          to keep ``σ_innov`` defined and positive.
    """

    t_std_residual_c: float
    persistence_phi: float
    t_amplitude_c: float
    gpd_upper: GPDTail | None = None
    gpd_lower: GPDTail | None = None

    def __post_init__(self) -> None:
        if self.t_std_residual_c < 0:
            raise ValueError("t_std_residual_c must be >= 0")
        if not -0.999 <= self.persistence_phi <= 0.999:
            raise ValueError(
                f"persistence_phi must be in (-1, 1), got {self.persistence_phi}"
            )
        if self.t_amplitude_c < 0:
            raise ValueError("t_amplitude_c must be >= 0")


# ---------------------------------------------------------------------------
# Thermal model
# ---------------------------------------------------------------------------


@dataclass
class ExtremeEventReport:
    """
    Summary of GPD-driven extreme events that fired in a simulated run.

    Useful as a sanity check on the calibration and to expose a "did
    anything unusual happen" hint to the UI.

    Attributes:
        upper_event_days: Indices of days where an upper-tail GPD draw
            replaced the AR(1) innovation (0-indexed within the
            simulated horizon).
        lower_event_days: Indices of days where a lower-tail GPD draw
            replaced the AR(1) innovation.
        max_excess_upper_c: Maximum upper-tail excess (°C) observed in
            the run, or 0 if no upper event fired.
        max_excess_lower_c: Maximum lower-tail excess (°C) observed, or 0.
    """

    upper_event_days: list[int] = field(default_factory=list)
    lower_event_days: list[int] = field(default_factory=list)
    max_excess_upper_c: float = 0.0
    max_excess_lower_c: float = 0.0


class ThermalModel:
    """
    Stochastic ambient-temperature generator.

    Combines :class:`HarmonicSeasonalMean`, a list of 12
    :class:`ThermalMonthParams`, and an optional linear climate trend into
    a model that produces daily mean and hourly time series for arbitrary
    horizons.

    The simulator is deterministic for a fixed seed (NumPy + SciPy generator
    threading) so test assertions on percentiles / autocorrelations are
    stable across runs.

    Attributes:
        harmonic: Deterministic seasonal mean component.
        monthly_params: List of exactly 12 :class:`ThermalMonthParams`,
            index 0 = January, index 11 = December.
        climate_trend_c_per_year: Linear °C/year trend added to the
            seasonal mean (0 = stationary climate). Tipical EU value
            ~+0.03 °C/year; default 0 keeps backward-compat / no trend.

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.thermal import (
            ThermalModel, HarmonicSeasonalMean, ThermalMonthParams, GPDTail,
        )

        # Simple stationary model: ±10°C seasonal swing around 12°C,
        # uniform monthly noise σ=2°C, mild persistence, no extremes.
        harmonic = HarmonicSeasonalMean(a0=12.0, a1=-10.0, a2=0.0)
        params = [
            ThermalMonthParams(
                t_std_residual_c=2.0,
                persistence_phi=0.7,
                t_amplitude_c=5.0,
            )
            for _ in range(12)
        ]
        model = ThermalModel(harmonic, params)

        rng = np.random.default_rng(42)
        daily = model.simulate_daily_means(n_days=365, rng=rng)
        assert daily.shape == (365,)
        ```

    Notes:
        - The AR(1) residual carries over across month boundaries — only
          the ``φ`` and ``σ_innov`` change. This is more physical than a
          forced reset.
        - When an extreme draw fires, the AR(1) state is updated *to the
          drawn residual* so the persistence carries the extreme forward
          for several days, reproducing the typical "heatwave plateau".
    """

    def __init__(
        self,
        harmonic: HarmonicSeasonalMean,
        monthly_params: Sequence[ThermalMonthParams],
        climate_trend_c_per_year: float = 0.0,
    ) -> None:
        if len(monthly_params) != 12:
            raise ValueError(
                f"monthly_params must have exactly 12 entries (got {len(monthly_params)})"
            )
        self.harmonic = harmonic
        self.monthly_params: tuple[ThermalMonthParams, ...] = tuple(monthly_params)
        self.climate_trend_c_per_year = float(climate_trend_c_per_year)

    # ------------------------------------------------------------------ daily

    def simulate_daily_means(
        self,
        n_days: int,
        rng: np.random.Generator,
        track_events: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, ExtremeEventReport]:
        """
        Simulate ``n_days`` of daily mean temperature (°C).

        Args:
            n_days: Horizon length in days.
            rng: NumPy random generator (use ``np.random.default_rng(seed)``).
            track_events: When ``True``, also returns an
                :class:`ExtremeEventReport` listing the GPD events that
                fired. Defaults to ``False`` so the common case (the
                fan-chart preview) stays cheap.

        Returns:
            ``np.ndarray`` of shape ``(n_days,)`` with daily means in °C.
            If ``track_events`` is ``True``, returns a tuple
            ``(array, ExtremeEventReport)``.

        Notes:
            Algorithm per day ``d`` (0-indexed):

            1. ``doy = d mod 365``,  ``year = d / 365``.
            2. ``month = month_of_year(doy)`` → look up ThermalMonthParams.
            3. ``seasonal = harmonic.evaluate(doy) + trend · year``.
            4. ``σ_innov = σ_residual · √(1 − φ²)``.
            5. ``new_residual = φ · prev_residual + rng.normal() · σ_innov``.
            6. If ``rng.random() < gpd_upper.exceedance_prob``:
                draw excess, ``new_residual = max(new_residual,
                threshold + excess)``.
            7. Symmetric for lower tail.
            8. ``T[d] = seasonal + new_residual``.

            The asymmetric ``max`` / ``min`` rule means the extreme draw
            only fires if it would actually push the residual into the
            tail — avoids inflating events that were already near-tail by
            chance.
        """
        if n_days <= 0:
            raise ValueError(f"n_days must be > 0 (got {n_days})")

        result = np.zeros(n_days, dtype=float)
        report = ExtremeEventReport() if track_events else None

        prev_residual = 0.0
        omega = 2.0 * np.pi / DAYS_IN_YEAR

        # Pre-compute day-of-year and month indices for the full horizon.
        # This is cheap and vectorisable; only the AR(1) loop stays scalar.
        days = np.arange(n_days)
        doy = (days % 365).astype(np.int64)
        years = (days // 365).astype(np.int64)
        months = np.asarray(month_of_year(doy), dtype=np.int64)

        # Vectorised deterministic component.
        seasonal = (
            self.harmonic.a0
            + self.harmonic.a1 * np.cos(omega * doy)
            + self.harmonic.a2 * np.sin(omega * doy)
            + self.climate_trend_c_per_year * years
        )

        # Pre-draw the standard-normal innovations for the whole horizon to
        # reduce overhead in the per-day loop. The actual scaling depends
        # on the month's σ_innov so we draw N(0,1) here and rescale inside.
        innovations = rng.standard_normal(n_days)
        uniforms_upper = rng.random(n_days)
        uniforms_lower = rng.random(n_days)

        for d in range(n_days):
            params = self.monthly_params[months[d]]
            phi = float(params.persistence_phi)
            sigma_innov = params.t_std_residual_c * np.sqrt(max(0.0, 1.0 - phi * phi))
            new_residual = phi * prev_residual + innovations[d] * sigma_innov

            up = params.gpd_upper
            if up is not None and up.exceedance_prob > 0 and uniforms_upper[d] < up.exceedance_prob:
                excess = up.sample_excess(rng)
                candidate = up.threshold + excess
                if candidate > new_residual:
                    new_residual = candidate
                    if report is not None:
                        report.upper_event_days.append(d)
                        if excess > report.max_excess_upper_c:
                            report.max_excess_upper_c = float(excess)

            lo = params.gpd_lower
            if lo is not None and lo.exceedance_prob > 0 and uniforms_lower[d] < lo.exceedance_prob:
                excess = lo.sample_excess(rng)
                candidate = -(lo.threshold + excess)
                if candidate < new_residual:
                    new_residual = candidate
                    if report is not None:
                        report.lower_event_days.append(d)
                        if excess > report.max_excess_lower_c:
                            report.max_excess_lower_c = float(excess)

            result[d] = seasonal[d] + new_residual
            prev_residual = new_residual

        if report is not None:
            return result, report
        return result

    # ----------------------------------------------------------------- hourly

    def to_hourly(
        self,
        daily_means: np.ndarray,
        start_day_of_year: int = 0,
    ) -> np.ndarray:
        """
        Lift a daily-mean series into a 24-hour series with a sinusoidal
        diurnal profile.

        For each day ``d``:

            T(h) = daily_means[d] + amplitude[month(d)] · cos(2π · (h − 14) / 24)

        Peak at 14:00 (≈ solar afternoon), trough at 02:00, amplitude
        equal to the month-specific ``t_amplitude_c``. This deliberate
        simplification keeps the Phase-16 electrical model and the
        Phase-17 HVAC model on a single shared T_amb(t) source without
        introducing further free parameters.

        Args:
            daily_means: Output of :meth:`simulate_daily_means`,
                shape ``(n_days,)``.
            start_day_of_year: Day-of-year offset for the first entry.
                Defaults to 0 (January 1).

        Returns:
            ``np.ndarray`` of shape ``(n_days * 24,)`` with hourly °C.
            Index ``d*24 + h`` is hour ``h`` of day ``d``.

        Example:
            ```python
            daily = model.simulate_daily_means(365, rng=np.random.default_rng(0))
            hourly = model.to_hourly(daily)
            assert hourly.shape == (365 * 24,)
            # Hour 14 of any day is at the diurnal peak:
            assert hourly[14] > daily[0]
            ```
        """
        n_days = len(daily_means)
        if n_days <= 0:
            return np.zeros(0, dtype=float)

        # Diurnal cosine pattern: peak at hour 14, trough at hour 02.
        h = np.arange(24)
        cos_diurnal = np.cos(2.0 * np.pi * (h - 14.0) / 24.0)  # shape (24,)

        # Per-day month lookup → per-day amplitude.
        doy = (np.arange(n_days) + start_day_of_year) % 365
        months = np.asarray(month_of_year(doy), dtype=np.int64)
        amplitudes = np.array(
            [self.monthly_params[m].t_amplitude_c for m in months],
            dtype=float,
        )  # shape (n_days,)

        # Outer-product-ish: shape (n_days, 24).
        hourly = daily_means[:, None] + amplitudes[:, None] * cos_diurnal[None, :]
        return hourly.reshape(n_days * 24)


# ---------------------------------------------------------------------------
# Preview helper for the wizard fan chart
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemperaturePreviewResult:
    """
    Output of :func:`simulate_temperature_preview`.

    The result carries **two complementary views**:

    1. *Daily-mean fan chart* (``days``, ``mean_c``, ``p05_c``, ``p95_c``,
       ``sample_paths_c``) — the long-term seasonal evolution of the
       *daily mean* temperature with Monte-Carlo uncertainty band.
    2. *Per-month hourly distributions* (``monthly_*``) — for each of the
       12 calendar months, percentiles of the **hourly** temperatures
       collected across every path and every hour of every day in that
       month. These percentiles capture both:
       - the **diurnal swing** (peak at ~14:00, trough at ~02:00,
         amplitude = month-specific ``t_amplitude_c``);
       - the **inter-path variability** (seasonal mean + AR(1)
         residuals + GPD-driven extremes).

       The Y-axis of the corresponding chart therefore answers
       questions like "in July, at the warmest moment, how hot does it
       get?" and "in January, how cold can the night get?".

    Attributes:
        days: Day indices (0..n_days-1) on the x-axis of the daily fan
            chart.
        mean_c: Cross-path mean per day, shape ``(n_days,)``.
        p05_c: 5th percentile per day, shape ``(n_days,)``.
        p95_c: 95th percentile per day, shape ``(n_days,)``.
        sample_paths_c: Subset of individual paths for plotting (light
            grey strokes behind the band). Shape
            ``(min(n_paths, max_sample_paths), n_days)``.
        monthly_p05_c: 5th percentile of hourly temperatures per month
            across all paths. Shape ``(12,)``. Index 0 = January.
        monthly_p25_c: 25th percentile per month, shape ``(12,)``.
        monthly_p50_c: Median per month, shape ``(12,)``.
        monthly_p75_c: 75th percentile per month, shape ``(12,)``.
        monthly_p95_c: 95th percentile per month, shape ``(12,)``.
        monthly_min_c: Minimum hourly temperature per month
            (across all paths and hours), shape ``(12,)``.
        monthly_max_c: Maximum hourly temperature per month, shape
            ``(12,)``.
    """

    days: np.ndarray
    mean_c: np.ndarray
    p05_c: np.ndarray
    p95_c: np.ndarray
    sample_paths_c: np.ndarray
    monthly_p05_c: np.ndarray
    monthly_p25_c: np.ndarray
    monthly_p50_c: np.ndarray
    monthly_p75_c: np.ndarray
    monthly_p95_c: np.ndarray
    monthly_min_c: np.ndarray
    monthly_max_c: np.ndarray


def simulate_temperature_preview(
    model: ThermalModel,
    n_paths: int = 50,
    n_years: int = 1,
    seed: int = 42,
    max_sample_paths: int = 50,
) -> TemperaturePreviewResult:
    """
    Run ``n_paths`` independent ``n_years``-long simulations and collapse
    them into a fan-chart-friendly payload (mean + p05/p95 + sample paths).

    This mirrors the Phase-10 price-profile preview pattern so the UI can
    reuse the same fan-chart widget shape with minimal adapter code.

    Args:
        model: A calibrated :class:`ThermalModel`.
        n_paths: Number of independent MC paths to draw. Default 50.
            Capped at 1000 server-side in the API for response weight.
        n_years: Horizon length per path. Default 1.
        seed: Master seed. Each path gets a deterministic sub-seed
            ``seed + path_index`` so the output is bit-stable across
            equivalent calls.
        max_sample_paths: Max number of individual paths returned in
            ``sample_paths_c``. The rest are aggregated into mean/p05/p95
            but not exposed individually (keeps payload reasonable).

    Returns:
        :class:`TemperaturePreviewResult` ready to be JSON-serialised.

    Raises:
        ValueError: If ``n_paths`` or ``n_years`` ≤ 0.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")
    if n_years <= 0:
        raise ValueError("n_years must be > 0")

    n_days = int(n_years * 365)
    paths = np.zeros((n_paths, n_days), dtype=float)
    # Pre-compute the month index of every simulated day so we can group
    # hourly temperatures by calendar month later (year-agnostic).
    day_indices = np.arange(n_days)
    months_per_day = np.asarray(month_of_year(day_indices % 365), dtype=np.int64)

    # Collect *hourly* values per month across all paths. Lists-of-arrays
    # are simpler than reshape gymnastics and the memory footprint is
    # ~ n_paths * n_days * 24 * 8 B (~3.5 MB for 50 paths × 1 year).
    hourly_by_month: list[list[np.ndarray]] = [[] for _ in range(12)]

    for p in range(n_paths):
        rng = np.random.default_rng(seed + p)
        daily = model.simulate_daily_means(n_days, rng)
        paths[p] = daily
        # Lift to hourly using the diurnal sinusoid embedded in the model.
        hourly = model.to_hourly(daily)  # shape (n_days * 24,)
        # Reshape to (n_days, 24), then group rows by month index.
        hourly_2d = hourly.reshape(n_days, 24)
        for m in range(12):
            mask = months_per_day == m
            if mask.any():
                hourly_by_month[m].append(hourly_2d[mask].ravel())

    # Compute per-month percentiles over the concatenated hourly arrays.
    monthly_p05 = np.zeros(12)
    monthly_p25 = np.zeros(12)
    monthly_p50 = np.zeros(12)
    monthly_p75 = np.zeros(12)
    monthly_p95 = np.zeros(12)
    monthly_min = np.zeros(12)
    monthly_max = np.zeros(12)
    for m in range(12):
        if hourly_by_month[m]:
            stacked = np.concatenate(hourly_by_month[m])
            monthly_p05[m] = float(np.percentile(stacked, 5))
            monthly_p25[m] = float(np.percentile(stacked, 25))
            monthly_p50[m] = float(np.percentile(stacked, 50))
            monthly_p75[m] = float(np.percentile(stacked, 75))
            monthly_p95[m] = float(np.percentile(stacked, 95))
            monthly_min[m] = float(stacked.min())
            monthly_max[m] = float(stacked.max())

    mean = paths.mean(axis=0)
    p05 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    n_sample = min(max_sample_paths, n_paths)
    # Deterministic stride so the same subset is exposed across calls.
    if n_paths <= max_sample_paths:
        sample = paths
    else:
        idx = np.linspace(0, n_paths - 1, n_sample, dtype=int)
        sample = paths[idx]

    return TemperaturePreviewResult(
        days=np.arange(n_days),
        mean_c=mean,
        p05_c=p05,
        p95_c=p95,
        sample_paths_c=sample,
        monthly_p05_c=monthly_p05,
        monthly_p25_c=monthly_p25,
        monthly_p50_c=monthly_p50,
        monthly_p75_c=monthly_p75,
        monthly_p95_c=monthly_p95,
        monthly_min_c=monthly_min,
        monthly_max_c=monthly_max,
    )
