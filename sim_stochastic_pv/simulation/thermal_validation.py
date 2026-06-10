"""
Backtest of a calibrated :class:`ThermalModel` against observed extremes.

Answers the trust question every user asks before believing a stochastic
climate model: *"would this model have produced the temperatures my site
actually saw?"*. The harness compares the distribution of **simulated
annual extremes** (hottest hourly temperature and coldest hourly
temperature of each year) against the **observed annual extremes** from
the same daily archive the model was calibrated on.

Methodology
-----------
- Observed annual maxima come from the archive's daily ``t_max`` series
  (and minima from ``t_min``), grouped by calendar year. Partial years
  (fewer than :data:`MIN_DAYS_PER_OBSERVED_YEAR` valid days) are skipped
  so a half-downloaded year cannot fake a mild extreme.
- The model simulates ``n_paths`` independent windows of the same length
  as the observed window. Because the calibration epoch sits at the
  *midpoint* of the archive window (the trend is centred there), the
  backtest re-anchors the seasonal mean at the *window start* so
  simulated year ``y`` is climatologically aligned with observed year
  ``y`` — without this shift a positive trend would make the simulated
  window systematically warmer than the observed one.
- Simulated annual extremes are pooled across paths and years into
  percentiles; the comparison metrics are the **coverage** (fraction of
  observed annual extremes inside the simulated p05–p95 band — expect
  ≈ 0.9 for a well-calibrated model) and the **median bias** (simulated
  median minus observed median, °C).

Example:
    ```python
    result = backtest_annual_extremes(model, dates, t_max_c, t_min_c)
    print(result.tmax_median_bias_c)   # ≈ 0 for a healthy calibration
    print(result.tmax_coverage)        # ≈ 0.9
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .thermal import HarmonicSeasonalMean, ThermalModel


# An observed calendar year enters the backtest only when it has at least
# this many valid daily observations: prevents partial years (e.g. the
# current year downloaded in March) from contributing a fake mild extreme.
MIN_DAYS_PER_OBSERVED_YEAR = 300


@dataclass
class ExtremesBacktestResult:
    """
    Outcome of :func:`backtest_annual_extremes`.

    All temperatures are °C. The ``sim_*`` percentiles describe the pooled
    distribution of simulated annual extremes (``n_paths × n_years``
    values per tail).

    Attributes:
        observed_years: Calendar years that entered the comparison.
        observed_annual_tmax: Observed hottest daily ``t_max`` per year,
            aligned with ``observed_years``.
        observed_annual_tmin: Observed coldest daily ``t_min`` per year.
        sim_tmax_p05: 5th percentile of the simulated annual maxima.
        sim_tmax_p50: Median of the simulated annual maxima.
        sim_tmax_p95: 95th percentile of the simulated annual maxima.
        sim_tmin_p05: 5th percentile of the simulated annual minima
            (the *colder* end of the band).
        sim_tmin_p50: Median of the simulated annual minima.
        sim_tmin_p95: 95th percentile of the simulated annual minima.
        tmax_coverage: Fraction of observed annual maxima falling inside
            ``[sim_tmax_p05, sim_tmax_p95]``. ≈ 0.9 when healthy.
        tmin_coverage: Same for the minima.
        tmax_median_bias_c: ``sim_tmax_p50 − median(observed maxima)``.
            Positive = the model runs hot on extremes.
        tmin_median_bias_c: ``sim_tmin_p50 − median(observed minima)``.
        n_paths: Number of simulated paths.
        n_years: Window length in years (per path).
    """

    observed_years: list[int] = field(default_factory=list)
    observed_annual_tmax: list[float] = field(default_factory=list)
    observed_annual_tmin: list[float] = field(default_factory=list)
    sim_tmax_p05: float = 0.0
    sim_tmax_p50: float = 0.0
    sim_tmax_p95: float = 0.0
    sim_tmin_p05: float = 0.0
    sim_tmin_p50: float = 0.0
    sim_tmin_p95: float = 0.0
    tmax_coverage: float = 0.0
    tmin_coverage: float = 0.0
    tmax_median_bias_c: float = 0.0
    tmin_median_bias_c: float = 0.0
    n_paths: int = 0
    n_years: int = 0


def observed_annual_extremes(
    dates: Sequence[str],
    t_max_c: Sequence[float | None],
    t_min_c: Sequence[float | None],
    min_days_per_year: int = MIN_DAYS_PER_OBSERVED_YEAR,
) -> tuple[list[int], list[float], list[float]]:
    """
    Extract per-calendar-year extremes from parallel daily archive arrays.

    Args:
        dates: ISO-format date strings (``"2020-01-01"``).
        t_max_c: Daily maximum temperatures aligned with ``dates``
            (``None``/NaN entries ignored).
        t_min_c: Daily minimum temperatures aligned with ``dates``.
        min_days_per_year: Minimum number of valid days for a year to be
            included (default :data:`MIN_DAYS_PER_OBSERVED_YEAR`).

    Returns:
        Tuple ``(years, annual_tmax, annual_tmin)`` sorted by year. A year
        appears only when *both* series have enough valid days.

    Raises:
        ValueError: Arrays have mismatched lengths.
    """
    n = len(dates)
    if len(t_max_c) != n or len(t_min_c) != n:
        raise ValueError(
            f"length mismatch: dates={n}, t_max={len(t_max_c)}, t_min={len(t_min_c)}"
        )

    tmax = np.array(
        [v if v is not None else np.nan for v in t_max_c], dtype=float
    )
    tmin = np.array(
        [v if v is not None else np.nan for v in t_min_c], dtype=float
    )
    years = np.array(
        [int(d[0:4]) if len(d) >= 4 else -1 for d in dates], dtype=np.int64
    )

    out_years: list[int] = []
    out_tmax: list[float] = []
    out_tmin: list[float] = []
    for y in sorted({int(v) for v in years if v > 0}):
        mask = years == y
        valid_max = np.isfinite(tmax[mask])
        valid_min = np.isfinite(tmin[mask])
        if valid_max.sum() < min_days_per_year or valid_min.sum() < min_days_per_year:
            continue
        out_years.append(y)
        out_tmax.append(float(np.nanmax(tmax[mask])))
        out_tmin.append(float(np.nanmin(tmin[mask])))
    return out_years, out_tmax, out_tmin


def backtest_annual_extremes(
    model: ThermalModel,
    dates: Sequence[str],
    t_max_c: Sequence[float | None],
    t_min_c: Sequence[float | None],
    n_paths: int = 50,
    seed: int = 42,
) -> ExtremesBacktestResult:
    """
    Compare simulated vs. observed annual temperature extremes.

    Pipeline:

    1. Extract observed annual extremes per calendar year (partial years
       skipped, see :func:`observed_annual_extremes`).
    2. Re-anchor the model's seasonal mean at the window *start*
       (``a0 − trend · (L−1)/2``) so simulated year ``y`` aligns with
       observed year ``y`` despite the trend being centred on the window
       midpoint during calibration.
    3. Simulate ``n_paths`` windows of ``L`` years; lift each to hourly
       (diurnal sinusoid + clear-sky amplitude coupling) and collect the
       annual max and min of the hourly series per (path, year).
    4. Pool into percentiles, compute coverage and median bias.

    Args:
        model: Calibrated :class:`ThermalModel` (typically loaded from a
            saved climate profile).
        dates: ISO dates of the observed daily archive.
        t_max_c: Observed daily maxima aligned with ``dates``.
        t_min_c: Observed daily minima.
        n_paths: Independent simulation paths (default 50 — enough for
            stable p05/p95 on 10-year windows while keeping the endpoint
            latency in the seconds range).
        seed: Master RNG seed; path ``p`` uses ``seed + p``.

    Returns:
        :class:`ExtremesBacktestResult`. When fewer than 2 observed years
        survive the validity filter the simulated percentiles are still
        computed (against a window of ``max(n_years, 1)`` years) but the
        coverage/bias fields stay 0 — the caller should surface "archivio
        insufficiente" instead of a verdict.

    Raises:
        ValueError: ``n_paths`` ≤ 0 or empty inputs.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")
    if not dates:
        raise ValueError("empty archive input")

    obs_years, obs_tmax, obs_tmin = observed_annual_extremes(
        dates, t_max_c, t_min_c
    )
    n_years = max(len(obs_years), 1)

    # Re-anchor the epoch on the window start (see module docstring).
    midpoint = (n_years - 1) / 2.0
    backtest_model = ThermalModel(
        harmonic=HarmonicSeasonalMean(
            a0=model.harmonic.a0 - model.climate_trend_c_per_year * midpoint,
            a1=model.harmonic.a1,
            a2=model.harmonic.a2,
        ),
        monthly_params=model.monthly_params,
        climate_trend_c_per_year=model.climate_trend_c_per_year,
    )

    n_days = 365 * n_years
    sim_max: list[float] = []
    sim_min: list[float] = []
    for p in range(n_paths):
        rng = np.random.default_rng(seed + p)
        daily = backtest_model.simulate_daily_means(n_days, rng)
        hourly = backtest_model.to_hourly(daily).reshape(n_days, 24)
        day_max = hourly.max(axis=1)
        day_min = hourly.min(axis=1)
        for y in range(n_years):
            sl = slice(y * 365, (y + 1) * 365)
            sim_max.append(float(day_max[sl].max()))
            sim_min.append(float(day_min[sl].min()))

    sim_max_arr = np.asarray(sim_max)
    sim_min_arr = np.asarray(sim_min)

    result = ExtremesBacktestResult(
        observed_years=obs_years,
        observed_annual_tmax=obs_tmax,
        observed_annual_tmin=obs_tmin,
        sim_tmax_p05=float(np.percentile(sim_max_arr, 5)),
        sim_tmax_p50=float(np.percentile(sim_max_arr, 50)),
        sim_tmax_p95=float(np.percentile(sim_max_arr, 95)),
        sim_tmin_p05=float(np.percentile(sim_min_arr, 5)),
        sim_tmin_p50=float(np.percentile(sim_min_arr, 50)),
        sim_tmin_p95=float(np.percentile(sim_min_arr, 95)),
        n_paths=n_paths,
        n_years=n_years,
    )

    if len(obs_years) >= 2:
        obs_tmax_arr = np.asarray(obs_tmax)
        obs_tmin_arr = np.asarray(obs_tmin)
        result.tmax_coverage = float(
            np.mean(
                (obs_tmax_arr >= result.sim_tmax_p05)
                & (obs_tmax_arr <= result.sim_tmax_p95)
            )
        )
        result.tmin_coverage = float(
            np.mean(
                (obs_tmin_arr >= result.sim_tmin_p05)
                & (obs_tmin_arr <= result.sim_tmin_p95)
            )
        )
        result.tmax_median_bias_c = float(
            result.sim_tmax_p50 - np.median(obs_tmax_arr)
        )
        result.tmin_median_bias_c = float(
            result.sim_tmin_p50 - np.median(obs_tmin_arr)
        )

    return result
