"""
Calibrate a :class:`ThermalModel` from raw daily archive data.

Given an arbitrary-length daily time series of (date, tmean, tmax, tmin) —
typically pulled from the Open-Meteo Archive over the last 10–30 years —
this module fits:

1. The annual harmonic seasonal mean ``a0 + a1·cos + a2·sin`` by least
   squares on the daily means (over the full multi-year window).
2. Per-month AR(1) parameters ``(t_std_residual_c, persistence_phi)``
   from the detrended residuals.
3. Per-month per-tail GPD models ``(threshold, shape ξ, scale σ,
   exceedance_prob)`` via Peaks-Over-Threshold on the residuals.
4. Per-month diurnal half-amplitude ``t_amplitude_c`` from the average
   (tmax − tmin) / 2.

The output is a fully constructed :class:`ThermalModel` plus an audit
struct (:class:`CalibrationReport`) the API endpoint exposes for
"trust but verify" in the UI.

Design notes
------------
- POT calibration uses a per-month percentile threshold (default p90).
  scipy.stats.genpareto's ``fit(floc=0)`` MLE is the standard estimator
  for the GPD shape and scale on excesses ≥ 0.
- The harmonic is fit on the *whole* window in one pass — much simpler
  and statistically tighter than a per-month constant. The seasonal
  cycle is what it is regardless of which month you're in.
- If a month has fewer than ``min_samples_per_month`` valid days
  (e.g. very short archive window or excessive NaNs) the GPD fits
  are skipped (tails ``None``) — the model degrades gracefully to
  "seasonal + AR(1) only" for that month.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

from .thermal import (
    DAYS_IN_YEAR,
    GPDTail,
    HarmonicSeasonalMean,
    ThermalModel,
    ThermalMonthParams,
    month_of_year,
)


# Minimum number of valid daily observations per month required to attempt
# the GPD fit. With fewer samples the maximum-likelihood estimator becomes
# unreliable, and the tail is left as ``None``.
DEFAULT_MIN_SAMPLES_PER_MONTH_GPD = 60

# Default percentile used for the POT threshold (per month).
DEFAULT_POT_PERCENTILE = 90.0

# Fallback diurnal amplitude (°C) when no (tmax, tmin) data is provided.
DEFAULT_FALLBACK_AMPLITUDE_C = 4.0

# Minimum number of (residual, amplitude) pairs in a month to fit the
# amplitude-vs-residual slope. Below this the slope stays at 0 (constant
# amplitude, legacy behaviour).
MIN_SAMPLES_AMP_SLOPE = 30

# Hard clip for the fitted amplitude slope (°C of half-amplitude per °C of
# residual). Physical values are well inside ±1; the clip only guards
# against degenerate regressions on pathological inputs.
AMP_SLOPE_CLIP = 2.0

# Minimum span (years) of ``year_offset`` data required to fit the linear
# climate trend. With a shorter span the trend estimate is dominated by
# interannual noise and is left at 0.
MIN_YEARS_SPAN_FOR_TREND = 3.0


@dataclass(frozen=True)
class DailyArchiveSample:
    """
    One day of input climate data, as used by the calibration.

    Attributes:
        day_of_year: 0-indexed day of year (0..364). Used for the
            harmonic regression and the month lookup.
        t_mean_c: Daily mean temperature (°C). Required.
        t_max_c: Daily maximum temperature (°C). Optional — contributes
            to the diurnal-amplitude calibration only.
        t_min_c: Daily minimum temperature (°C). Optional.
        year_offset: Years elapsed since the start of the calibration
            window (e.g. 0.0 for the first January, 9.5 for mid-year of
            the tenth year). Optional — when present on enough samples
            the calibration fits a linear warming trend and detrends the
            residuals before the per-month AR(1)/GPD fits; when absent
            the calibration is trend-blind (legacy behaviour).

    Notes:
        Either ``t_mean_c`` alone or the trio ``(t_mean_c, t_max_c, t_min_c)``
        can be supplied. When max/min are missing the calibration uses
        :data:`DEFAULT_FALLBACK_AMPLITUDE_C` for the diurnal swing.
    """

    day_of_year: int
    t_mean_c: float
    t_max_c: float | None = None
    t_min_c: float | None = None
    year_offset: float | None = None


@dataclass
class CalibrationReport:
    """
    Audit information about a calibration run.

    Attributes:
        n_samples: Total number of valid daily samples consumed.
        years_covered: Window length, equal to ``n_samples / 365`` rounded.
        rmse_harmonic_c: Root mean square error of the harmonic seasonal
            fit (°C). A small value (≤ 4 °C for European mid-latitudes)
            indicates a well-behaved annual cycle.
        fitted_trend_c_per_year: Linear warming trend fitted on the
            harmonic residuals vs. years-since-window-start (°C/year).
            0.0 when the samples carry no ``year_offset`` information.
            Recent Po-valley/Apennine windows run ≈ +0.05..+0.12 °C/year.
        per_month_n_samples: Per-month sample counts. Indexed 0..11.
        per_month_gpd_upper_fitted: True if the upper-tail GPD was fitted
            for that month, False if skipped.
        per_month_gpd_lower_fitted: Same for the lower tail.
        per_month_amp_slope: Fitted sensitivity of the diurnal
            half-amplitude to the daily residual, per month (°C/°C).
    """

    n_samples: int = 0
    years_covered: float = 0.0
    rmse_harmonic_c: float = 0.0
    fitted_trend_c_per_year: float = 0.0
    per_month_n_samples: list[int] = field(default_factory=lambda: [0] * 12)
    per_month_gpd_upper_fitted: list[bool] = field(default_factory=lambda: [False] * 12)
    per_month_gpd_lower_fitted: list[bool] = field(default_factory=lambda: [False] * 12)
    per_month_amp_slope: list[float] = field(default_factory=lambda: [0.0] * 12)


# ---------------------------------------------------------------------------
# Harmonic seasonal fit
# ---------------------------------------------------------------------------


def fit_harmonic_seasonal_mean(
    day_of_year: np.ndarray,
    t_mean_c: np.ndarray,
) -> tuple[HarmonicSeasonalMean, float]:
    """
    Least-squares fit of the first-order annual harmonic to (DOY, Tmean).

    Model:

        T_mean(d) = a0 + a1·cos(2π·d/365.25) + a2·sin(2π·d/365.25) + ε

    Args:
        day_of_year: int array of day-of-year values (0..364), shape (N,).
        t_mean_c: float array of corresponding daily means, shape (N,).

    Returns:
        A tuple ``(harmonic, rmse_c)`` where ``harmonic`` is the fitted
        :class:`HarmonicSeasonalMean` and ``rmse_c`` is the residual root
        mean square error of the fit (°C).

    Raises:
        ValueError: Input arrays have mismatched lengths or are empty.

    Notes:
        Uses :func:`numpy.linalg.lstsq` (QR-based, stable). For our N's
        (≤ 11 000 days for 30 years) this is essentially instantaneous.
    """
    if day_of_year.shape != t_mean_c.shape:
        raise ValueError(
            f"shape mismatch: doy {day_of_year.shape} vs tmean {t_mean_c.shape}"
        )
    if day_of_year.size == 0:
        raise ValueError("empty calibration input")

    omega = 2.0 * np.pi / DAYS_IN_YEAR
    doy = day_of_year.astype(float)
    design = np.stack(
        [np.ones_like(doy), np.cos(omega * doy), np.sin(omega * doy)],
        axis=1,
    )  # shape (N, 3)
    coeffs, *_ = np.linalg.lstsq(design, t_mean_c, rcond=None)
    a0, a1, a2 = (float(c) for c in coeffs)

    predicted = design @ coeffs
    residuals = t_mean_c - predicted
    rmse = float(np.sqrt(np.mean(residuals * residuals)))

    return HarmonicSeasonalMean(a0=a0, a1=a1, a2=a2), rmse


# ---------------------------------------------------------------------------
# Per-month AR(1)
# ---------------------------------------------------------------------------


def fit_ar1(residuals: np.ndarray) -> tuple[float, float]:
    """
    Fit AR(1) ``(σ, φ)`` to a residual series.

    Estimators:
        - ``φ`` = lag-1 sample autocorrelation, clamped to [-0.95, 0.95]
          to keep the simulator stable.
        - ``σ`` = sample standard deviation of the input residuals.

    The simulator uses ``σ_innov = σ · √(1 − φ²)`` so the marginal std
    of the simulated AR(1) matches ``σ``. We return the marginal std
    here (not the innovation std) because that's what
    :class:`ThermalMonthParams` expects.

    Args:
        residuals: 1D array of residuals (°C) within one calendar month
            across all years in the window.

    Returns:
        ``(t_std_residual_c, persistence_phi)``. If the input has fewer
        than 2 samples, returns ``(0.0, 0.0)`` (no noise).
    """
    if residuals.size < 2:
        return 0.0, 0.0
    sigma = float(np.std(residuals, ddof=1))
    if sigma == 0.0:
        return 0.0, 0.0
    # Lag-1 autocorrelation
    centered = residuals - residuals.mean()
    num = float(np.sum(centered[:-1] * centered[1:]))
    den = float(np.sum(centered * centered))
    phi = num / den if den > 0 else 0.0
    phi = float(np.clip(phi, -0.95, 0.95))
    return sigma, phi


# ---------------------------------------------------------------------------
# Per-month per-tail GPD
# ---------------------------------------------------------------------------


def fit_gpd_tail(
    residuals: np.ndarray,
    tail: str,
    threshold_percentile: float = DEFAULT_POT_PERCENTILE,
    min_excesses: int = 5,
) -> GPDTail | None:
    """
    Fit a single-tail GPD on the residuals via Peaks-Over-Threshold.

    For the upper tail: threshold = ``np.percentile(residuals, p)`` and
    excesses are positive (residual − threshold > 0).

    For the lower tail: threshold (magnitude) =
    ``-np.percentile(residuals, 100 - p)`` and excesses are positive
    (-residual − threshold > 0). The returned :class:`GPDTail` stores
    the threshold magnitude (positive), and the simulator interprets it
    as the *negative* threshold in the residual space.

    Args:
        residuals: 1D residuals for one month across all years.
        tail: ``"upper"`` or ``"lower"``.
        threshold_percentile: For the upper tail this is the percentile
            *above which* the GPD is fitted (default 90 = top 10% of
            residuals). For the lower tail it is symmetric: we use
            ``100 - p`` as the lower percentile cut.
        min_excesses: Minimum number of excess observations required to
            attempt the MLE fit. If fewer, returns ``None`` (gracefully
            degrade — calibration audit records this).

    Returns:
        A fitted :class:`GPDTail` or ``None`` if too few excesses or
        if the MLE failed.

    Notes:
        ``scipy.stats.genpareto.fit`` is the MLE estimator. We fix the
        location parameter ``floc=0`` because excesses by construction
        start at zero. If MLE returns a shape ≥ 1 (heavy tail with
        infinite mean) we clamp to 0.99 to keep the returned object
        usable — extreme heavy tails are likely calibration artefacts
        on small samples.
    """
    if tail not in ("upper", "lower"):
        raise ValueError(f"tail must be 'upper' or 'lower', got {tail!r}")

    if residuals.size < min_excesses + 1:
        return None

    if tail == "upper":
        threshold = float(np.percentile(residuals, threshold_percentile))
        excesses = residuals[residuals > threshold] - threshold
    else:
        threshold_signed = float(np.percentile(residuals, 100.0 - threshold_percentile))
        # Want excesses > 0 below the negative threshold; convert to magnitude.
        below = residuals[residuals < threshold_signed]
        excesses = (-below) - (-threshold_signed)
        threshold = -threshold_signed  # store as positive magnitude

    if excesses.size < min_excesses or threshold <= 0:
        return None

    try:
        shape, _loc, scale = stats.genpareto.fit(excesses, floc=0.0)
    except Exception:  # pragma: no cover - extremely rare
        return None

    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        return None

    # Guard against shape ≥ 1 (infinite mean) — clamp to keep the object
    # constructable. With a clamp at 0.99 the expected excess stays
    # finite (σ / (1 − 0.99) = 100·σ which is large but bounded).
    shape = float(np.clip(shape, -0.5, 0.99))

    exceedance_prob = excesses.size / residuals.size

    return GPDTail(
        threshold=threshold,
        shape=shape,
        scale=float(scale),
        exceedance_prob=float(exceedance_prob),
    )


# ---------------------------------------------------------------------------
# Top-level calibration
# ---------------------------------------------------------------------------


def fit_linear_trend(
    year_offset: np.ndarray,
    residuals: np.ndarray,
    min_years_span: float = MIN_YEARS_SPAN_FOR_TREND,
) -> float:
    """
    Least-squares slope of the harmonic residuals vs. time (°C/year).

    Fitting the trend on the *residuals* (observed minus seasonal
    harmonic) instead of the raw series keeps the two regressions
    orthogonal: the harmonic absorbs the within-year cycle, the trend
    absorbs the across-year drift.

    Args:
        year_offset: Years since the window start per sample, shape (N,).
            May contain NaN for samples without year information — those
            are excluded from the fit.
        residuals: Harmonic residuals (°C), same shape.
        min_years_span: Minimum span (max − min year_offset) required to
            attempt the fit; below it the trend estimate would be noise
            and 0.0 is returned.

    Returns:
        The fitted slope in °C/year, or 0.0 when the span is too short
        or fewer than 2 valid samples exist.
    """
    valid = np.isfinite(year_offset) & np.isfinite(residuals)
    if valid.sum() < 2:
        return 0.0
    x = year_offset[valid]
    if float(x.max() - x.min()) < min_years_span:
        return 0.0
    y = residuals[valid]
    slope = np.polyfit(x - x.min(), y, 1)[0]
    return float(slope)


def fit_amplitude_slope(
    residuals: np.ndarray,
    amplitudes: np.ndarray,
    min_samples: int = MIN_SAMPLES_AMP_SLOPE,
) -> float:
    """
    Least-squares slope of the diurnal half-amplitude vs. the daily
    residual, for one month (°C of amplitude per °C of residual).

    Captures the clear-sky effect: positive temperature anomalies are
    usually sunny days with a wider day/night swing, so the hottest days
    peak higher than ``daily_mean + average_amplitude`` would suggest.

    Args:
        residuals: Detrended harmonic residuals for the month, shape (N,).
        amplitudes: Matching half-amplitudes ``(tmax − tmin)/2``, shape (N,).
            NaN pairs are excluded.
        min_samples: Minimum number of valid pairs to attempt the fit;
            below it 0.0 is returned (constant amplitude).

    Returns:
        The fitted slope clipped to ``[-AMP_SLOPE_CLIP, AMP_SLOPE_CLIP]``,
        or 0.0 when the sample is too small or degenerate.
    """
    valid = np.isfinite(residuals) & np.isfinite(amplitudes)
    if valid.sum() < min_samples:
        return 0.0
    x = residuals[valid]
    y = amplitudes[valid]
    if float(np.std(x)) == 0.0:
        return 0.0
    slope = np.polyfit(x, y, 1)[0]
    return float(np.clip(slope, -AMP_SLOPE_CLIP, AMP_SLOPE_CLIP))


def calibrate_thermal_model(
    samples: Sequence[DailyArchiveSample],
    climate_trend_c_per_year: float | None = None,
    pot_percentile: float = DEFAULT_POT_PERCENTILE,
    min_samples_per_month_gpd: int = DEFAULT_MIN_SAMPLES_PER_MONTH_GPD,
    fallback_amplitude_c: float = DEFAULT_FALLBACK_AMPLITUDE_C,
) -> tuple[ThermalModel, CalibrationReport]:
    """
    Build a :class:`ThermalModel` from raw daily archive data.

    Pipeline:

    1. Stack samples → arrays.
    2. Fit harmonic seasonal mean over the full window.
    3. Compute per-day residuals = t_mean − harmonic(doy).
    4. Fit the linear warming trend on residuals vs. ``year_offset``
       (when the samples carry year information) and **detrend** the
       residuals before any per-month statistic — otherwise the warming
       of the recent years is smeared into σ/GPD instead of being
       extrapolated forward.
    5. For each month:
       a. Subset detrended residuals belonging to that month.
       b. Fit AR(1) → ``(σ, φ)``.
       c. Fit GPD upper / lower tails (or ``None`` if sample too small).
       d. Compute mean diurnal half-amplitude from tmax/tmin if available
          (else ``fallback_amplitude_c``) plus the amplitude-vs-residual
          slope (clear-sky coupling, see
          :attr:`ThermalMonthParams.amp_slope_per_c`).
    6. Wrap into a :class:`ThermalModel` whose trend is the fitted one
       (or the explicit override).

    Args:
        samples: Iterable of :class:`DailyArchiveSample`. Order does not
            matter — the calibration is invariant under permutation.
        climate_trend_c_per_year: ``None`` (default) lets the calibration
            *fit* the trend from the data and bake it into the model
            (0.0 when the samples carry no ``year_offset``). Pass an
            explicit float to force a specific trend — e.g. 0.0 for a
            deliberately stationary model, or an IPCC-derived value.
            The per-month statistics are detrended with the *fitted*
            slope in both cases, so an override changes only the forward
            extrapolation, not the residual calibration.
        pot_percentile: Threshold percentile for the GPD fits (default 90).
        min_samples_per_month_gpd: Minimum month sample count to attempt
            GPD; below this, both tails are ``None``.
        fallback_amplitude_c: Used when no tmax/tmin pair is available in
            a given month.

    Returns:
        Tuple ``(ThermalModel, CalibrationReport)``.

    Raises:
        ValueError: ``samples`` is empty.
    """
    if not samples:
        raise ValueError("calibrate_thermal_model: no samples provided")

    sample_list = list(samples)
    n = len(sample_list)
    doy = np.asarray([s.day_of_year for s in sample_list], dtype=np.int64)
    tmean = np.asarray([s.t_mean_c for s in sample_list], dtype=float)
    tmax = np.asarray(
        [s.t_max_c if s.t_max_c is not None else np.nan for s in sample_list],
        dtype=float,
    )
    tmin = np.asarray(
        [s.t_min_c if s.t_min_c is not None else np.nan for s in sample_list],
        dtype=float,
    )
    year_offset = np.asarray(
        [s.year_offset if s.year_offset is not None else np.nan for s in sample_list],
        dtype=float,
    )

    # 1. Harmonic seasonal mean
    harmonic, rmse = fit_harmonic_seasonal_mean(doy, tmean)

    # 2. Residuals = observed - harmonic
    residuals = tmean - harmonic.evaluate(doy)

    # 3. Linear warming trend on the residuals, then detrend so the
    #    per-month statistics describe the *stationary* fluctuation.
    fitted_trend = fit_linear_trend(year_offset, residuals)
    if fitted_trend != 0.0:
        x = np.where(np.isfinite(year_offset), year_offset, 0.0)
        # Centre the detrend on the window midpoint so the harmonic a0
        # stays the window-average temperature and the model extrapolates
        # the trend from "now" (the calibration epoch) forward.
        residuals = residuals - fitted_trend * (x - float(np.nanmean(year_offset)))

    # 4. Per-month bookkeeping
    months = month_of_year(doy)
    report = CalibrationReport(
        n_samples=n,
        years_covered=round(n / 365.0, 2),
        rmse_harmonic_c=rmse,
        fitted_trend_c_per_year=fitted_trend,
    )
    monthly_params: list[ThermalMonthParams] = []

    for m in range(12):
        mask = months == m
        m_residuals = residuals[mask]
        m_tmax = tmax[mask]
        m_tmin = tmin[mask]
        report.per_month_n_samples[m] = int(mask.sum())

        sigma, phi = fit_ar1(m_residuals)

        # Diurnal amplitude: monthly mean + residual coupling.
        valid_pairs = ~np.isnan(m_tmax) & ~np.isnan(m_tmin)
        if valid_pairs.any():
            amplitudes_all = (m_tmax - m_tmin) / 2.0
            t_amplitude_c = float(np.nanmean(amplitudes_all[valid_pairs]))
            amp_slope = fit_amplitude_slope(m_residuals, amplitudes_all)
        else:
            t_amplitude_c = fallback_amplitude_c
            amp_slope = 0.0
        report.per_month_amp_slope[m] = amp_slope

        # GPD tails (only if enough samples)
        gpd_upper: GPDTail | None = None
        gpd_lower: GPDTail | None = None
        if m_residuals.size >= min_samples_per_month_gpd:
            gpd_upper = fit_gpd_tail(m_residuals, tail="upper", threshold_percentile=pot_percentile)
            gpd_lower = fit_gpd_tail(m_residuals, tail="lower", threshold_percentile=pot_percentile)

        report.per_month_gpd_upper_fitted[m] = gpd_upper is not None
        report.per_month_gpd_lower_fitted[m] = gpd_lower is not None

        monthly_params.append(
            ThermalMonthParams(
                t_std_residual_c=max(sigma, 0.0),
                persistence_phi=phi,
                t_amplitude_c=max(t_amplitude_c, 0.0),
                gpd_upper=gpd_upper,
                gpd_lower=gpd_lower,
                amp_slope_per_c=amp_slope,
            )
        )

    model = ThermalModel(
        harmonic=harmonic,
        monthly_params=monthly_params,
        climate_trend_c_per_year=(
            fitted_trend
            if climate_trend_c_per_year is None
            else float(climate_trend_c_per_year)
        ),
    )
    return model, report


# ---------------------------------------------------------------------------
# Convenience: build samples from Open-Meteo-style daily arrays
# ---------------------------------------------------------------------------


def samples_from_daily_arrays(
    dates: Sequence[str],
    t_mean_c: Sequence[float | None],
    t_max_c: Sequence[float | None] | None = None,
    t_min_c: Sequence[float | None] | None = None,
) -> list[DailyArchiveSample]:
    """
    Convert parallel daily arrays (typically from Open-Meteo's
    ``daily`` payload) into a list of :class:`DailyArchiveSample`,
    dropping rows with missing ``t_mean_c``.

    The calendar year embedded in each ISO date populates
    :attr:`DailyArchiveSample.year_offset` (years since the first valid
    sample's year), which enables the linear-trend fit in
    :func:`calibrate_thermal_model`.

    Args:
        dates: ISO-format date strings, e.g. ``"2020-01-01"``.
        t_mean_c: Daily mean temperatures (°C) aligned with ``dates``.
            ``None`` / NaN entries are skipped.
        t_max_c: Optional daily max series (same alignment).
        t_min_c: Optional daily min series.

    Returns:
        List of valid :class:`DailyArchiveSample`. Order matches input.

    Raises:
        ValueError: Arrays have mismatched lengths.
    """
    n = len(dates)
    if len(t_mean_c) != n:
        raise ValueError(f"t_mean_c length {len(t_mean_c)} != dates length {n}")
    if t_max_c is not None and len(t_max_c) != n:
        raise ValueError(f"t_max_c length {len(t_max_c)} != dates length {n}")
    if t_min_c is not None and len(t_min_c) != n:
        raise ValueError(f"t_min_c length {len(t_min_c)} != dates length {n}")

    samples: list[DailyArchiveSample] = []
    base_year: int | None = None
    for i, iso_date in enumerate(dates):
        tm = t_mean_c[i]
        if tm is None:
            continue
        try:
            tm_f = float(tm)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(tm_f):
            continue
        try:
            year = int(iso_date[0:4])
            month = int(iso_date[5:7])
            day = int(iso_date[8:10])
        except (ValueError, IndexError):
            continue
        # Cumulative day-of-year (non-leap) from month + day.
        cum = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
        doy = cum[month - 1] + (day - 1)
        doy = max(0, min(364, doy))
        if base_year is None:
            base_year = year
        year_offset = (year - base_year) + doy / 365.0
        tx = _coerce_optional(t_max_c[i]) if t_max_c is not None else None
        tn = _coerce_optional(t_min_c[i]) if t_min_c is not None else None
        samples.append(
            DailyArchiveSample(
                day_of_year=doy,
                t_mean_c=tm_f,
                t_max_c=tx,
                t_min_c=tn,
                year_offset=year_offset,
            )
        )
    return samples


def _coerce_optional(value: float | None) -> float | None:
    """Convert to float or return None on NaN / None / invalid."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f
