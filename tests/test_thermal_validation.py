"""
Tests for the extremes backtest harness and the calibration corrections
that feed it (fitted warming trend, clear-sky amplitude coupling, GPD
tail replacement).

Statistical assertions follow the project rule: properties with explicit
tolerances, fixed seeds, no per-value checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim_stochastic_pv.simulation.thermal import (
    HarmonicSeasonalMean,
    ThermalModel,
    ThermalMonthParams,
)
from sim_stochastic_pv.simulation.thermal_calibration import (
    calibrate_thermal_model,
    samples_from_daily_arrays,
)
from sim_stochastic_pv.simulation.thermal_validation import (
    backtest_annual_extremes,
    observed_annual_extremes,
)


# ---------------------------------------------------------------------------
# Synthetic archive generator
# ---------------------------------------------------------------------------


def _synthetic_archive(
    n_years: int = 10,
    trend_c_per_year: float = 0.0,
    amp_mean: float = 5.0,
    amp_slope: float = 0.0,
    sigma: float = 2.0,
    phi: float = 0.7,
    seed: int = 7,
) -> tuple[list[str], list[float], list[float], list[float]]:
    """Generate (dates, tmean, tmax, tmin) with known ground truth.

    Daily mean = sinusoid + trend + AR(1) noise; the half-amplitude of
    the diurnal swing is ``amp_mean + amp_slope · residual`` so the
    calibration's amplitude regression has a known target.
    """
    rng = np.random.default_rng(seed)
    dates: list[str] = []
    tmean: list[float] = []
    tmax: list[float] = []
    tmin: list[float] = []
    residual = 0.0
    sigma_innov = sigma * math.sqrt(1.0 - phi * phi)
    cum = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365)
    for y in range(n_years):
        year = 2014 + y
        for doy in range(365):
            month = next(m for m in range(12) if cum[m] <= doy < cum[m + 1])
            day = doy - cum[month] + 1
            dates.append(f"{year:04d}-{month + 1:02d}-{day:02d}")
            seasonal = 12.0 - 10.0 * math.cos(2 * math.pi * doy / 365.25)
            residual = phi * residual + rng.standard_normal() * sigma_innov
            mean = seasonal + trend_c_per_year * (y + doy / 365.0) + residual
            amp = max(0.5, amp_mean + amp_slope * residual)
            tmean.append(mean)
            tmax.append(mean + amp)
            tmin.append(mean - amp)
    return dates, tmean, tmax, tmin


# ---------------------------------------------------------------------------
# Calibration: fitted trend
# ---------------------------------------------------------------------------


class TestFittedTrend:
    def test_trend_recovered_within_tolerance(self) -> None:
        """A +0.10 °C/year synthetic trend is recovered by the fit.

        The OLS slope estimator on AR(1) noise has a large variance on
        short windows (the effective sample size shrinks by
        (1+φ)/(1−φ)), so the property is tested on a 20-year window with
        moderate noise rather than tightening the tolerance artificially.
        """
        dates, tmean, tmax, tmin = _synthetic_archive(
            n_years=20, trend_c_per_year=0.10, sigma=1.0, phi=0.5
        )
        samples = samples_from_daily_arrays(dates, tmean, tmax, tmin)
        model, report = calibrate_thermal_model(samples)
        assert report.fitted_trend_c_per_year == pytest.approx(0.10, abs=0.04)
        # Default (None) bakes the fitted trend into the model.
        assert model.climate_trend_c_per_year == report.fitted_trend_c_per_year

    def test_explicit_override_wins_over_fitted(self) -> None:
        """Passing an explicit trend forces the model value, while the
        report still records what the data said."""
        dates, tmean, tmax, tmin = _synthetic_archive(
            n_years=20, trend_c_per_year=0.10, sigma=1.0, phi=0.5
        )
        samples = samples_from_daily_arrays(dates, tmean, tmax, tmin)
        model, report = calibrate_thermal_model(
            samples, climate_trend_c_per_year=0.0
        )
        assert model.climate_trend_c_per_year == 0.0
        assert report.fitted_trend_c_per_year == pytest.approx(0.10, abs=0.04)

    def test_no_year_info_means_no_trend(self) -> None:
        """Samples without year_offset → trend stays 0 (legacy behaviour)."""
        dates, tmean, tmax, tmin = _synthetic_archive(n_years=5)
        samples = samples_from_daily_arrays(dates, tmean, tmax, tmin)
        stripped = [
            type(s)(
                day_of_year=s.day_of_year,
                t_mean_c=s.t_mean_c,
                t_max_c=s.t_max_c,
                t_min_c=s.t_min_c,
                year_offset=None,
            )
            for s in samples
        ]
        model, report = calibrate_thermal_model(stripped)
        assert report.fitted_trend_c_per_year == 0.0
        assert model.climate_trend_c_per_year == 0.0

    def test_detrending_keeps_sigma_clean(self) -> None:
        """With a strong trend, detrended residual σ should match the
        no-trend ground truth instead of absorbing the drift."""
        base_dates, base_mean, base_max, base_min = _synthetic_archive(
            n_years=10, trend_c_per_year=0.0, seed=11
        )
        base_model, _ = calibrate_thermal_model(
            samples_from_daily_arrays(base_dates, base_mean, base_max, base_min)
        )
        dates, tmean, tmax, tmin = _synthetic_archive(
            n_years=10, trend_c_per_year=0.3, seed=11
        )
        model, _ = calibrate_thermal_model(
            samples_from_daily_arrays(dates, tmean, tmax, tmin)
        )
        # Same seed → same noise; σ per month should be close despite the
        # 3 °C drift across the trended window.
        for m in range(12):
            s_trend = model.monthly_params[m].t_std_residual_c
            s_base = base_model.monthly_params[m].t_std_residual_c
            assert s_trend == pytest.approx(s_base, rel=0.15)


# ---------------------------------------------------------------------------
# Calibration: amplitude-residual coupling
# ---------------------------------------------------------------------------


class TestAmplitudeSlope:
    def test_slope_recovered(self) -> None:
        """A synthetic amp = 5 + 0.3·residual is recovered per month."""
        dates, tmean, tmax, tmin = _synthetic_archive(
            n_years=10, amp_slope=0.3
        )
        samples = samples_from_daily_arrays(dates, tmean, tmax, tmin)
        model, report = calibrate_thermal_model(samples)
        for m in range(12):
            assert report.per_month_amp_slope[m] == pytest.approx(0.3, abs=0.05)
            assert model.monthly_params[m].amp_slope_per_c == pytest.approx(
                0.3, abs=0.05
            )

    def test_zero_slope_data_yields_zero_slope(self) -> None:
        dates, tmean, tmax, tmin = _synthetic_archive(n_years=10, amp_slope=0.0)
        samples = samples_from_daily_arrays(dates, tmean, tmax, tmin)
        _, report = calibrate_thermal_model(samples)
        for m in range(12):
            assert abs(report.per_month_amp_slope[m]) < 0.05

    def test_hourly_max_grows_with_positive_slope(self) -> None:
        """With a positive slope, hot days swing wider: the maximum of
        the hourly series exceeds the constant-amplitude equivalent."""
        params_flat = [
            ThermalMonthParams(
                t_std_residual_c=2.0, persistence_phi=0.7, t_amplitude_c=5.0
            )
            for _ in range(12)
        ]
        params_coupled = [
            ThermalMonthParams(
                t_std_residual_c=2.0,
                persistence_phi=0.7,
                t_amplitude_c=5.0,
                amp_slope_per_c=0.3,
            )
            for _ in range(12)
        ]
        harmonic = HarmonicSeasonalMean(a0=12.0, a1=-10.0, a2=0.0)
        flat = ThermalModel(harmonic, params_flat)
        coupled = ThermalModel(harmonic, params_coupled)
        daily = flat.simulate_daily_means(365 * 3, np.random.default_rng(3))
        hourly_flat = flat.to_hourly(daily)
        hourly_coupled = coupled.to_hourly(daily)
        assert hourly_coupled.max() > hourly_flat.max() + 0.5
        # And zero slope reproduces the legacy output bit-for-bit.
        params_zero = [
            ThermalMonthParams(
                t_std_residual_c=2.0,
                persistence_phi=0.7,
                t_amplitude_c=5.0,
                amp_slope_per_c=0.0,
            )
            for _ in range(12)
        ]
        zero = ThermalModel(harmonic, params_zero)
        np.testing.assert_array_equal(zero.to_hourly(daily), hourly_flat)


# ---------------------------------------------------------------------------
# Observed extremes extraction
# ---------------------------------------------------------------------------


class TestObservedExtremes:
    def test_partial_years_are_skipped(self) -> None:
        dates, tmean, tmax, tmin = _synthetic_archive(n_years=3)
        # Truncate the last year to ~60 days.
        cut = 2 * 365 + 60
        years, obs_max, obs_min = observed_annual_extremes(
            dates[:cut], tmax[:cut], tmin[:cut]
        )
        assert years == [2014, 2015]
        assert len(obs_max) == 2 and len(obs_min) == 2
        # Summer maxima of the sinusoid (+22 °C peak + amplitude + noise).
        assert all(v > 20.0 for v in obs_max)
        assert all(v < 5.0 for v in obs_min)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            observed_annual_extremes(["2020-01-01"], [1.0, 2.0], [0.0])


# ---------------------------------------------------------------------------
# Backtest end-to-end on synthetic ground truth
# ---------------------------------------------------------------------------


class TestBacktest:
    def test_self_consistent_calibration_passes_backtest(self) -> None:
        """Calibrating on synthetic data and backtesting against the same
        data must give high coverage and small median bias — this is the
        property that failed before the Fase-28 corrections."""
        dates, tmean, tmax, tmin = _synthetic_archive(
            n_years=10, trend_c_per_year=0.08, amp_slope=0.25, seed=23
        )
        samples = samples_from_daily_arrays(dates, tmean, tmax, tmin)
        model, _ = calibrate_thermal_model(samples)
        result = backtest_annual_extremes(
            model, dates, tmax, tmin, n_paths=30, seed=5
        )
        assert result.n_years == 10
        assert len(result.observed_years) == 10
        assert result.tmax_coverage >= 0.6
        assert result.tmin_coverage >= 0.6
        assert abs(result.tmax_median_bias_c) <= 1.5
        assert abs(result.tmin_median_bias_c) <= 1.5
        # Band sanity: p05 < p50 < p95 on both tails.
        assert result.sim_tmax_p05 < result.sim_tmax_p50 < result.sim_tmax_p95
        assert result.sim_tmin_p05 < result.sim_tmin_p50 < result.sim_tmin_p95

    def test_deliberately_wrong_model_fails_backtest(self) -> None:
        """A model 5 °C colder than the data must be flagged by the bias
        metric — guards against a vacuously-passing harness."""
        dates, tmean, tmax, tmin = _synthetic_archive(n_years=8, seed=29)
        samples = samples_from_daily_arrays(dates, tmean, tmax, tmin)
        model, _ = calibrate_thermal_model(samples)
        cold = ThermalModel(
            harmonic=HarmonicSeasonalMean(
                a0=model.harmonic.a0 - 5.0,
                a1=model.harmonic.a1,
                a2=model.harmonic.a2,
            ),
            monthly_params=model.monthly_params,
            climate_trend_c_per_year=model.climate_trend_c_per_year,
        )
        result = backtest_annual_extremes(
            cold, dates, tmax, tmin, n_paths=20, seed=5
        )
        assert result.tmax_median_bias_c < -3.0

    def test_too_short_archive_yields_no_verdict_metrics(self) -> None:
        dates, tmean, tmax, tmin = _synthetic_archive(n_years=1)
        samples = samples_from_daily_arrays(dates, tmean, tmax, tmin)
        model, _ = calibrate_thermal_model(samples)
        # 100 days only → no complete observed year.
        result = backtest_annual_extremes(
            model, dates[:100], tmax[:100], tmin[:100], n_paths=10, seed=1
        )
        assert result.observed_years == []
        assert result.tmax_coverage == 0.0
        assert result.tmax_median_bias_c == 0.0

    def test_invalid_inputs_raise(self) -> None:
        dates, tmean, tmax, tmin = _synthetic_archive(n_years=1)
        model, _ = calibrate_thermal_model(
            samples_from_daily_arrays(dates, tmean, tmax, tmin)
        )
        with pytest.raises(ValueError):
            backtest_annual_extremes(model, dates, tmax, tmin, n_paths=0)
        with pytest.raises(ValueError):
            backtest_annual_extremes(model, [], [], [], n_paths=10)
