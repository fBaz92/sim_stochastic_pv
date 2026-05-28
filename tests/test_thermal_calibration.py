"""
Tests for the Phase-15 thermal calibration pipeline.

We feed synthetic daily series with known statistics into
:func:`calibrate_thermal_model` and verify that the recovered
parameters land near the ground truth within MC tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.simulation.thermal import (
    GPDTail,
    HarmonicSeasonalMean,
    ThermalModel,
    ThermalMonthParams,
)
from sim_stochastic_pv.simulation.thermal_calibration import (
    DailyArchiveSample,
    calibrate_thermal_model,
    fit_ar1,
    fit_gpd_tail,
    fit_harmonic_seasonal_mean,
    samples_from_daily_arrays,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_series(
    n_years: int,
    a0: float,
    a1: float,
    a2: float,
    sigma: float,
    phi: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (doy, tmean) for ``n_years`` of synthetic daily means
    following the same model the calibration assumes (harmonic +
    AR(1) noise)."""
    rng = np.random.default_rng(seed)
    n = n_years * 365
    doy = np.arange(n) % 365
    omega = 2 * np.pi / 365.25
    seasonal = a0 + a1 * np.cos(omega * doy) + a2 * np.sin(omega * doy)
    residual = np.zeros(n)
    sigma_innov = sigma * np.sqrt(max(0.0, 1.0 - phi * phi))
    prev = 0.0
    for d in range(n):
        e = rng.standard_normal() * sigma_innov
        prev = phi * prev + e
        residual[d] = prev
    return doy, seasonal + residual


# ---------------------------------------------------------------------------
# Harmonic fit
# ---------------------------------------------------------------------------


class TestFitHarmonic:
    def test_recovers_known_coefficients(self) -> None:
        doy, tmean = _make_synthetic_series(
            n_years=20, a0=12.0, a1=-9.0, a2=2.0, sigma=1.5, phi=0.6, seed=1
        )
        harmonic, rmse = fit_harmonic_seasonal_mean(doy, tmean)
        assert harmonic.a0 == pytest.approx(12.0, abs=0.3)
        assert harmonic.a1 == pytest.approx(-9.0, abs=0.3)
        assert harmonic.a2 == pytest.approx(2.0, abs=0.3)
        assert rmse < 2.5

    def test_perfect_fit_when_no_noise(self) -> None:
        doy = np.arange(365)
        omega = 2 * np.pi / 365.25
        tmean = 10.0 - 5.0 * np.cos(omega * doy) + 3.0 * np.sin(omega * doy)
        harmonic, rmse = fit_harmonic_seasonal_mean(doy, tmean)
        assert harmonic.a0 == pytest.approx(10.0, abs=1e-6)
        assert harmonic.a1 == pytest.approx(-5.0, abs=1e-6)
        assert harmonic.a2 == pytest.approx(3.0, abs=1e-6)
        assert rmse < 1e-6

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError):
            fit_harmonic_seasonal_mean(np.array([], dtype=int), np.array([]))


# ---------------------------------------------------------------------------
# AR(1) fit
# ---------------------------------------------------------------------------


class TestFitAR1:
    def test_recovers_phi(self) -> None:
        rng = np.random.default_rng(2)
        n = 10_000
        sigma_target = 2.0
        phi_target = 0.75
        sigma_innov = sigma_target * np.sqrt(1 - phi_target ** 2)
        residual = np.zeros(n)
        prev = 0.0
        for d in range(n):
            e = rng.standard_normal() * sigma_innov
            prev = phi_target * prev + e
            residual[d] = prev
        sigma, phi = fit_ar1(residual)
        assert sigma == pytest.approx(sigma_target, rel=0.1)
        assert phi == pytest.approx(phi_target, abs=0.03)

    def test_empty_returns_zero(self) -> None:
        assert fit_ar1(np.array([])) == (0.0, 0.0)
        assert fit_ar1(np.array([1.0])) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# GPD tail fit
# ---------------------------------------------------------------------------


class TestFitGPDTail:
    def test_upper_tail_returns_positive_threshold(self) -> None:
        """A right-skewed residual distribution yields a positive
        threshold and a finite, sensible GPD."""
        rng = np.random.default_rng(3)
        # Exponential noise → naturally heavy-tailed.
        residuals = rng.exponential(scale=2.0, size=5_000)
        tail = fit_gpd_tail(residuals, tail="upper", threshold_percentile=90.0)
        assert tail is not None
        assert tail.threshold > 0
        assert tail.scale > 0
        assert 0.05 <= tail.exceedance_prob <= 0.12

    def test_lower_tail_returns_magnitude(self) -> None:
        """A left-skewed distribution yields a positive threshold
        magnitude in the returned GPDTail (the simulator interprets
        it as the *negative* threshold)."""
        rng = np.random.default_rng(4)
        residuals = -rng.exponential(scale=2.0, size=5_000)
        tail = fit_gpd_tail(residuals, tail="lower", threshold_percentile=90.0)
        assert tail is not None
        assert tail.threshold > 0   # stored as magnitude
        assert tail.scale > 0

    def test_too_few_excesses_returns_none(self) -> None:
        residuals = np.array([0.0, 1.0, 2.0])
        assert fit_gpd_tail(residuals, tail="upper") is None

    def test_invalid_tail_raises(self) -> None:
        with pytest.raises(ValueError):
            fit_gpd_tail(np.zeros(10), tail="sideways")


# ---------------------------------------------------------------------------
# Top-level calibrate_thermal_model
# ---------------------------------------------------------------------------


class TestCalibrateThermalModel:
    def test_full_pipeline_recovers_harmonic_and_ar1(self) -> None:
        """20 years of synthetic data should let the calibration recover
        the harmonic coefficients within ±0.5 °C and per-month σ within
        ±20%."""
        doy, tmean = _make_synthetic_series(
            n_years=20, a0=12.0, a1=-10.0, a2=1.0, sigma=2.0, phi=0.7, seed=5
        )
        samples = [
            DailyArchiveSample(
                day_of_year=int(d),
                t_mean_c=float(t),
                t_max_c=float(t) + 4.0,
                t_min_c=float(t) - 4.0,
            )
            for d, t in zip(doy, tmean)
        ]
        model, report = calibrate_thermal_model(samples)

        assert report.n_samples == len(samples)
        assert report.years_covered == pytest.approx(20.0, abs=0.1)
        assert report.rmse_harmonic_c < 3.0
        assert model.harmonic.a0 == pytest.approx(12.0, abs=0.3)
        assert model.harmonic.a1 == pytest.approx(-10.0, abs=0.3)
        assert model.harmonic.a2 == pytest.approx(1.0, abs=0.5)

        # Per-month σ should converge to the ground truth, allowing for
        # cross-month variability around the 20-year window.
        sigmas = [p.t_std_residual_c for p in model.monthly_params]
        assert min(sigmas) > 1.4
        assert max(sigmas) < 2.7

        # All months should produce a diurnal amplitude of about 4°C
        # because we built tmax/tmin = tmean ± 4.
        amplitudes = [p.t_amplitude_c for p in model.monthly_params]
        assert all(3.5 < a < 4.5 for a in amplitudes)

    def test_no_tmax_tmin_falls_back_to_default_amplitude(self) -> None:
        """When tmax/tmin missing, calibration uses
        DEFAULT_FALLBACK_AMPLITUDE_C for the amplitude."""
        doy, tmean = _make_synthetic_series(
            n_years=5, a0=10.0, a1=-8.0, a2=0.0, sigma=1.5, phi=0.5, seed=6
        )
        samples = [
            DailyArchiveSample(day_of_year=int(d), t_mean_c=float(t))
            for d, t in zip(doy, tmean)
        ]
        model, _ = calibrate_thermal_model(samples, fallback_amplitude_c=3.7)
        for p in model.monthly_params:
            assert p.t_amplitude_c == pytest.approx(3.7, abs=1e-6)

    def test_empty_samples_raises(self) -> None:
        with pytest.raises(ValueError):
            calibrate_thermal_model([])

    def test_short_window_skips_gpd(self) -> None:
        """A 1-year window has ~30 samples per month — below the default
        ``min_samples_per_month_gpd=60``, so all GPD tails should be
        skipped."""
        doy, tmean = _make_synthetic_series(
            n_years=1, a0=10.0, a1=-8.0, a2=0.0, sigma=1.0, phi=0.5, seed=7
        )
        samples = [
            DailyArchiveSample(day_of_year=int(d), t_mean_c=float(t))
            for d, t in zip(doy, tmean)
        ]
        model, report = calibrate_thermal_model(samples)
        assert not any(report.per_month_gpd_upper_fitted)
        assert not any(report.per_month_gpd_lower_fitted)
        for p in model.monthly_params:
            assert p.gpd_upper is None
            assert p.gpd_lower is None

    def test_simulator_round_trip(self) -> None:
        """Calibrate from synthetic data, then simulate from the
        calibrated model and verify the harmonic-fit RMSE is reasonable."""
        doy, tmean = _make_synthetic_series(
            n_years=10, a0=12.0, a1=-9.0, a2=0.5, sigma=2.0, phi=0.6, seed=8
        )
        samples = [
            DailyArchiveSample(day_of_year=int(d), t_mean_c=float(t))
            for d, t in zip(doy, tmean)
        ]
        model, _ = calibrate_thermal_model(samples)
        rng = np.random.default_rng(123)
        simulated = model.simulate_daily_means(10 * 365, rng)
        # Resimulated series stationary-ish around recovered harmonic.
        omega = 2 * np.pi / 365.25
        sim_doy = np.arange(simulated.size) % 365
        seasonal = (
            model.harmonic.a0
            + model.harmonic.a1 * np.cos(omega * sim_doy)
            + model.harmonic.a2 * np.sin(omega * sim_doy)
        )
        residual = simulated - seasonal
        assert abs(residual.mean()) < 0.5      # near-zero mean
        assert 1.0 < residual.std() < 3.5      # in the right ballpark


# ---------------------------------------------------------------------------
# samples_from_daily_arrays — Open-Meteo adapter
# ---------------------------------------------------------------------------


class TestSamplesFromDailyArrays:
    def test_parses_iso_dates_to_doy(self) -> None:
        dates = ["2020-01-01", "2020-02-01", "2020-12-31"]
        tmean = [-2.0, 4.0, 1.0]
        samples = samples_from_daily_arrays(dates, tmean)
        assert [s.day_of_year for s in samples] == [0, 31, 364]
        assert [s.t_mean_c for s in samples] == [-2.0, 4.0, 1.0]

    def test_drops_none_and_nan_rows(self) -> None:
        dates = ["2020-01-01", "2020-01-02", "2020-01-03"]
        tmean = [None, float("nan"), 3.0]
        samples = samples_from_daily_arrays(dates, tmean)
        assert len(samples) == 1
        assert samples[0].t_mean_c == 3.0

    def test_tmax_tmin_optional(self) -> None:
        dates = ["2020-07-15"]
        samples = samples_from_daily_arrays(dates, [25.0])
        assert samples[0].t_max_c is None
        assert samples[0].t_min_c is None

        samples2 = samples_from_daily_arrays(
            dates, [25.0], t_max_c=[30.0], t_min_c=[20.0]
        )
        assert samples2[0].t_max_c == 30.0
        assert samples2[0].t_min_c == 20.0

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            samples_from_daily_arrays(["2020-01-01"], [1.0, 2.0])
