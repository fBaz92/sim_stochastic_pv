"""
Unit tests for the Phase-15 ThermalModel core.

Verifies the statistical properties of the synthetic temperature series:
- vectorised seasonal harmonic eval matches the closed form;
- AR(1) lag-1 autocorrelation of residuals matches the requested φ within
  tolerance (Monte Carlo over enough samples);
- the GPD tail injection actually generates the requested exceedance
  frequency and magnitudes;
- diurnal interpolation preserves the daily mean by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.simulation.thermal import (
    DAYS_IN_YEAR,
    GPDTail,
    HarmonicSeasonalMean,
    ThermalModel,
    ThermalMonthParams,
    month_of_year,
    simulate_temperature_preview,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _flat_monthly_params(
    std: float = 2.0,
    phi: float = 0.7,
    amp: float = 5.0,
    gpd_upper: GPDTail | None = None,
    gpd_lower: GPDTail | None = None,
) -> list[ThermalMonthParams]:
    """Build 12 identical ThermalMonthParams (lets the seasonal harmonic do
    the seasonality on its own)."""
    return [
        ThermalMonthParams(
            t_std_residual_c=std,
            persistence_phi=phi,
            t_amplitude_c=amp,
            gpd_upper=gpd_upper,
            gpd_lower=gpd_lower,
        )
        for _ in range(12)
    ]


def _trivial_model(**kwargs) -> ThermalModel:
    harmonic = HarmonicSeasonalMean(a0=12.0, a1=-10.0, a2=0.0)
    return ThermalModel(harmonic, _flat_monthly_params(**kwargs))


# ---------------------------------------------------------------------------
# month_of_year + harmonic
# ---------------------------------------------------------------------------


class TestMonthOfYear:
    def test_boundaries(self) -> None:
        assert month_of_year(0) == 0           # Jan 1
        assert month_of_year(30) == 0          # Jan 31
        assert month_of_year(31) == 1          # Feb 1
        assert month_of_year(58) == 1          # Feb 28
        assert month_of_year(59) == 2          # Mar 1
        assert month_of_year(364) == 11        # Dec 31

    def test_vectorised(self) -> None:
        doy = np.array([0, 31, 59, 364])
        np.testing.assert_array_equal(month_of_year(doy), [0, 1, 2, 11])

    def test_out_of_range_is_clamped(self) -> None:
        assert month_of_year(-5) == 0
        assert month_of_year(500) == 11


class TestHarmonicSeasonalMean:
    def test_constant_when_amplitudes_zero(self) -> None:
        h = HarmonicSeasonalMean(a0=10.0, a1=0.0, a2=0.0)
        for d in [0, 100, 364]:
            assert h.evaluate(d) == pytest.approx(10.0)

    def test_peak_and_trough(self) -> None:
        # a1 = -10 puts the trough at day 0 and the peak at day 365/2.
        h = HarmonicSeasonalMean(a0=12.0, a1=-10.0, a2=0.0)
        peak_day = int(DAYS_IN_YEAR / 2)
        assert h.evaluate(0) == pytest.approx(2.0, abs=1e-6)
        assert h.evaluate(peak_day) == pytest.approx(22.0, abs=0.1)

    def test_array_input(self) -> None:
        h = HarmonicSeasonalMean(a0=12.0, a1=-10.0, a2=0.0)
        out = h.evaluate(np.array([0, 100, 200]))
        assert out.shape == (3,)


# ---------------------------------------------------------------------------
# GPDTail validation
# ---------------------------------------------------------------------------


class TestGPDTailValidation:
    def test_scale_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            GPDTail(threshold=5.0, shape=0.1, scale=0.0, exceedance_prob=0.05)

    def test_shape_must_be_below_one(self) -> None:
        with pytest.raises(ValueError):
            GPDTail(threshold=5.0, shape=1.0, scale=1.0, exceedance_prob=0.05)

    def test_exceedance_prob_in_range(self) -> None:
        with pytest.raises(ValueError):
            GPDTail(threshold=5.0, shape=0.1, scale=1.0, exceedance_prob=1.5)


# ---------------------------------------------------------------------------
# ThermalModel.simulate_daily_means — basic shape / determinism
# ---------------------------------------------------------------------------


class TestSimulateDailyMeansBasic:
    def test_returns_expected_shape(self) -> None:
        model = _trivial_model()
        rng = np.random.default_rng(42)
        out = model.simulate_daily_means(365, rng)
        assert out.shape == (365,)
        assert np.isfinite(out).all()

    def test_reproducible_with_same_seed(self) -> None:
        model = _trivial_model()
        a = model.simulate_daily_means(365, np.random.default_rng(123))
        b = model.simulate_daily_means(365, np.random.default_rng(123))
        np.testing.assert_array_equal(a, b)

    def test_seasonal_shape_preserved_no_noise(self) -> None:
        """With σ=0 and no GPD, the simulator output is exactly the
        deterministic seasonal harmonic — a sanity check on the
        vectorised seasonal computation."""
        model = ThermalModel(
            HarmonicSeasonalMean(a0=12.0, a1=-10.0, a2=0.0),
            _flat_monthly_params(std=0.0, phi=0.0, amp=0.0),
        )
        out = model.simulate_daily_means(365, np.random.default_rng(1))
        expected = model.harmonic.evaluate(np.arange(365))
        np.testing.assert_allclose(out, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# AR(1) statistical properties — Monte Carlo with tolerances
# ---------------------------------------------------------------------------


class TestAR1Properties:
    def test_residual_std_matches_target(self) -> None:
        """For a stationary AR(1) the marginal std should converge to the
        ``t_std_residual_c`` of the params, within MC tolerance.

        We use a constant seasonal mean (a1=a2=0) so the simulated series
        is just the AR(1) residual offset by a0, and check std(out - a0).
        """
        std_target = 2.5
        model = ThermalModel(
            HarmonicSeasonalMean(a0=10.0, a1=0.0, a2=0.0),
            _flat_monthly_params(std=std_target, phi=0.7, amp=0.0),
        )
        out = model.simulate_daily_means(10_000, np.random.default_rng(7))
        empirical_std = np.std(out - 10.0)
        assert empirical_std == pytest.approx(std_target, rel=0.1)

    def test_lag1_autocorrelation_matches_phi(self) -> None:
        """Empirical lag-1 autocorrelation of the residual converges to
        the requested ``persistence_phi`` within MC tolerance."""
        phi_target = 0.8
        model = ThermalModel(
            HarmonicSeasonalMean(a0=0.0, a1=0.0, a2=0.0),
            _flat_monthly_params(std=2.0, phi=phi_target, amp=0.0),
        )
        out = model.simulate_daily_means(10_000, np.random.default_rng(11))
        r = np.corrcoef(out[:-1], out[1:])[0, 1]
        assert r == pytest.approx(phi_target, abs=0.05)

    def test_zero_phi_is_iid(self) -> None:
        """φ = 0 ⇒ lag-1 autocorrelation ≈ 0 (white noise)."""
        model = ThermalModel(
            HarmonicSeasonalMean(a0=0.0, a1=0.0, a2=0.0),
            _flat_monthly_params(std=2.0, phi=0.0, amp=0.0),
        )
        out = model.simulate_daily_means(5_000, np.random.default_rng(13))
        r = np.corrcoef(out[:-1], out[1:])[0, 1]
        assert abs(r) < 0.05


# ---------------------------------------------------------------------------
# GPD extreme-event injection
# ---------------------------------------------------------------------------


class TestGPDExtremeEvents:
    def test_upper_events_fire_at_ar1_crossing_rate(self) -> None:
        """Tail replacement: events fire when the AR(1) residual crosses
        the POT threshold, so with the threshold at the marginal p90
        (≈ 1.2816·σ for a Gaussian) roughly 10% of days are events.
        Persistence (the replaced residual feeds the next day) pushes the
        rate slightly above the iid value, hence the asymmetric band."""
        upper = GPDTail(threshold=1.2816, shape=0.1, scale=1.0, exceedance_prob=0.1)
        model = ThermalModel(
            HarmonicSeasonalMean(a0=0.0, a1=0.0, a2=0.0),
            _flat_monthly_params(std=1.0, phi=0.5, gpd_upper=upper),
        )
        _, report = model.simulate_daily_means(
            5_000, np.random.default_rng(17), track_events=True,
        )
        empirical_rate = len(report.upper_event_days) / 5_000
        assert 0.07 < empirical_rate < 0.20

    def test_upper_events_never_fire_with_unreachable_threshold(self) -> None:
        """With a threshold far beyond the AR(1) marginal (5σ) the
        replacement never triggers — extreme draws are not injected
        independently of the AR(1) dynamics."""
        upper = GPDTail(threshold=5.0, shape=0.1, scale=2.0, exceedance_prob=0.1)
        model = ThermalModel(
            HarmonicSeasonalMean(a0=0.0, a1=0.0, a2=0.0),
            _flat_monthly_params(std=1.0, phi=0.5, gpd_upper=upper),
        )
        _, report = model.simulate_daily_means(
            5_000, np.random.default_rng(17), track_events=True,
        )
        assert report.upper_event_days == []

    def test_lower_events_produce_negative_excursions(self) -> None:
        """A lower-tail-only model should produce days with residuals
        below the (negative) threshold whenever the AR(1) crosses it."""
        lower = GPDTail(threshold=1.2816, shape=0.1, scale=1.0, exceedance_prob=0.1)
        model = ThermalModel(
            HarmonicSeasonalMean(a0=0.0, a1=0.0, a2=0.0),
            _flat_monthly_params(std=1.0, phi=0.5, gpd_lower=lower),
        )
        out, report = model.simulate_daily_means(
            5_000, np.random.default_rng(19), track_events=True,
        )
        assert len(report.lower_event_days) > 0
        # On firing days the residual sits below the negative threshold.
        firing_residuals = out[report.lower_event_days]
        assert (firing_residuals < -1.2816).all()

    def test_disabled_tails_dont_fire(self) -> None:
        model = _trivial_model()  # no GPD tails at all
        _, report = model.simulate_daily_means(
            2_000, np.random.default_rng(23), track_events=True,
        )
        assert report.upper_event_days == []
        assert report.lower_event_days == []


# ---------------------------------------------------------------------------
# Climate trend
# ---------------------------------------------------------------------------


class TestClimateTrend:
    def test_positive_trend_lifts_year_over_year_mean(self) -> None:
        """A +0.1 °C/year trend → year 9 mean ~0.9 °C above year 0 mean.

        We use σ=0 so the trend is the only thing that moves the curve.
        """
        model = ThermalModel(
            HarmonicSeasonalMean(a0=10.0, a1=0.0, a2=0.0),
            _flat_monthly_params(std=0.0, phi=0.0, amp=0.0),
            climate_trend_c_per_year=0.1,
        )
        n_years = 10
        out = model.simulate_daily_means(n_years * 365, np.random.default_rng(0))
        year_means = [out[y * 365:(y + 1) * 365].mean() for y in range(n_years)]
        # The deterministic seasonal mean is 10°C every year, so year_means
        # should be {10.0, 10.1, 10.2, ..., 10.9} exactly.
        expected = [10.0 + 0.1 * y for y in range(n_years)]
        np.testing.assert_allclose(year_means, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Hourly interpolation
# ---------------------------------------------------------------------------


class TestHourlyInterpolation:
    def test_shape_is_24_per_day(self) -> None:
        model = _trivial_model()
        daily = model.simulate_daily_means(10, np.random.default_rng(0))
        hourly = model.to_hourly(daily)
        assert hourly.shape == (10 * 24,)

    def test_daily_mean_preserved_when_amp_zero(self) -> None:
        """With t_amplitude_c=0 the hourly series should equal the daily
        mean repeated."""
        model = ThermalModel(
            HarmonicSeasonalMean(a0=10.0, a1=0.0, a2=0.0),
            _flat_monthly_params(std=0.0, phi=0.0, amp=0.0),
        )
        daily = model.simulate_daily_means(3, np.random.default_rng(0))
        hourly = model.to_hourly(daily)
        for d in range(3):
            np.testing.assert_allclose(
                hourly[d * 24:(d + 1) * 24], daily[d], atol=1e-9,
            )

    def test_peak_at_14h(self) -> None:
        """With a non-zero amplitude, hour 14 should be the daily max
        and hour 02 the daily min."""
        model = _trivial_model(amp=5.0)
        daily = np.array([20.0])
        hourly = model.to_hourly(daily)
        assert hourly[14] == pytest.approx(20.0 + 5.0, abs=1e-6)
        assert hourly[2] == pytest.approx(20.0 - 5.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Preview helper
# ---------------------------------------------------------------------------


class TestPreviewHelper:
    def test_preview_returns_expected_shapes(self) -> None:
        model = _trivial_model()
        preview = simulate_temperature_preview(model, n_paths=30, n_years=1, seed=42)
        assert preview.days.shape == (365,)
        assert preview.mean_c.shape == (365,)
        assert preview.p05_c.shape == (365,)
        assert preview.p95_c.shape == (365,)
        assert preview.sample_paths_c.shape == (30, 365)
        # Band ordering invariant.
        assert (preview.p05_c <= preview.mean_c).all()
        assert (preview.mean_c <= preview.p95_c).all()

    def test_sample_paths_capped(self) -> None:
        model = _trivial_model()
        preview = simulate_temperature_preview(
            model, n_paths=200, n_years=1, seed=1, max_sample_paths=20,
        )
        assert preview.sample_paths_c.shape == (20, 365)

    def test_reproducible_with_same_seed(self) -> None:
        model = _trivial_model()
        a = simulate_temperature_preview(model, n_paths=10, n_years=1, seed=99)
        b = simulate_temperature_preview(model, n_paths=10, n_years=1, seed=99)
        np.testing.assert_array_equal(a.mean_c, b.mean_c)
        np.testing.assert_array_equal(a.sample_paths_c, b.sample_paths_c)

    def test_monthly_hourly_stats_shape_and_ordering(self) -> None:
        """The 12 monthly hourly distributions have correct shape and the
        percentile ordering p05 ≤ p25 ≤ p50 ≤ p75 ≤ p95 holds in every
        month."""
        model = _trivial_model(amp=5.0)
        preview = simulate_temperature_preview(model, n_paths=20, n_years=1, seed=7)

        for name in ("monthly_p05_c", "monthly_p25_c", "monthly_p50_c",
                     "monthly_p75_c", "monthly_p95_c",
                     "monthly_min_c", "monthly_max_c"):
            arr = getattr(preview, name)
            assert arr.shape == (12,)

        for m in range(12):
            assert preview.monthly_min_c[m] <= preview.monthly_p05_c[m]
            assert preview.monthly_p05_c[m] <= preview.monthly_p25_c[m]
            assert preview.monthly_p25_c[m] <= preview.monthly_p50_c[m]
            assert preview.monthly_p50_c[m] <= preview.monthly_p75_c[m]
            assert preview.monthly_p75_c[m] <= preview.monthly_p95_c[m]
            assert preview.monthly_p95_c[m] <= preview.monthly_max_c[m]

    def test_monthly_includes_diurnal_swing(self) -> None:
        """With a non-zero diurnal amplitude, the per-month p95 must be
        clearly above the daily-mean p95 — proves the hourly view
        captures the afternoon peak that the daily-mean view loses."""
        model = _trivial_model(std=0.1, phi=0.0, amp=8.0)  # big diurnal swing
        preview = simulate_temperature_preview(model, n_paths=10, n_years=1, seed=11)

        # Daily-mean p95 should be ~22 °C in July; the hourly p95 of July
        # should be ~22 + 8 ≈ 30 °C — well above the daily p95.
        july_daily_p95 = float(np.percentile(preview.p95_c[150:212], 50))
        # July = month index 6
        july_hourly_p95 = preview.monthly_p95_c[6]
        assert july_hourly_p95 > july_daily_p95 + 3.0

    def test_monthly_p50_tracks_seasonal_cycle(self) -> None:
        """The 12 monthly medians should follow the seasonal sinusoid:
        January coldest, July warmest."""
        model = _trivial_model(std=1.0, phi=0.0, amp=3.0)
        preview = simulate_temperature_preview(model, n_paths=15, n_years=1, seed=3)
        argmin_month = int(np.argmin(preview.monthly_p50_c))
        argmax_month = int(np.argmax(preview.monthly_p50_c))
        assert argmin_month in (0, 1, 11)   # Dec / Jan / Feb
        assert argmax_month in (5, 6, 7)    # Jun / Jul / Aug
