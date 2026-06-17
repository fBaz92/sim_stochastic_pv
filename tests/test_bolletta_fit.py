"""Tests for the bill-based load-profile auto-fit."""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.simulation.load_profiles import (
    ARERA_BASELINE_ANNUAL_KWH,
    AreraLoadProfile,
    BL_TABLE,
    DEFAULT_PRESENCE_CALENDAR,
    MonthPresencePattern,
    PresenceCalendar,
    annual_kwh_from_bimonthly,
    build_scaled_arera_factory,
    compute_arera_baseline_annual_kwh,
    fit_bolletta_profile,
)


def _uniform(pattern: MonthPresencePattern) -> PresenceCalendar:
    return PresenceCalendar(months=tuple(pattern for _ in range(12)))


# Always-home and never-home reference calendars.
ALWAYS_HOME = _uniform(MonthPresencePattern(full_weeks=5, weekends=True))
HALF_TIME = _uniform(MonthPresencePattern(full_weeks=2, extra_weekdays=1, weekends=False))


class TestAreraBaseline:
    """compute_arera_baseline_annual_kwh behaviour."""

    def test_default_is_about_1196(self):
        assert compute_arera_baseline_annual_kwh() == pytest.approx(1196.0, abs=5.0)
        assert ARERA_BASELINE_ANNUAL_KWH == pytest.approx(1196.0, abs=5.0)

    def test_scaling_is_linear(self):
        base = compute_arera_baseline_annual_kwh()
        doubled = compute_arera_baseline_annual_kwh(BL_TABLE * 2.0)
        assert doubled == pytest.approx(2.0 * base)

    def test_deterministic(self):
        assert compute_arera_baseline_annual_kwh() == compute_arera_baseline_annual_kwh()

    def test_rejects_bad_shape(self):
        with pytest.raises(ValueError):
            compute_arera_baseline_annual_kwh(np.zeros((3, 12)))


class TestAnnualFromBimonthly:
    """annual_kwh_from_bimonthly reduction."""

    def test_sums_six_periods(self):
        assert annual_kwh_from_bimonthly([400, 350, 300, 280, 320, 450]) == 2100.0

    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            annual_kwh_from_bimonthly([100, 200, 300])

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            annual_kwh_from_bimonthly([400, 350, 300, 280, 320, -10])


class TestFitBollettaProfile:
    """fit_bolletta_profile scaling logic."""

    def test_identity_fit_when_target_equals_baseline(self):
        """target == ARERA baseline → home_scale ≈ 1, independent of presence."""
        for cal in (ALWAYS_HOME, HALF_TIME, DEFAULT_PRESENCE_CALENDAR):
            fit = fit_bolletta_profile(ARERA_BASELINE_ANNUAL_KWH, cal)
            assert fit["home_scale_factor"] == pytest.approx(1.0, rel=1e-6)

    def test_energy_split_sums_to_target(self):
        fit = fit_bolletta_profile(2400.0, DEFAULT_PRESENCE_CALENDAR)
        total = fit["estimated_home_kwh"] + fit["estimated_away_kwh"]
        assert total == pytest.approx(2400.0, rel=1e-6)

    def test_doubling_target_raises_scale(self):
        """Always-home: doubling the bill doubles the home scale (1→2)."""
        single = fit_bolletta_profile(ARERA_BASELINE_ANNUAL_KWH, ALWAYS_HOME)
        double = fit_bolletta_profile(2.0 * ARERA_BASELINE_ANNUAL_KWH, ALWAYS_HOME)
        assert single["home_scale_factor"] == pytest.approx(1.0, rel=1e-6)
        assert double["home_scale_factor"] == pytest.approx(2.0, rel=1e-6)

    def test_less_presence_needs_higher_scale(self):
        """Same bill, less time home → each home day must consume more."""
        target = 3000.0
        full = fit_bolletta_profile(target, ALWAYS_HOME)
        part = fit_bolletta_profile(target, HALF_TIME)
        assert part["home_scale_factor"] > full["home_scale_factor"]

    def test_low_target_keeps_scale_positive(self):
        """An implausibly low bill floors (but never inverts) the home scale."""
        fit = fit_bolletta_profile(50.0, DEFAULT_PRESENCE_CALENDAR)
        assert fit["home_scale_factor"] > 0.0

    def test_scale_always_positive_across_targets(self):
        for target in (10.0, 500.0, 1196.0, 5000.0, 12000.0):
            fit = fit_bolletta_profile(target, DEFAULT_PRESENCE_CALENDAR)
            assert fit["home_scale_factor"] > 0.0

    def test_rejects_non_positive_target(self):
        with pytest.raises(ValueError):
            fit_bolletta_profile(0.0, DEFAULT_PRESENCE_CALENDAR)

    def test_derived_profile_data_shape(self):
        """The savable data block carries the level, calendar and derived scale."""
        fit = fit_bolletta_profile(2400.0, DEFAULT_PRESENCE_CALENDAR, house_type="apartment_standard")
        data = fit["derived_profile_data"]
        assert data["input_level"] == "bolletta"
        assert "months" in data["presence_calendar"]
        assert data["bolletta"]["annual_kwh"] == 2400.0
        assert data["bolletta"]["house_type"] == "apartment_standard"
        assert data["_derived"]["home_scale_factor"] == pytest.approx(
            fit["home_scale_factor"]
        )

    def test_min_max_days_match_calendar(self):
        fit = fit_bolletta_profile(2400.0, DEFAULT_PRESENCE_CALENDAR)
        expected_min, expected_max = DEFAULT_PRESENCE_CALENDAR.to_min_max_days_home()
        assert fit["min_days_home"] == expected_min
        assert fit["max_days_home"] == expected_max


class TestBuildScaledAreraFactory:
    """build_scaled_arera_factory output."""

    def _load_at(self, profile, hour=10, weekday=0, month=0):
        """Sample one F1-band hour (Monday 10:00, January)."""
        return profile.get_hourly_load_kw(
            year_index=0, month_in_year=month, day_in_month=0,
            hour_in_day=hour, weekday=weekday,
        )

    def test_factor_one_matches_default(self):
        scaled = build_scaled_arera_factory(1.0)()
        default = AreraLoadProfile()
        assert self._load_at(scaled) == pytest.approx(self._load_at(default))

    def test_factor_two_doubles_load(self):
        scaled = build_scaled_arera_factory(2.0)()
        default = AreraLoadProfile()
        assert self._load_at(scaled) == pytest.approx(2.0 * self._load_at(default))

    def test_factory_makes_fresh_instances(self):
        factory = build_scaled_arera_factory(1.5)
        assert factory() is not factory()

    def test_rejects_non_positive_scale(self):
        with pytest.raises(ValueError):
            build_scaled_arera_factory(0.0)
