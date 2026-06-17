"""Tests for the presence calendar (occupancy backbone of home/away profiles)."""

from __future__ import annotations

import pytest

from sim_stochastic_pv.calendar_utils import MONTH_LENGTHS
from sim_stochastic_pv.simulation.load_profiles import (
    DEFAULT_PRESENCE_CALENDAR,
    HOUSE_TYPE_PRESETS,
    MonthPresencePattern,
    PRESENCE_CALENDAR_PRESETS,
    PresenceCalendar,
)


def _uniform(pattern: MonthPresencePattern) -> PresenceCalendar:
    """Helper: a calendar repeating one pattern across all 12 months."""
    return PresenceCalendar(months=tuple(pattern for _ in range(12)))


class TestMonthPresencePatternValidation:
    """Range validation in MonthPresencePattern.__post_init__."""

    def test_defaults_are_valid(self):
        p = MonthPresencePattern()
        assert p.full_weeks == 4
        assert p.weekends is True

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"full_weeks": 6},
            {"full_weeks": -1},
            {"extra_weekdays": 8},
            {"extra_weekdays": -1},
            {"visit_probability": 1.5},
            {"visit_probability": -0.1},
        ],
    )
    def test_out_of_range_raises(self, kwargs):
        with pytest.raises(ValueError):
            MonthPresencePattern(**kwargs)


class TestToMinMaxDaysHome:
    """PresenceCalendar.to_min_max_days_home over representative patterns."""

    def test_full_time_occupies_every_day(self):
        """full_weeks=5 saturates each month to its calendar length."""
        cal = _uniform(MonthPresencePattern(full_weeks=5, weekends=True))
        min_days, max_days = cal.to_min_max_days_home()
        assert min_days == list(MONTH_LENGTHS)
        assert max_days == list(MONTH_LENGTHS)

    def test_weekends_only_is_about_nine_days(self):
        """A weekend-only building is home ~2/7 of each month."""
        cal = _uniform(MonthPresencePattern(full_weeks=0, weekends=True))
        min_days, _ = cal.to_min_max_days_home()
        # 31-day month → round(31*2/7) = 9; 28-day month → 8.
        assert min_days[0] == 9  # January (31 days)
        assert min_days[1] == 8  # February (28 days)
        for m, lo in enumerate(min_days):
            assert 0 <= lo <= MONTH_LENGTHS[m]

    def test_no_weekends_floor_is_full_weeks_plus_extra(self):
        """With weekends off the floor is exactly full_weeks*7 + extra."""
        cal = _uniform(
            MonthPresencePattern(full_weeks=3, extra_weekdays=2, weekends=False)
        )
        min_days, max_days = cal.to_min_max_days_home()
        assert all(lo == 23 for lo in min_days)  # 3*7 + 2
        assert max_days == min_days  # visit_probability defaults to 0

    def test_vacation_pattern_peaks_in_summer(self):
        """The summer_vacation preset is near-full Jun–Aug, sparse otherwise."""
        cal = PRESENCE_CALENDAR_PRESETS["summer_vacation"]
        min_days, _ = cal.to_min_max_days_home()
        for summer_month in (5, 6, 7):  # Jun, Jul, Aug
            assert min_days[summer_month] >= 28
        for winter_month in (0, 1, 11):  # Jan, Feb, Dec
            assert min_days[winter_month] < 12

    def test_bounds_are_clamped_to_month_length(self):
        """An over-specified month never exceeds its calendar length."""
        cal = _uniform(
            MonthPresencePattern(
                full_weeks=5, extra_weekdays=7, weekends=True, visit_probability=1.0
            )
        )
        min_days, max_days = cal.to_min_max_days_home()
        for m in range(12):
            assert min_days[m] == MONTH_LENGTHS[m]
            assert max_days[m] == MONTH_LENGTHS[m]
            assert min_days[m] <= max_days[m]

    def test_visit_probability_widens_band(self):
        """visit_probability>0 lifts the max above the min; 0 keeps them equal."""
        no_visits = _uniform(MonthPresencePattern(full_weeks=0, weekends=True))
        with_visits = _uniform(
            MonthPresencePattern(full_weeks=0, weekends=True, visit_probability=0.5)
        )
        lo_n, hi_n = no_visits.to_min_max_days_home()
        lo_v, hi_v = with_visits.to_min_max_days_home()
        assert hi_n == lo_n  # no visits → degenerate band
        assert all(hi >= lo for lo, hi in zip(lo_v, hi_v))
        assert any(hi > lo for lo, hi in zip(lo_v, hi_v))  # band actually widened
        assert lo_v == lo_n  # the floor is unchanged by visit probability


class TestPresenceFractionAndExpectation:
    """annual_presence_fraction / expected_days_home invariants."""

    @pytest.mark.parametrize("name", list(PRESENCE_CALENDAR_PRESETS))
    def test_fraction_in_unit_interval(self, name):
        frac = PRESENCE_CALENDAR_PRESETS[name].annual_presence_fraction()
        assert 0.0 <= frac <= 1.0

    def test_full_time_fraction_is_one(self):
        cal = _uniform(MonthPresencePattern(full_weeks=5, weekends=True))
        assert cal.annual_presence_fraction() == pytest.approx(1.0)

    def test_never_home_fraction_is_zero(self):
        cal = _uniform(MonthPresencePattern(full_weeks=0, extra_weekdays=0, weekends=False))
        assert cal.annual_presence_fraction() == pytest.approx(0.0)

    def test_expected_is_midpoint_of_band(self):
        """expected_days_home[m] == (min[m] + max[m]) / 2."""
        cal = PRESENCE_CALENDAR_PRESETS["summer_vacation"]
        min_days, max_days = cal.to_min_max_days_home()
        expected = cal.expected_days_home()
        for m in range(12):
            assert expected[m] == pytest.approx((min_days[m] + max_days[m]) / 2.0)

    def test_fraction_matches_expected_over_365(self):
        """annual_presence_fraction == sum(expected_days_home) / 365."""
        cal = PRESENCE_CALENDAR_PRESETS["primary_residence"]
        frac = cal.annual_presence_fraction()
        assert frac == pytest.approx(sum(cal.expected_days_home()) / 365.0)

    def test_default_calendar_is_about_23_days(self):
        """The packaged default is ~23 occupied days/month (primary residence)."""
        min_days, _ = DEFAULT_PRESENCE_CALENDAR.to_min_max_days_home()
        assert all(lo == 23 for lo in min_days)


class TestJsonRoundTrip:
    """to_dict / from_dict serialization."""

    @pytest.mark.parametrize("name", list(PRESENCE_CALENDAR_PRESETS))
    def test_calendar_round_trip(self, name):
        cal = PRESENCE_CALENDAR_PRESETS[name]
        restored = PresenceCalendar.from_dict(cal.to_dict())
        assert restored == cal
        assert restored.to_min_max_days_home() == cal.to_min_max_days_home()

    def test_pattern_round_trip(self):
        p = MonthPresencePattern(
            full_weeks=2, extra_weekdays=3, weekends=False, visit_probability=0.25
        )
        assert MonthPresencePattern.from_dict(p.to_dict()) == p

    def test_from_dict_applies_defaults_for_missing_keys(self):
        p = MonthPresencePattern.from_dict({"full_weeks": 1})
        assert p.full_weeks == 1
        assert p.weekends is True
        assert p.extra_weekdays == 0
        assert p.visit_probability == 0.0

    def test_from_dict_rejects_wrong_month_count(self):
        with pytest.raises(ValueError):
            PresenceCalendar.from_dict({"months": [{}] * 11})


class TestPresenceCalendarStructure:
    """Construction-time invariants."""

    def test_requires_twelve_months(self):
        with pytest.raises(ValueError):
            PresenceCalendar(months=tuple(MonthPresencePattern() for _ in range(11)))

    def test_house_type_presets_are_well_formed(self):
        assert "apartment_standard" in HOUSE_TYPE_PRESETS
        for key, preset in HOUSE_TYPE_PRESETS.items():
            assert preset.label_it
            assert preset.floor_area_m2 > 0
            assert preset.baseline_annual_kwh > 0
