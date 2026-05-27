"""Tests for load profiles module (updated to match current implementation)."""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.simulation.load_profiles import (
    LoadProfile,
    MonthlyAverageLoadProfile,
    AreraLoadProfile,
    HomeAwayLoadProfile,
    VariableLoadProfile,
    LoadScenarioBlueprint,
    make_flat_monthly_load_profiles,
)
from sim_stochastic_pv.calendar_utils import build_calendar, MONTH_LENGTHS


# ---------------------------------------------------------------------------
# MonthlyAverageLoadProfile
# ---------------------------------------------------------------------------

class TestMonthlyAverageLoadProfile:
    """Tests for MonthlyAverageLoadProfile."""

    def test_initialization_with_monthly_avg_kwh(self):
        """Profile can be initialized with the simple monthly_avg_kwh interface."""
        monthly_avg = [300.0] * 12
        profile = MonthlyAverageLoadProfile(monthly_avg_kwh=monthly_avg)
        assert profile.monthly_avg_kwh == monthly_avg

    def test_initialization_with_profiles_w(self):
        """Profile can be initialized with the detailed (12, 24) Watts array."""
        profiles_w = np.full((12, 24), 200.0)
        profile = MonthlyAverageLoadProfile(profiles_w)
        assert profile.monthly_profiles_kw.shape == (12, 24)

    def test_hourly_load_shape_via_get_hourly_load_kw(self):
        """get_hourly_load_kw returns a float for every valid combination."""
        profile = MonthlyAverageLoadProfile(monthly_avg_kwh=[300.0] * 12)

        # Iterate over a representative sample of time points
        for month in range(12):
            for hour in range(24):
                val = profile.get_hourly_load_kw(
                    year_index=0,
                    month_in_year=month,
                    day_in_month=0,
                    hour_in_day=hour,
                    weekday=0,
                )
                assert isinstance(val, float)
                assert val >= 0.0

    def test_monthly_avg_kwh_converts_to_correct_watts(self):
        """kWh→W conversion uses actual days in each month."""
        monthly_kwh = [300.0] * 12
        profile = MonthlyAverageLoadProfile(monthly_avg_kwh=monthly_kwh)

        for m in range(12):
            expected_w = monthly_kwh[m] * 1000.0 / (MONTH_LENGTHS[m] * 24)
            # Internal storage is in kW
            actual_kw = profile.monthly_profiles_kw[m, 0]
            assert abs(actual_kw - expected_w / 1000.0) < 1e-9

    def test_deterministic_same_seed(self):
        """Profile is deterministic: same inputs → same output."""
        profile = MonthlyAverageLoadProfile(monthly_avg_kwh=[300.0] * 12)
        v1 = profile.get_hourly_load_kw(0, 3, 10, 14, 2)
        v2 = profile.get_hourly_load_kw(0, 3, 10, 14, 2)
        assert v1 == v2

    def test_raises_on_wrong_shape(self):
        """ValueError raised when profiles_w has wrong shape."""
        with pytest.raises(ValueError):
            MonthlyAverageLoadProfile(np.full((12, 12), 100.0))

    def test_raises_when_neither_provided(self):
        """ValueError raised when neither argument is provided."""
        with pytest.raises(ValueError):
            MonthlyAverageLoadProfile()

    def test_raises_when_both_provided(self):
        """ValueError raised when both arguments are provided."""
        with pytest.raises(ValueError):
            MonthlyAverageLoadProfile(
                np.full((12, 24), 100.0),
                monthly_avg_kwh=[300.0] * 12,
            )


# ---------------------------------------------------------------------------
# AreraLoadProfile
# ---------------------------------------------------------------------------

class TestAreraLoadProfile:
    """Tests for AreraLoadProfile (Italian tariff-based)."""

    def test_default_initialization(self):
        """Profile initialises without arguments using the built-in BL_TABLE."""
        profile = AreraLoadProfile()
        assert profile.bl_table.shape == (12, 3)

    def test_get_hourly_load_kw_returns_float(self):
        """get_hourly_load_kw returns a non-negative float."""
        profile = AreraLoadProfile()
        val = profile.get_hourly_load_kw(
            year_index=0,
            month_in_year=0,
            day_in_month=0,
            hour_in_day=10,
            weekday=0,
        )
        assert isinstance(val, float)
        assert val > 0.0

    def test_f1_band_weekday_peak(self):
        """Monday 10 am is F1 (peak) — highest consumption."""
        profile = AreraLoadProfile()
        f1 = profile.get_hourly_load_kw(0, 0, 0, 10, 0)   # Mon 10 am
        f3 = profile.get_hourly_load_kw(0, 0, 6, 3, 6)     # Sun 3 am (F3 off-peak)
        # F1 load should differ from F3 load
        assert f1 != f3

    def test_seasonal_variation(self):
        """Different months return different load values (seasonal effect)."""
        profile = AreraLoadProfile()
        jan = profile.get_hourly_load_kw(0, 0, 0, 10, 0)
        aug = profile.get_hourly_load_kw(0, 7, 0, 10, 0)
        # BL_TABLE has distinct rows for Jan and Aug
        assert jan != aug

    def test_custom_bl_table(self):
        """Custom bl_table is respected."""
        custom = np.full((12, 3), 500.0)
        profile = AreraLoadProfile(bl_table=custom)
        val = profile.get_hourly_load_kw(0, 0, 0, 10, 0)
        assert abs(val - 0.5) < 1e-9   # 500 W = 0.5 kW


# ---------------------------------------------------------------------------
# HomeAwayLoadProfile
# ---------------------------------------------------------------------------

class TestHomeAwayLoadProfile:
    """Tests for HomeAwayLoadProfile (stochastic occupancy)."""

    def _make_home_away(self, min_days=15, max_days=20) -> HomeAwayLoadProfile:
        """Helper: build a minimal HomeAwayLoadProfile."""
        home = MonthlyAverageLoadProfile(np.full((12, 24), 300.0))
        away = AreraLoadProfile()
        return HomeAwayLoadProfile(
            home_profile=home,
            away_profile=away,
            min_days_home=[min_days] * 12,
            max_days_home=[max_days] * 12,
        )

    def test_initialization_stores_attributes(self):
        """Constructor stores min/max_days_home as arrays of length 12."""
        profile = self._make_home_away()
        assert len(profile.min_days_home) == 12
        assert len(profile.max_days_home) == 12

    def test_get_hourly_load_kw_returns_positive_float(self):
        """Returns a non-negative float after reset_for_run."""
        profile = self._make_home_away()
        rng = np.random.default_rng(42)
        profile.reset_for_run(rng=rng, n_years=1)

        val = profile.get_hourly_load_kw(0, 0, 0, 12, 0)
        assert isinstance(val, float)
        assert val >= 0.0

    def test_different_seeds_produce_different_loads(self):
        """Different seeds should produce different occupancy patterns."""
        profile1 = self._make_home_away()
        profile2 = self._make_home_away()

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(999)

        profile1.reset_for_run(rng=rng1, n_years=1)
        profile2.reset_for_run(rng=rng2, n_years=1)

        loads1 = [profile1.get_hourly_load_kw(0, 0, d, 12, d % 7) for d in range(28)]
        loads2 = [profile2.get_hourly_load_kw(0, 0, d, 12, d % 7) for d in range(28)]

        assert loads1 != loads2

    def test_home_days_count_within_bounds(self):
        """Number of home days per month is within [min_days, max_days]."""
        profile = self._make_home_away(min_days=5, max_days=10)
        rng = np.random.default_rng(7)
        profile.reset_for_run(rng=rng, n_years=1)

        schedule = profile._get_month_schedule(0, 0)  # January schedule
        home_count = schedule.sum()
        assert 5 <= home_count <= 10

    def test_raises_on_wrong_min_max_length(self):
        """ValueError when min_days_home/max_days_home not length 12."""
        home = MonthlyAverageLoadProfile(np.full((12, 24), 300.0))
        away = AreraLoadProfile()
        with pytest.raises(ValueError):
            HomeAwayLoadProfile(home, away, [15] * 6, [20] * 12)


# ---------------------------------------------------------------------------
# VariableLoadProfile
# ---------------------------------------------------------------------------

class TestVariableLoadProfile:
    """Tests for VariableLoadProfile (stochastic daily multipliers)."""

    def _base_profile(self) -> MonthlyAverageLoadProfile:
        return MonthlyAverageLoadProfile(np.full((12, 24), 200.0))

    def test_initialization(self):
        """Profile stores p05_delta and p95_delta."""
        profile = VariableLoadProfile(
            base_profile=self._base_profile(),
            p05_delta=-0.1,
            p95_delta=0.1,
        )
        assert profile.p05_delta == -0.1
        assert profile.p95_delta == 0.1

    def test_get_hourly_load_kw_returns_positive_float(self):
        """Returns a non-negative float after reset_for_run."""
        profile = VariableLoadProfile(self._base_profile(), -0.1, 0.1)
        rng = np.random.default_rng(42)
        profile.reset_for_run(rng=rng, n_years=1)

        val = profile.get_hourly_load_kw(0, 0, 0, 12, 0)
        assert isinstance(val, float)
        assert val >= 0.0

    def test_all_hours_same_day_use_same_multiplier(self):
        """All 24 hours in the same day share the same random multiplier."""
        profile = VariableLoadProfile(self._base_profile(), -0.2, 0.2)
        rng = np.random.default_rng(42)
        profile.reset_for_run(rng=rng, n_years=1)

        loads = [
            profile.get_hourly_load_kw(0, 0, 0, h, 0) for h in range(24)
        ]
        # All values must be equal (same day → same multiplier)
        assert all(abs(v - loads[0]) < 1e-12 for v in loads)

    def test_different_days_can_have_different_multipliers(self):
        """Different days usually have different multipliers."""
        profile = VariableLoadProfile(self._base_profile(), -0.3, 0.3)
        rng = np.random.default_rng(42)
        profile.reset_for_run(rng=rng, n_years=1)

        day0 = profile.get_hourly_load_kw(0, 0, 0, 12, 0)
        day1 = profile.get_hourly_load_kw(0, 0, 1, 12, 1)
        # With high variation and different random draws, they should differ
        assert day0 != day1

    def test_raises_on_non_negative_p05(self):
        """ValueError when p05_delta >= 0."""
        with pytest.raises(ValueError):
            VariableLoadProfile(self._base_profile(), p05_delta=0.0, p95_delta=0.1)

    def test_raises_on_non_positive_p95(self):
        """ValueError when p95_delta <= 0."""
        with pytest.raises(ValueError):
            VariableLoadProfile(self._base_profile(), p05_delta=-0.1, p95_delta=0.0)


# ---------------------------------------------------------------------------
# LoadScenarioBlueprint
# ---------------------------------------------------------------------------

class TestLoadScenarioBlueprint:
    """Tests for LoadScenarioBlueprint."""

    def test_build_single_profile(self):
        """Blueprint with one factory returns wrapped/unwrapped profile."""
        blueprint = LoadScenarioBlueprint(
            home_profile_factory=lambda: AreraLoadProfile(),
        )
        profile = blueprint.build_load_profile()
        assert isinstance(profile, AreraLoadProfile)

    def test_build_home_away_profile(self):
        """Blueprint with both factories returns HomeAwayLoadProfile."""
        profiles_w = np.full((12, 24), 200.0)
        blueprint = LoadScenarioBlueprint(
            home_profile_factory=lambda: MonthlyAverageLoadProfile(profiles_w),
            away_profile_factory=lambda: AreraLoadProfile(),
            min_days_home=[15] * 12,
            max_days_home=[20] * 12,
        )
        profile = blueprint.build_load_profile()
        assert isinstance(profile, HomeAwayLoadProfile)

    def test_build_with_variation_wraps_in_variable_profile(self):
        """Providing percentiles wraps the profile in VariableLoadProfile."""
        blueprint = LoadScenarioBlueprint(
            home_profile_factory=lambda: AreraLoadProfile(),
            home_variation_percentiles=(-0.1, 0.1),
        )
        profile = blueprint.build_load_profile()
        assert isinstance(profile, VariableLoadProfile)

    def test_build_raises_without_any_factory(self):
        """ValueError when no factory is provided."""
        with pytest.raises(ValueError):
            LoadScenarioBlueprint().build_load_profile()

    def test_build_raises_home_away_without_days(self):
        """ValueError when both factories given but min/max_days missing."""
        blueprint = LoadScenarioBlueprint(
            home_profile_factory=lambda: AreraLoadProfile(),
            away_profile_factory=lambda: AreraLoadProfile(),
            # min_days_home / max_days_home deliberately omitted
        )
        with pytest.raises(ValueError):
            blueprint.build_load_profile()


# ---------------------------------------------------------------------------
# make_flat_monthly_load_profiles helper
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_make_flat_returns_correct_shape(self):
        """Helper returns (12, 24) float array."""
        profiles = make_flat_monthly_load_profiles(base_load_w=200.0)
        assert profiles.shape == (12, 24)
        assert profiles.dtype == float

    def test_make_flat_all_values_equal(self):
        """All values in the returned array equal base_load_w."""
        profiles = make_flat_monthly_load_profiles(base_load_w=150.0)
        assert np.all(profiles == 150.0)

    def test_make_flat_default_value(self):
        """Default base_load_w is 140 W."""
        profiles = make_flat_monthly_load_profiles()
        assert np.all(profiles == 140.0)


# ---------------------------------------------------------------------------
# Backward-compatibility: all public names importable
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Tests for backward compatibility of imports."""

    def test_all_exports_available(self):
        """All expected classes should be importable from main module."""
        from sim_stochastic_pv.simulation.load_profiles import (
            LoadProfile,
            MonthlyAverageLoadProfile,
            AreraLoadProfile,
            HomeAwayLoadProfile,
            VariableLoadProfile,
            LoadScenarioBlueprint,
            make_flat_monthly_load_profiles,
        )

        assert LoadProfile is not None
        assert MonthlyAverageLoadProfile is not None
        assert AreraLoadProfile is not None
        assert HomeAwayLoadProfile is not None
        assert VariableLoadProfile is not None
        assert LoadScenarioBlueprint is not None
        assert make_flat_monthly_load_profiles is not None
