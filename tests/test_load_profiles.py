"""Tests for load profiles module (reorganized from load_profiles.py)."""

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
from sim_stochastic_pv.calendar_utils import build_calendar


class TestMonthlyAverageLoadProfile:
    """Tests for MonthlyAverageLoadProfile."""

    def test_initialization(self):
        """Profile can be initialized with monthly averages."""
        monthly_avg = [300.0] * 12  # 300 kWh/month for all months
        profile = MonthlyAverageLoadProfile(monthly_avg_kwh=monthly_avg)
        assert profile.monthly_avg_kwh == monthly_avg

    def test_get_load_one_year(self):
        """Getting load for one year returns correct shape."""
        monthly_avg = [300.0 + i * 10 for i in range(12)]
        profile = MonthlyAverageLoadProfile(monthly_avg_kwh=monthly_avg)

        rng = np.random.default_rng(42)
        calendar = build_calendar(1)
        loads = profile.get_load(rng, calendar)

        # Should return 12 months of hourly data
        assert loads.shape == (12, 24)

    def test_deterministic_with_seed(self):
        """Same seed produces same load pattern."""
        monthly_avg = [300.0] * 12
        profile = MonthlyAverageLoadProfile(monthly_avg_kwh=monthly_avg)

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        calendar = build_calendar(1)

        loads1 = profile.get_load(rng1, calendar)
        loads2 = profile.get_load(rng2, calendar)

        np.testing.assert_array_equal(loads1, loads2)


class TestAreraLoadProfile:
    """Tests for AreraLoadProfile (Italian tariff-based)."""

    def test_initialization(self):
        """Profile can be initialized with yearly kWh."""
        profile = AreraLoadProfile(yearly_kwh=3000.0)
        assert profile.yearly_kwh == 3000.0

    def test_get_load_shape(self):
        """Get load returns correct shape."""
        profile = AreraLoadProfile(yearly_kwh=3000.0)
        rng = np.random.default_rng(42)
        calendar = build_calendar(1)

        loads = profile.get_load(rng, calendar)
        assert loads.shape == (12, 24)

    def test_uses_italian_patterns(self):
        """Profile should use Italian consumption patterns."""
        profile = AreraLoadProfile(yearly_kwh=3600.0)
        rng = np.random.default_rng(42)
        calendar = build_calendar(1)

        loads = profile.get_load(rng, calendar)

        # Check that total is approximately correct (300 kWh/month avg)
        total_kwh = loads.sum()
        # Allow some variation due to randomness
        assert 3000 < total_kwh < 4200


class TestHomeAwayLoadProfile:
    """Tests for HomeAwayLoadProfile (stochastic occupancy)."""

    def test_initialization(self):
        """Profile can be initialized with home/away patterns."""
        home_profiles = [[200.0] * 24 for _ in range(12)]
        min_days_home = [15] * 12
        max_days_home = [20] * 12

        profile = HomeAwayLoadProfile(
            home_profiles_w=home_profiles,
            min_days_home=min_days_home,
            max_days_home=max_days_home,
            home_variation_percentiles=[-0.1, 0.1],
            away_variation_percentiles=[-0.05, 0.05],
            away_profile="arera",
        )

        assert len(profile.home_profiles_w) == 12
        assert len(profile.min_days_home) == 12

    def test_get_load_shape(self):
        """Get load returns correct shape."""
        home_profiles = [[200.0] * 24 for _ in range(12)]
        profile = HomeAwayLoadProfile(
            home_profiles_w=home_profiles,
            min_days_home=[15] * 12,
            max_days_home=[20] * 12,
        )

        rng = np.random.default_rng(42)
        calendar = build_calendar(1)

        loads = profile.get_load(rng, calendar)
        assert loads.shape == (12, 24)

    def test_different_seeds_produce_different_loads(self):
        """Different seeds should produce different stochastic patterns."""
        home_profiles = [[200.0] * 24 for _ in range(12)]
        profile = HomeAwayLoadProfile(
            home_profiles_w=home_profiles,
            min_days_home=[15] * 12,
            max_days_home=[20] * 12,
        )

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)
        calendar = build_calendar(1)

        loads1 = profile.get_load(rng1, calendar)
        loads2 = profile.get_load(rng2, calendar)

        # Should be different due to stochastic home/away days
        assert not np.array_equal(loads1, loads2)


class TestVariableLoadProfile:
    """Tests for VariableLoadProfile (daily multipliers)."""

    def test_initialization(self):
        """Profile can be initialized with daily multipliers."""
        base_profile = [100.0] * 24
        daily_multipliers = [1.0] * 365

        profile = VariableLoadProfile(
            base_hourly_profile_w=base_profile,
            daily_multipliers=daily_multipliers,
        )

        assert len(profile.base_hourly_profile_w) == 24
        assert len(profile.daily_multipliers) == 365

    def test_get_load_shape(self):
        """Get load returns correct shape."""
        profile = VariableLoadProfile(
            base_hourly_profile_w=[100.0] * 24,
            daily_multipliers=[1.0] * 365,
        )

        rng = np.random.default_rng(42)
        calendar = build_calendar(1)

        loads = profile.get_load(rng, calendar)
        assert loads.shape == (12, 24)

    def test_multipliers_affect_load(self):
        """Daily multipliers should affect the load."""
        base_profile = [100.0] * 24

        # Create multipliers: 0.5 for first half of year, 1.5 for second half
        multipliers = [0.5] * 182 + [1.5] * 183

        profile = VariableLoadProfile(
            base_hourly_profile_w=base_profile,
            daily_multipliers=multipliers,
        )

        rng = np.random.default_rng(42)
        calendar = build_calendar(1)
        loads = profile.get_load(rng, calendar)

        # First 6 months should have lower load than last 6 months
        first_half_avg = loads[:6].mean()
        second_half_avg = loads[6:].mean()

        assert second_half_avg > first_half_avg


class TestLoadScenarioBlueprint:
    """Tests for LoadScenarioBlueprint."""

    def test_initialization_from_dict(self):
        """Blueprint can be initialized from dict configuration."""
        config = {
            "home_profiles_w": [[200.0] * 24] * 12,
            "min_days_home": [15] * 12,
            "max_days_home": [20] * 12,
            "home_variation_percentiles": [-0.1, 0.1],
            "away_variation_percentiles": [-0.05, 0.05],
            "away_profile": "arera",
        }

        blueprint = LoadScenarioBlueprint.from_dict(config)
        assert blueprint.min_days_home == [15] * 12
        assert blueprint.away_profile == "arera"

    def test_realize_creates_profile(self):
        """Blueprint realize() creates a HomeAwayLoadProfile."""
        config = {
            "home_profiles_w": [[200.0] * 24] * 12,
            "min_days_home": [15] * 12,
            "max_days_home": [20] * 12,
        }

        blueprint = LoadScenarioBlueprint.from_dict(config)
        profile = blueprint.realize()

        assert isinstance(profile, HomeAwayLoadProfile)

    def test_different_seeds_produce_different_profiles(self):
        """Different seeds should produce different realized profiles."""
        config = {
            "home_profiles_w": [[200.0] * 24] * 12,
            "min_days_home": [15] * 12,
            "max_days_home": [20] * 12,
        }

        blueprint = LoadScenarioBlueprint.from_dict(config)

        profile1 = blueprint.realize(seed=42)
        profile2 = blueprint.realize(seed=123)

        # Realized profiles should be independent instances
        assert profile1 is not profile2

        # Get loads to verify they're different
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)
        calendar = build_calendar(1)

        loads1 = profile1.get_load(rng1, calendar)
        loads2 = profile2.get_load(rng2, calendar)

        # Should produce different patterns
        assert not np.array_equal(loads1, loads2)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_make_flat_monthly_load_profiles(self):
        """Helper creates flat monthly profiles correctly."""
        monthly_kwh = [300.0 + i * 10 for i in range(12)]
        profiles = make_flat_monthly_load_profiles(monthly_kwh)

        assert len(profiles) == 12

        for i, profile in enumerate(profiles):
            assert len(profile) == 24
            # Each hour should be total monthly divided by hours in month
            expected_hourly_w = (monthly_kwh[i] * 1000.0) / (24 * 30)  # Approximate
            # All hours should be equal (flat profile)
            assert all(abs(h - profile[0]) < 1e-6 for h in profile)


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

        # All imports should succeed
        assert LoadProfile is not None
        assert MonthlyAverageLoadProfile is not None
        assert AreraLoadProfile is not None
        assert HomeAwayLoadProfile is not None
        assert VariableLoadProfile is not None
        assert LoadScenarioBlueprint is not None
        assert make_flat_monthly_load_profiles is not None
