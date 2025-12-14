"""
Load scenario blueprint for configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .base import LoadProfile, BL_TABLE
from .variable import VariableLoadProfile
from .home_away import HomeAwayLoadProfile
from .arera import AreraLoadProfile
from ...calendar_utils import MONTH_LENGTHS

@dataclass
class LoadScenarioBlueprint:
    """
    Configuration blueprint for building complex load profile compositions.

    Provides a declarative way to assemble load profiles with optional
    home/away patterns and stochastic variations. Acts as a factory pattern
    for creating composable LoadProfile hierarchies.

    This class simplifies construction of common load profile scenarios:
    - Single profile (optionally with variation)
    - Home/away dual-profile (with optional variations on each)
    - Vacation home patterns (seasonal occupancy with stochastic variation)

    Build Logic:
    - If both factories provided → HomeAwayLoadProfile(home, away)
    - If only home_factory → home profile (possibly with variation)
    - If only away_factory → away profile (possibly with variation)
    - If neither factory → ValueError

    Each factory is wrapped with VariableLoadProfile if percentiles provided.

    Attributes:
        home_profile_factory: Factory function creating home/occupied profile.
        away_profile_factory: Factory function creating away/standby profile.
        min_days_home: Minimum home days per month (12 values) for home/away.
        max_days_home: Maximum home days per month (12 values) for home/away.
        home_variation_percentiles: Optional (p05, p95) for home profile variation.
        away_variation_percentiles: Optional (p05, p95) for away profile variation.

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import (
            LoadScenarioBlueprint,
            AreraLoadProfile,
            MonthlyAverageLoadProfile,
            make_flat_monthly_load_profiles
        )

        # Scenario 1: Single profile with variation
        blueprint_simple = LoadScenarioBlueprint(
            home_profile_factory=lambda: AreraLoadProfile(),
            home_variation_percentiles=(-0.10, 0.10)  # ±10% variation
        )
        load_simple = blueprint_simple.build_load_profile()
        # Returns: VariableLoadProfile(AreraLoadProfile())

        # Scenario 2: Home/Away without variation
        blueprint_homeaway = LoadScenarioBlueprint(
            home_profile_factory=lambda: AreraLoadProfile(),
            away_profile_factory=lambda: MonthlyAverageLoadProfile(
                make_flat_monthly_load_profiles(80.0)
            ),
            min_days_home=[20]*12,  # Home 20-25 days/month
            max_days_home=[25]*12
        )
        load_homeaway = blueprint_homeaway.build_load_profile()
        # Returns: HomeAwayLoadProfile(home=ARERA, away=80W constant)

        # Scenario 3: Vacation home with variations
        blueprint_vacation = LoadScenarioBlueprint(
            home_profile_factory=lambda: AreraLoadProfile(),
            away_profile_factory=lambda: MonthlyAverageLoadProfile(
                make_flat_monthly_load_profiles(70.0)
            ),
            min_days_home=[5, 5, 5, 5, 5, 20, 25, 25, 5, 5, 5, 5],  # Summer use
            max_days_home=[10, 10, 10, 10, 15, 30, 31, 31, 15, 10, 10, 10],
            home_variation_percentiles=(-0.15, 0.15),  # ±15% when home
            away_variation_percentiles=(-0.05, 0.05)   # ±5% when away (minimal)
        )
        load_vacation = blueprint_vacation.build_load_profile()
        # Returns: HomeAwayLoadProfile(
        #     home=VariableLoadProfile(ARERA, ±15%),
        #     away=VariableLoadProfile(70W constant, ±5%)
        # )
        ```

    Notes:
        - Factory pattern: Defers profile creation until build_load_profile() called
        - Variation decoration automatic: Percentiles trigger VariableLoadProfile wrapping
        - Validation: Checks min/max_days_home present for home/away scenarios
        - Flexible: Supports any LoadProfile subclass via factories
    """

    home_profile_factory: Callable[[], LoadProfile] | None = None
    """Factory creating home/occupied LoadProfile. Called by build_load_profile()."""

    away_profile_factory: Callable[[], LoadProfile] | None = None
    """Factory creating away/standby LoadProfile. Called by build_load_profile()."""

    min_days_home: List[int] | None = None
    """Minimum home days per month (12 ints). Required if both factories provided."""

    max_days_home: List[int] | None = None
    """Maximum home days per month (12 ints). Required if both factories provided."""

    home_variation_percentiles: Tuple[float, float] | None = None
    """Optional (p05_delta, p95_delta) for VariableLoadProfile wrapping home profile."""

    away_variation_percentiles: Tuple[float, float] | None = None
    """Optional (p05_delta, p95_delta) for VariableLoadProfile wrapping away profile."""

    @staticmethod
    def _apply_variation(
        profile: LoadProfile,
        percentiles: Tuple[float, float] | None,
    ) -> LoadProfile:
        """
        Optionally wrap profile with VariableLoadProfile for stochastic variation.

        If percentiles provided, wraps the base profile with VariableLoadProfile
        using the specified variation bounds. If percentiles is None, returns
        profile unchanged.

        Args:
            profile: Base LoadProfile to optionally decorate.
                Can be any LoadProfile subclass.
            percentiles: Optional (p05_delta, p95_delta) tuple for variation bounds.
                p05_delta: 5th percentile (must be negative, e.g., -0.10).
                p95_delta: 95th percentile (must be positive, e.g., +0.10).
                If None, no wrapping applied (deterministic profile returned).

        Returns:
            LoadProfile: Either the original profile (if percentiles=None)
                or VariableLoadProfile(profile, p05, p95) decorator.

        Example:
            ```python
            from sim_stochastic_pv.simulation.load_profiles import (
                LoadScenarioBlueprint,
                AreraLoadProfile
            )

            base = AreraLoadProfile()

            # No variation
            profile1 = LoadScenarioBlueprint._apply_variation(base, None)
            # Returns: base (unchanged)

            # With variation
            profile2 = LoadScenarioBlueprint._apply_variation(base, (-0.08, 0.08))
            # Returns: VariableLoadProfile(base, -0.08, 0.08)
            ```

        Notes:
            - Internal helper method (static, no self needed)
            - Decorator pattern: Wraps profile without modifying it
            - Validation performed by VariableLoadProfile constructor
        """
        if percentiles is None:
            return profile
        p05, p95 = percentiles
        return VariableLoadProfile(
            base_profile=profile,
            p05_delta=p05,
            p95_delta=p95,
        )

    def build_load_profile(self) -> LoadProfile:
        """
        Construct LoadProfile from blueprint configuration.

        Instantiates and assembles the configured load profile hierarchy
        by calling factories and applying decorators. The returned profile
        is ready for use in energy simulations.

        Build Algorithm:
        1. If both factories provided:
           - Call home_factory() and away_factory()
           - Apply variations if percentiles specified
           - Validate min/max_days_home present
           - Return HomeAwayLoadProfile(home, away, min_days, max_days)
        2. Else if home_factory only:
           - Call home_factory()
           - Apply variation if home_variation_percentiles specified
           - Return (possibly wrapped) home profile
        3. Else if away_factory only:
           - Call away_factory()
           - Apply variation if away_variation_percentiles specified
           - Return (possibly wrapped) away profile
        4. Else: Raise ValueError (no factory provided)

        Returns:
            LoadProfile: Fully constructed load profile ready for simulation.
                Type depends on configuration:
                - HomeAwayLoadProfile: If both factories provided
                - VariableLoadProfile: If single factory + percentiles
                - Concrete LoadProfile subclass: If single factory, no variation

        Raises:
            ValueError: If no factory provided (both None).
            ValueError: If both factories provided but min/max_days_home missing.
            Any exceptions from factory functions (e.g., invalid arguments).

        Example:
            ```python
            from sim_stochastic_pv.simulation.load_profiles import (
                LoadScenarioBlueprint,
                AreraLoadProfile,
                MonthlyAverageLoadProfile,
                make_flat_monthly_load_profiles
            )

            # Build scenario 1: Simple ARERA with variation
            blueprint1 = LoadScenarioBlueprint(
                home_profile_factory=lambda: AreraLoadProfile(),
                home_variation_percentiles=(-0.10, 0.10)
            )
            profile1 = blueprint1.build_load_profile()
            # Type: VariableLoadProfile wrapping AreraLoadProfile

            # Build scenario 2: Home/away vacation home
            blueprint2 = LoadScenarioBlueprint(
                home_profile_factory=lambda: AreraLoadProfile(),
                away_profile_factory=lambda: MonthlyAverageLoadProfile(
                    make_flat_monthly_load_profiles(75.0)
                ),
                min_days_home=[5]*6 + [25]*2 + [5]*4,  # Summer occupancy
                max_days_home=[10]*6 + [31]*2 + [10]*4,
                home_variation_percentiles=(-0.12, 0.12)
            )
            profile2 = blueprint2.build_load_profile()
            # Type: HomeAwayLoadProfile with VariableLoadProfile-wrapped home

            # Use in simulation
            import numpy as np
            rng = np.random.default_rng(42)
            profile2.reset_for_run(rng=rng, n_years=20)
            load = profile2.get_hourly_load_kw(0, 6, 15, 12, 0)  # July 16, noon
            ```

        Notes:
            - Factories called during build (lazy instantiation)
            - Decorators applied automatically based on percentiles
            - Validation ensures required parameters present
            - Each call creates fresh profile instances (factories re-executed)
            - Use in API/config systems for declarative load profile definition
        """
        if self.home_profile_factory and self.away_profile_factory:
            if self.min_days_home is None or self.max_days_home is None:
                raise ValueError("min_days_home and max_days_home are required for home/away scenarios")
            home_profile = self._apply_variation(
                self.home_profile_factory(),
                self.home_variation_percentiles,
            )
            away_profile = self._apply_variation(
                self.away_profile_factory(),
                self.away_variation_percentiles,
            )
            return HomeAwayLoadProfile(
                home_profile=home_profile,
                away_profile=away_profile,
                min_days_home=self.min_days_home,
                max_days_home=self.max_days_home,
            )

        if self.home_profile_factory:
            return self._apply_variation(
                self.home_profile_factory(),
                self.home_variation_percentiles,
            )

        if self.away_profile_factory:
            return self._apply_variation(
                self.away_profile_factory(),
                self.away_variation_percentiles,
            )

        raise ValueError("At least one profile factory must be provided")
