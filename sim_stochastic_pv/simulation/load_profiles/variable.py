"""
Variable load profile with daily stochastic multipliers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .base import LoadProfile, BL_TABLE
from ...calendar_utils import MONTH_LENGTHS

# Z-score for 95% confidence interval (±1.645σ covers 90% central mass)
Z_VALUE_95 = 1.6448536269514722


class VariableLoadProfile(LoadProfile):
    """
    Stochastic decorator applying daily random multipliers to base load profile.

    Wraps any LoadProfile and adds day-to-day consumption variability by
    applying random multipliers sampled from a truncated normal distribution.
    This models realistic household consumption uncertainty: some days use more
    electricity (guests, cooking, heating), others less (away, mild weather).

    Each calendar day gets a unique random multiplier sampled once and cached.
    All 24 hours in that day use the same multiplier, preserving the base
    profile's diurnal shape while scaling the overall daily consumption.

    Multiplier Distribution:
        multiplier ~ 1 + TruncatedNormal(0, σ, p05_delta, p95_delta)
        where σ = max(|p05|, |p95|) / 1.6448536 (95th percentile z-score)

    Use cases:
    - Add realism to deterministic profiles (ARERA, monthly averages)
    - Model consumption uncertainty in Monte Carlo simulations
    - Represent unpredictable household behavior (occupancy, appliance use)
    - Create realistic load distributions for risk analysis

    Attributes:
        base_profile: Wrapped LoadProfile providing reference consumption.
        p05_delta: 5th percentile variation bound (negative, e.g., -0.10 = -10%).
        p95_delta: 95th percentile variation bound (positive, e.g., +0.10 = +10%).

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.load_profiles import (
            VariableLoadProfile,
            MonthlyAverageLoadProfile,
            make_flat_monthly_load_profiles
        )

        # Start with deterministic profile
        base = MonthlyAverageLoadProfile(
            make_flat_monthly_load_profiles(200.0)  # 200W constant
        )

        # Add ±10% daily variation
        variable_model = VariableLoadProfile(
            base_profile=base,
            p05_delta=-0.10,  # 5% of days: 10% below (180W)
            p95_delta=0.10    # 5% of days: 10% above (220W)
        )

        # Reset for Monte Carlo path
        rng = np.random.default_rng(seed=42)
        variable_model.reset_for_run(rng=rng, n_years=20)

        # Day 1: multiplier might be 0.95 → all hours: 200W × 0.95 = 190W
        load_day1 = variable_model.get_hourly_load_kw(0, 0, 0, 12, 0)  # 0.19 kW

        # Day 2: different multiplier, e.g., 1.07 → all hours: 200W × 1.07 = 214W
        load_day2 = variable_model.get_hourly_load_kw(0, 0, 1, 12, 1)  # 0.214 kW

        # Conservative variation (±5%)
        model_conservative = VariableLoadProfile(base, -0.05, 0.05)

        # High variation (±20%, e.g., vacation home uncertainty)
        model_high_var = VariableLoadProfile(base, -0.20, 0.20)
        ```

    Notes:
        - Stochastic: Each day gets random multiplier from truncated normal
        - Day-level granularity: Same multiplier for all 24 hours in a day
        - Preserves shape: Diurnal pattern maintained, only scaled
        - Decorator pattern: Can wrap any LoadProfile subclass
        - Cascadable: Can wrap other stochastic profiles (e.g., HomeAwayLoadProfile)
        - Cache efficient: One sample per day, reused for all hours
        - Non-negative: Multipliers clipped to [0, ∞) to prevent negative loads
    """

    def __init__(
        self,
        base_profile: LoadProfile,
        p05_delta: float = -0.1,
        p95_delta: float = 0.1,
    ) -> None:
        """
        Initialize variable load profile with stochastic daily multipliers.

        Args:
            base_profile: Base LoadProfile to wrap and add variation to.
                Can be any LoadProfile subclass (deterministic or stochastic).
                The wrapped profile's consumption is multiplied by random factors.
            p05_delta: 5th percentile variation (must be negative).
                Lower bound for daily multipliers. Example: -0.10 means 5% of
                days have ≤10% reduction from base consumption.
                Typical values: -0.05 to -0.20 (conservative to high uncertainty).
            p95_delta: 95th percentile variation (must be positive).
                Upper bound for daily multipliers. Example: +0.10 means 5% of
                days have ≥10% increase from base consumption.
                Typical values: +0.05 to +0.20 (conservative to high uncertainty).

        Raises:
            ValueError: If p05_delta ≥ 0 (must be negative).
            ValueError: If p95_delta ≤ 0 (must be positive).

        Example:
            ```python
            from sim_stochastic_pv.simulation.load_profiles import (
                VariableLoadProfile,
                AreraLoadProfile
            )

            # Conservative residential (±8% variation)
            base = AreraLoadProfile()
            model = VariableLoadProfile(
                base_profile=base,
                p05_delta=-0.08,
                p95_delta=0.08
            )

            # High uncertainty commercial (±15% variation)
            model_commercial = VariableLoadProfile(
                base_profile=base,
                p05_delta=-0.15,
                p95_delta=0.15
            )

            # Asymmetric variation (more likely to exceed than undershoot)
            model_asymmetric = VariableLoadProfile(
                base_profile=base,
                p05_delta=-0.05,  # Rarely much less
                p95_delta=0.15    # Often significantly more
            )
            ```

        Notes:
            - Standard deviation σ derived from percentiles: σ = max(|p05|, |p95|) / 1.645
            - Symmetric bounds (|p05| = p95) produce symmetric multiplier distribution
            - Asymmetric bounds allow modeling skewed consumption patterns
            - Zero sigma (p05=p95=0) → no variation (passes through base profile)
            - Multiplier mean = 1.0 (on average, matches base profile)
        """
        if p05_delta >= 0:
            raise ValueError("p05_delta must be negative")
        if p95_delta <= 0:
            raise ValueError("p95_delta must be positive")
        self.base_profile = base_profile
        self.p05_delta = float(p05_delta)
        self.p95_delta = float(p95_delta)
        max_delta = max(abs(self.p05_delta), abs(self.p95_delta))
        self._sigma = max_delta / Z_VALUE_95 if max_delta > 0 else 0.0
        self._rng: np.random.Generator | None = None
        self._fallback_rng = np.random.default_rng()
        self._daily_multipliers: Dict[Tuple[int, int, int], float] = {}

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Reset wrapped profile and clear daily multipliers for new Monte Carlo path.

        Clears cached multipliers from previous path and propagates reset to
        the wrapped base profile. This ensures each Monte Carlo path gets
        independent random daily variations.

        Args:
            rng: Random number generator for stochastic multiplier sampling.
                Stored for use during get_hourly_load_kw() calls.
            n_years: Number of simulation years (passed to base_profile).
                Unused by this class but forwarded to wrapped profile.

        Notes:
            - Clears _daily_multipliers cache (forces regeneration)
            - Stores rng for lazy multiplier sampling during simulation
            - Resets base_profile with same rng for consistency
            - Called once per Monte Carlo path before year loop
        """
        self.base_profile.reset_for_run(rng=rng, n_years=n_years)
        self._daily_multipliers.clear()
        if rng is not None:
            self._rng = rng
        elif self._rng is None:
            self._rng = self._fallback_rng

    def _rng_for_variation(self) -> np.random.Generator:
        """
        Get random number generator for multiplier sampling.

        Returns the stored RNG or fallback if not initialized. Internal helper
        ensuring valid RNG is always available.

        Returns:
            np.random.Generator: Active random number generator instance.

        Notes:
            - Internal helper method (not part of public API)
            - Automatically initializes fallback if _rng not set
            - Used by _sample_multiplier() for random variation generation
        """
        if self._rng is None:
            self._rng = self._fallback_rng
        return self._rng

    def _sample_multiplier(self) -> float:
        """
        Sample random daily multiplier from truncated normal distribution.

        Generates a single multiplier value representing the day's consumption
        scaling factor. Sampled from normal distribution centered at 0 with
        standard deviation derived from percentile bounds, then clipped and
        shifted to produce multipliers around 1.0.

        Algorithm:
        1. Sample variation ~ Normal(0, σ)
        2. Clip to [p05_delta, p95_delta]
        3. Compute multiplier = 1.0 + variation
        4. Clip to [0, ∞) to prevent negative loads

        Returns:
            float: Daily consumption multiplier (non-negative).
                Typical range: 0.8-1.2 for ±10% variation bounds.
                Mean ≈ 1.0 (on average matches base profile).
                Examples:
                - 0.92: Day with 8% reduction
                - 1.0: Day matching base profile
                - 1.15: Day with 15% increase

        Notes:
            - Returns 1.0 if σ = 0 (no variation configured)
            - Truncated normal ensures realistic bounds (no extreme outliers)
            - Non-negative guarantee: max(0.0, multiplier) prevents negative loads
            - Each call samples independently (for different days)
        """
        if self._sigma <= 0.0:
            return 1.0
        rng = self._rng_for_variation()
        variation = rng.normal(loc=0.0, scale=self._sigma)
        variation = np.clip(variation, self.p05_delta, self.p95_delta)
        multiplier = 1.0 + variation
        return max(0.0, multiplier)

    def _get_multiplier(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
    ) -> float:
        """
        Get or lazily sample daily multiplier for specified calendar day.

        Checks cache for existing multiplier, otherwise samples new random
        value and caches it. This ensures all hours in the same day use the
        same multiplier while different days get independent random values.

        Args:
            year_index: Simulation year (0-based) for cache key.
            month_in_year: Month index (0-11) for cache key.
            day_in_month: Day index (0-based) for cache key.

        Returns:
            float: Daily consumption multiplier for this specific day.
                Cached value if previously sampled, new random value otherwise.

        Notes:
            - Cached by (year, month, day) key for efficiency
            - Lazy sampling: Only computed when first queried
            - Persistent within run: Same day always returns same multiplier
            - Independent across days: Different days get different multipliers
            - Cleared by reset_for_run(): Fresh multipliers each MC path
        """
        key = (year_index, month_in_year, day_in_month)
        multiplier = self._daily_multipliers.get(key)
        if multiplier is not None:
            return multiplier
        multiplier = self._sample_multiplier()
        self._daily_multipliers[key] = multiplier
        return multiplier

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get stochastic hourly load with daily random variation.

        Queries the base profile for the reference consumption, then applies
        the day's random multiplier to introduce realistic day-to-day variability.
        All hours in the same day use the same multiplier.

        Args:
            year_index: Simulation year (0-based).
            month_in_year: Month (0-11).
            day_in_month: Day (0-based) for multiplier lookup.
            hour_in_day: Hour (0-23) passed to base profile.
            weekday: Weekday (0-6) passed to base profile.

        Returns:
            float: Stochastic electricity consumption in kW.
                Formula: base_load × daily_multiplier
                where daily_multiplier is constant for all hours in the day.

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.load_profiles import (
                VariableLoadProfile,
                MonthlyAverageLoadProfile,
                make_flat_monthly_load_profiles
            )

            # Base: 200W constant
            base = MonthlyAverageLoadProfile(make_flat_monthly_load_profiles(200.0))

            # Add ±15% variation
            model = VariableLoadProfile(base, -0.15, 0.15)
            model.reset_for_run(rng=np.random.default_rng(42), n_years=1)

            # Day 0: multiplier might be 0.92
            load_d0_h0 = model.get_hourly_load_kw(0, 0, 0, 0, 0)   # 200W × 0.92 = 184W
            load_d0_h12 = model.get_hourly_load_kw(0, 0, 0, 12, 0)  # 200W × 0.92 = 184W (same)

            # Day 1: different multiplier, e.g., 1.08
            load_d1_h0 = model.get_hourly_load_kw(0, 0, 1, 0, 1)   # 200W × 1.08 = 216W
            load_d1_h12 = model.get_hourly_load_kw(0, 0, 1, 12, 1)  # 200W × 1.08 = 216W (same)
            ```

        Notes:
            - Stochastic: Different days return different scaled values
            - Day-consistent: All hours in same day use same multiplier
            - Multiplier lazy-sampled: First query triggers sampling and caching
            - Shape-preserving: Diurnal pattern maintained, only amplitude varies
            - Two-stage lookup: base_profile.get_hourly_load_kw() then multiply
        """
        base_kw = self.base_profile.get_hourly_load_kw(
            year_index=year_index,
            month_in_year=month_in_year,
            day_in_month=day_in_month,
            hour_in_day=hour_in_day,
            weekday=weekday,
        )
        multiplier = self._get_multiplier(year_index, month_in_year, day_in_month)
        return base_kw * multiplier


