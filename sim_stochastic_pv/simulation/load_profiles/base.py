"""
Base class for electricity consumption profile generators.
"""

from __future__ import annotations

import numpy as np

from ...calendar_utils import MONTH_LENGTHS

# Rows represent months (0=January..11=December)
# Columns represent the base load in Watt for ARERA tariff bands [F1, F2, F3].
# The table is used to build deterministic load profiles for the "away" periods.
BL_TABLE: np.ndarray = np.array([
    [110.6719368,  83.79888268, 157.0512821 ],  # Gen
    [ 68.18181818, 73.17073171,  72.91666667],  # Feb
    [ 99.56709957,113.5135135,   94.51219512],  # Mar
    [144.6280992, 120.6896552,  138.1578947 ],  # Apr
    [ 90.90909091, 89.47368421, 102.5641026 ],  # Mag
    [116.8831169, 130.1775148,  134.375     ],  # Giu
    [130.4347826, 145.2513966,  125.        ],  # Lug
    [251.0822511, 286.4864865,  280.4878049],  # Ago
    [ 95.04132231, 97.70114943,  92.10526316],  # Set
    [169.9604743, 111.7318436, 118.5897436 ],  # Ott
    [131.8181818,  83.33333333, 168.75     ],  # Nov
    [185.770751,  229.0502793,  211.5384615],  # Dic
])


class LoadProfile:
    """
    Abstract base class for hourly electricity consumption models.

    Defines the interface for load profile implementations used in energy
    system simulations. Load profiles provide hourly electricity consumption
    (in kW) as a function of time, enabling realistic modeling of household
    or building energy demand patterns.

    The interface supports both deterministic and stochastic load profiles:
    - Deterministic: Fixed consumption patterns (e.g., monthly averages)
    - Stochastic: Random daily variations and occupancy patterns (Monte Carlo)

    Subclasses must implement:
    - get_hourly_load_kw(): Return consumption for specific hour
    - reset_for_run(): Optional initialization before each Monte Carlo path

    Example subclasses:
    - MonthlyAverageLoadProfile: Fixed 24h pattern per month
    - AreraLoadProfile: Italian ARERA tariff-based consumption
    - HomeAwayLoadProfile: Stochastic presence/absence patterns
    - VariableLoadProfile: Daily stochastic multipliers

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import MonthlyAverageLoadProfile
        import numpy as np

        # Create simple constant load profile
        monthly_profiles_w = np.full((12, 24), 150.0)  # 150W constant
        load_model = MonthlyAverageLoadProfile(monthly_profiles_w)

        # Get consumption for first hour of first year
        load_kw = load_model.get_hourly_load_kw(
            year_index=0,
            month_in_year=0,
            day_in_month=0,
            hour_in_day=0,
            weekday=0
        )
        print(f"Load: {load_kw:.3f} kW")  # 0.150 kW

        # Reset for Monte Carlo path
        rng = np.random.default_rng(seed=42)
        load_model.reset_for_run(rng=rng, n_years=20)
        ```

    Notes:
        - All time indices are 0-based
        - year_index: 0 = first year, 1 = second year, etc.
        - month_in_year: 0 = Jan, 11 = Dec
        - day_in_month: 0 = first day of month
        - hour_in_day: 0 = midnight-1am, 23 = 11pm-midnight
        - weekday: 0 = Monday, 6 = Sunday
        - Consumption returned in kW (power, not energy)
    """

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Prepare load profile for new simulation run (optional hook).

        Called before each Monte Carlo simulation path to initialize or
        reset any stochastic state. The default implementation is a no-op,
        suitable for deterministic load profiles.

        Stochastic models should override this to:
        1. Store the provided random number generator
        2. Clear any cached stochastic schedules from previous runs
        3. Reset wrapped/nested profiles

        Args:
            rng: Random number generator for stochastic load variation.
                If None, model should use internal/default generator.
                Provided by Monte Carlo simulator to ensure reproducibility.
            n_years: Number of simulation years for preparation.
                If None, model may skip precomputation.

        Notes:
            - Default implementation does nothing (for deterministic models)
            - Stochastic models must store rng for get_hourly_load_kw() calls
            - Called once per Monte Carlo path
        """

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get electricity consumption for specific hour.

        Returns the load demand (in kW) at the specified time point. This is
        the core method called repeatedly during simulation to determine hourly
        consumption for energy system optimization and economic calculations.

        Args:
            year_index: Simulation year (0-based integer).
                0 = first year, 1 = second year, etc.
                Allows modeling consumption changes over time (rare).
            month_in_year: Month within year (0-based integer, 0-11).
                0 = January, 1 = February, ..., 11 = December.
                Used for seasonal consumption patterns.
            day_in_month: Day within month (0-based integer).
                0 = first day, 1 = second day, etc.
                Range: 0-27 (Feb) to 0-30 (31-day months).
                Used for daily stochastic variations.
            hour_in_day: Hour within day (0-based integer, 0-23).
                0 = midnight-1am, 12 = noon-1pm, 23 = 11pm-midnight.
                Core dimension for load profiles (diurnal patterns).
            weekday: Day of week (0-based integer, 0-6).
                0 = Monday, 1 = Tuesday, ..., 6 = Sunday.
                Used for weekday/weekend patterns.

        Returns:
            float: Electricity consumption in kW (kilowatts, power not energy).
                Must be non-negative. Typical residential values: 0.05-5.0 kW.
                Example: 0.150 kW = 150W standby, 3.0 kW = cooking + appliances.

        Raises:
            NotImplementedError: If subclass doesn't implement this method.

        Example:
            ```python
            # Implementing in a subclass
            class ConstantLoad(LoadProfile):
                def __init__(self, constant_kw):
                    self.constant_kw = constant_kw

                def get_hourly_load_kw(self, year_index, month_in_year,
                                       day_in_month, hour_in_day, weekday):
                    return self.constant_kw

            # Usage
            load = ConstantLoad(constant_kw=0.2)
            consumption = load.get_hourly_load_kw(0, 0, 0, 12, 0)
            print(f"{consumption} kW")  # 0.2 kW
            ```

        Notes:
            - Called very frequently: n_mc × n_years × 8760 times per MC simulation
            - Should be fast (avoid heavy computation; precompute in reset_for_run)
            - Return value used for energy system optimization (battery, grid)
            - Must be deterministic for given inputs within a single run
            - Stochastic variation comes from reset_for_run(), not here
        """
        raise NotImplementedError
