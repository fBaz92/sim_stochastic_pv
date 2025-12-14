"""
Italian ARERA tariff-based load profile implementation.
"""

from __future__ import annotations

import numpy as np

from .base import LoadProfile, BL_TABLE
from .helpers import get_load_w


class AreraLoadProfile(LoadProfile):
    """
    Italian ARERA tariff-based load profile (F1/F2/F3 bands).

    Models electricity consumption following the Italian ARERA (Autorità di
    Regolazione per Energia Reti e Ambiente) time-of-use tariff structure.
    The ARERA system divides hours into three price bands based on time and
    day of week, reflecting grid demand patterns:

    - F1 (Peak): Weekday business hours (Mon-Fri, 8am-7pm) - highest rates
    - F2 (Mid-peak): Weekday mornings/evenings + Saturday daytime - medium rates
    - F3 (Off-peak): Nights, Sundays, holidays - lowest rates

    This profile assigns different base consumption levels to each tariff band,
    enabling realistic modeling of Italian residential consumption patterns.
    The base load table (BL_TABLE) contains empirically derived consumption
    values for each month and tariff band.

    Attributes:
        bl_table: Base load table (12 × 3 array) in Watts.
            Rows: Months (Jan=0 ... Dec=11)
            Columns: Tariff bands (F1, F2, F3)
            Default: BL_TABLE (Italian residential averages)

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.load_profiles import AreraLoadProfile, BL_TABLE

        # Use default Italian consumption pattern
        load_model = AreraLoadProfile()

        # Weekday peak hour (Monday 10am, January)
        load_f1 = load_model.get_hourly_load_kw(
            year_index=0, month_in_year=0, day_in_month=0,
            hour_in_day=10, weekday=0  # Mon 10am
        )  # Returns F1 band value for January

        # Sunday night (off-peak)
        load_f3 = load_model.get_hourly_load_kw(
            year_index=0, month_in_year=0, day_in_month=6,
            hour_in_day=22, weekday=6  # Sun 10pm
        )  # Returns F3 band value (lower)

        # Custom base load table (e.g., commercial profile)
        bl_custom = np.array([
            # F1,   F2,   F3 (per month)
            [500.0, 300.0, 100.0],  # Jan (higher winter usage)
            [450.0, 280.0,  90.0],  # Feb
            # ... (12 rows total)
        ])
        load_commercial = AreraLoadProfile(bl_table=bl_custom)
        ```

    Notes:
        - Designed for Italian electricity market simulations
        - Deterministic: No stochastic variation (combine with VariableLoadProfile)
        - Weekday-aware: Distinguishes Mon-Fri, Saturday, Sunday
        - Hour-aware: Different bands throughout the day
        - Default BL_TABLE based on typical Italian residential consumption
    """

    def __init__(self, bl_table: np.ndarray = BL_TABLE) -> None:
        """
        Initialize ARERA load profile with tariff band consumption table.

        Args:
            bl_table: Base load table in Watts (shape: 12 × 3).
                Array structure:
                - Rows (12): Months (Jan=0, Feb=1, ..., Dec=11)
                - Columns (3): ARERA tariff bands [F1, F2, F3]
                  - Column 0: F1 (peak) consumption in Watts
                  - Column 1: F2 (mid-peak) consumption in Watts
                  - Column 2: F3 (off-peak) consumption in Watts
                - Values: Hourly load in Watts (non-negative floats)

                Default: BL_TABLE (Italian residential average from empirical data).
                Typical values: 70-300W depending on season and tariff band.

        Raises:
            ValueError: If bl_table is not shape (12, 3).

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.load_profiles import (
                AreraLoadProfile,
                BL_TABLE
            )

            # Option 1: Use default (Italian residential average)
            model_default = AreraLoadProfile()

            # Option 2: Scaled version (50% higher consumption)
            bl_scaled = BL_TABLE * 1.5
            model_scaled = AreraLoadProfile(bl_table=bl_scaled)

            # Option 3: Custom pattern (high peak, low off-peak)
            bl_custom = np.zeros((12, 3))
            for month in range(12):
                bl_custom[month, 0] = 400.0  # F1 peak
                bl_custom[month, 1] = 200.0  # F2 mid
                bl_custom[month, 2] = 80.0   # F3 off-peak
            model_custom = AreraLoadProfile(bl_table=bl_custom)
            ```

        Notes:
            - Default BL_TABLE reflects Italian residential consumption patterns
            - Table values vary by month (seasonal effects) and band (time-of-use)
            - Higher values in winter months (heating), summer (AC), August (variable)
            - F1 typically higher than F2, F2 higher than F3 (demand-based)
        """
        if bl_table.shape != (12, 3):
            raise ValueError("bl_table must have shape (12, 3)")
        self.bl_table = bl_table

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Get hourly load based on ARERA tariff bands and calendar position.

        Determines the appropriate ARERA tariff band (F1/F2/F3) based on the
        weekday and hour, then returns the corresponding base load value from
        the stored table for the specified month.

        Args:
            year_index: Ignored (no year variation).
            month_in_year: Month index (0-11) to select table row.
            day_in_month: Day within month (0-based) for weekday calculation.
            hour_in_day: Hour of day (0-23) for tariff band determination.
            weekday: Day of week (0=Mon, 6=Sun) for tariff band determination.

        Returns:
            float: Electricity consumption in kW for the specified time.
                Lookup: bl_table[month, band] where band ∈ {F1, F2, F3}
                determined by weekday and hour combination.

        Example:
            ```python
            from sim_stochastic_pv.simulation.load_profiles import AreraLoadProfile

            model = AreraLoadProfile()

            # Monday 10am in January (F1 peak band)
            load_peak = model.get_hourly_load_kw(
                year_index=0,
                month_in_year=0,   # January
                day_in_month=0,
                hour_in_day=10,    # 10am
                weekday=0          # Monday
            )  # Returns bl_table[0, 0] / 1000  (Jan F1)

            # Sunday 3am in January (F3 off-peak)
            load_offpeak = model.get_hourly_load_kw(
                year_index=0,
                month_in_year=0,   # January
                day_in_month=6,
                hour_in_day=3,     # 3am
                weekday=6          # Sunday
            )  # Returns bl_table[0, 2] / 1000  (Jan F3)

            # Saturday 8pm in August (F2 mid-peak)
            load_mid = model.get_hourly_load_kw(
                year_index=0,
                month_in_year=7,   # August
                day_in_month=5,
                hour_in_day=20,    # 8pm
                weekday=5          # Saturday
            )  # Returns bl_table[7, 1] / 1000  (Aug F2)
            ```

        Notes:
            - Tariff band assignment (see _get_band_arera for logic):
              - F1: Mon-Fri 8am-7pm (08:00-18:59)
              - F2: Mon-Fri 7am+19-22pm, Sat 7am-10pm
              - F3: All other hours (nights, Sundays)
            - Deterministic: Same weekday+hour+month always returns same value
            - Typical F1 > F2 > F3 (reflecting demand and pricing)
        """
        hour_in_month = day_in_month * 24 + hour_in_day
        first_weekday = (weekday - day_in_month) % 7
        load_w = get_load_w(
            month_index=month_in_year,
            hour_in_month=hour_in_month,
            first_weekday_of_month=first_weekday,
            bl_table=self.bl_table,
        )
        return load_w / 1000.0
