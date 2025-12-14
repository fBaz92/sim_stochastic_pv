"""
Helper functions for load profile calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from .base import LoadProfile, BL_TABLE
from ...calendar_utils import MONTH_LENGTHS

def get_load_w(
    month_index: int,
    hour_in_month: int,
    first_weekday_of_month: int = 0,
    bl_table: np.ndarray = BL_TABLE,
) -> float:
    """
    Calculate ARERA-based load consumption for specific hour in month.

    Determines the appropriate Italian ARERA tariff band (F1/F2/F3) for the
    given hour based on day of week and time of day, then returns the
    corresponding base load value from the lookup table.

    This function is the core of AreraLoadProfile, converting calendar position
    (month, hour) into consumption values using the Italian time-of-use tariff
    structure.

    Args:
        month_index: Month index (0-based integer, 0-11).
            0 = January, 1 = February, ..., 11 = December.
        hour_in_month: Absolute hour within the month (0-based integer).
            0 = first hour of month (day 0, hour 0).
            Range: 0 to (24 × days_in_month - 1).
            Example: hour 25 = day 1, hour 1 (second day, second hour).
        first_weekday_of_month: Weekday of first day of month (0-based integer).
            0 = Monday, 1 = Tuesday, ..., 6 = Sunday.
            Used to compute weekday for any hour_in_month.
            Default: 0 (month starts on Monday).
        bl_table: Base load lookup table in Watts (shape: 12 × 3).
            Rows: 12 months (Jan=0 ... Dec=11).
            Columns: 3 ARERA bands (F1, F2, F3).
            Default: BL_TABLE (Italian residential empirical data).

    Returns:
        float: Base electricity load in Watts for the specified hour.
            Looked up from bl_table[month, band] where band is determined
            by ARERA tariff rules.

    Raises:
        ValueError: If month_index not in [0, 11].
        ValueError: If hour_in_month not in valid range for the month.

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.load_profiles import (
            get_load_w,
            BL_TABLE
        )

        # January (month 0), first day (hours 0-23), starts on Monday (0)
        # Hour 10 = day 0, hour 10 = Monday 10am → F1 (peak)
        load_peak = get_load_w(
            month_index=0,
            hour_in_month=10,
            first_weekday_of_month=0,
            bl_table=BL_TABLE
        )  # Returns BL_TABLE[0, 0] ≈ 110.67W (Jan F1)

        # Hour 50 = day 2 (50÷24=2), hour 2 (50%24=2) = Wednesday 2am → F3 (off-peak)
        load_offpeak = get_load_w(
            month_index=0,
            hour_in_month=50,
            first_weekday_of_month=0
        )  # Returns BL_TABLE[0, 2] ≈ 157.05W (Jan F3)

        # Custom base load table
        bl_custom = np.array([
            [400.0, 250.0, 120.0],  # Jan: F1=400W, F2=250W, F3=120W
            # ... (11 more months)
        ])
        load_custom = get_load_w(0, 10, 0, bl_custom)  # 400W
        ```

    Notes:
        - Used internally by AreraLoadProfile.get_hourly_load_kw()
        - ARERA bands (see _get_band_arera for details):
          - F1 (peak): Mon-Fri 8am-7pm
          - F2 (mid): Mon-Fri 7am+19-22pm, Sat 7am-10pm
          - F3 (off): Nights, Sundays
        - February assumed 28 days (no leap year support)
        - Error messages in Italian for historical reasons
    """
    if not (0 <= month_index < 12):
        raise ValueError("month_index deve essere tra 0 e 11")

    days_in_month = MONTH_LENGTHS[month_index]
    max_hours = 24 * days_in_month
    if not (0 <= hour_in_month < max_hours):
        raise ValueError(
            f"hour_in_month deve essere tra 0 e {max_hours-1} per il mese {month_index}"
        )

    day_in_month = hour_in_month // 24
    hour = hour_in_month % 24
    weekday = (first_weekday_of_month + day_in_month) % 7
    band = _get_band_arera(weekday, hour)

    if band == "F1":
        bl = bl_table[month_index, 0]
    elif band == "F2":
        bl = bl_table[month_index, 1]
    else:
        bl = bl_table[month_index, 2]

    return bl


def _get_band_arera(weekday: int, hour: int) -> str:
    """
    Determine Italian ARERA tariff band for given weekday and hour.

    Implements the Italian ARERA (Autorità di Regolazione per Energia Reti e
    Ambiente) time-of-use tariff structure, which divides hours into three
    bands based on typical grid demand patterns.

    ARERA Band Rules:
    - F1 (Peak): Monday-Friday, 8am-7pm (08:00-18:59)
      Highest rates during weekday business hours (peak demand).
    - F2 (Mid-peak): Monday-Friday 7am + 7-10pm, Saturday 7am-10pm
      Intermediate rates during shoulder hours and Saturdays.
    - F3 (Off-peak): All other hours (nights, Sundays, holidays)
      Lowest rates during low-demand periods.

    Args:
        weekday: Day of week (0-based integer, 0-6).
            0 = Monday, 1 = Tuesday, ..., 5 = Saturday, 6 = Sunday.
        hour: Hour of day (0-based integer, 0-23).
            0 = midnight-1am, 8 = 8am-9am, 18 = 6pm-7pm, 23 = 11pm-midnight.

    Returns:
        str: ARERA tariff band identifier.
            - "F1": Peak hours (weekday daytime)
            - "F2": Mid-peak hours (weekday shoulder + Saturday daytime)
            - "F3": Off-peak hours (nights, Sundays)

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import _get_band_arera

        # Monday 10am → F1 (peak)
        band1 = _get_band_arera(weekday=0, hour=10)  # "F1"

        # Monday 7am → F2 (shoulder)
        band2 = _get_band_arera(weekday=0, hour=7)   # "F2"

        # Monday 3am → F3 (night)
        band3 = _get_band_arera(weekday=0, hour=3)   # "F3"

        # Saturday 2pm → F2 (Saturday daytime)
        band4 = _get_band_arera(weekday=5, hour=14)  # "F2"

        # Sunday 2pm → F3 (Sunday = always off-peak)
        band5 = _get_band_arera(weekday=6, hour=14)  # "F3"
        ```

    Detailed Rules:
        Sunday (weekday=6):
            All hours → F3

        Monday-Friday (weekday=0-4):
            Hour 8-18 (8am-7pm) → F1
            Hour 7, 19-22 (7am, 7pm-11pm) → F2
            Hour 0-6, 23 (midnight-7am, 11pm-midnight) → F3

        Saturday (weekday=5):
            Hour 7-22 (7am-11pm) → F2
            Hour 0-6, 23 (midnight-7am, 11pm-midnight) → F3

    Notes:
        - Designed for Italian electricity market
        - Based on ARERA regulatory framework
        - Holidays treated as Sundays (not implemented here)
        - Hour ranges are inclusive: 8 <= hour <= 18 includes hours 8-18
    """
    if weekday == 6:
        return "F3"

    if weekday in range(0, 5):
        if 8 <= hour <= 18:
            return "F1"
        elif hour == 7 or (19 <= hour <= 22):
            return "F2"
        else:
            return "F3"
    else:
        if 7 <= hour <= 22:
            return "F2"
        else:
            return "F3"


def make_flat_monthly_load_profiles(base_load_w: float = 140.0) -> np.ndarray:
    """
    Create constant load profile array (same value for all months and hours).

    Generates a 12×24 array filled with a single load value, suitable for
    initializing MonthlyAverageLoadProfile with constant consumption. Useful
    for simple baseline scenarios, standby loads, or testing.

    Args:
        base_load_w: Constant hourly load in Watts (non-negative float).
            Applied to all months and all hours uniformly.
            Default: 140.0W (typical Italian residential standby consumption).
            Examples:
            - 80W: Minimal standby (fridge, routers)
            - 150W: Light residential baseline
            - 300W: Active household consumption

    Returns:
        np.ndarray: Array of shape (12, 24) with dtype=float.
            All elements set to base_load_w.
            Rows: 12 months (Jan=0 ... Dec=11).
            Columns: 24 hours (0=midnight-1am ... 23=11pm-midnight).

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.load_profiles import (
            make_flat_monthly_load_profiles,
            MonthlyAverageLoadProfile
        )

        # Create constant 200W profile
        profiles = make_flat_monthly_load_profiles(base_load_w=200.0)
        print(profiles.shape)  # (12, 24)
        print(profiles[0, 0])  # 200.0
        print(profiles[6, 12]) # 200.0 (all values identical)

        # Use with MonthlyAverageLoadProfile
        load_model = MonthlyAverageLoadProfile(profiles)
        load = load_model.get_hourly_load_kw(0, 0, 0, 12, 0)  # 0.2 kW

        # Different constant values for different scenarios
        standby = make_flat_monthly_load_profiles(80.0)   # Away mode
        moderate = make_flat_monthly_load_profiles(200.0) # Light usage
        active = make_flat_monthly_load_profiles(400.0)   # High usage
        ```

    Notes:
        - Simplest possible load profile (no variation)
        - No seasonal or diurnal patterns
        - Primarily used as building block for more complex profiles
        - Can serve as base for VariableLoadProfile decoration
        - Efficient: Uses np.full (no loops)
    """
    return np.full((12, 24), base_load_w, dtype=float)


