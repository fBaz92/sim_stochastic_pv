from __future__ import annotations

from typing import List, Tuple

import numpy as np

MONTH_LENGTHS: List[int] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
"""Number of days in each month (January through December)."""


def build_calendar(
    n_years: int,
    start_weekday: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns arrays describing the calendar at daily resolution.

    Outputs:
      - month_index_for_day: global month index (0..12*n_years-1) for each day
      - month_in_year_for_day: month index 0..11
      - year_index_for_day: year index 0..n_years-1
      - day_in_month_for_day: day number within the month (0-based)
      - weekday_for_day: weekday index (0=Mon .. 6=Sun)
    """
    if not (0 <= start_weekday <= 6):
        raise ValueError("start_weekday must be between 0 (Mon) and 6 (Sun)")

    month_index_for_day = []
    month_in_year_for_day = []
    year_index_for_day = []
    day_in_month_for_day = []
    weekday_for_day = []

    weekday = start_weekday

    for y in range(n_years):
        for m, days in enumerate(MONTH_LENGTHS):
            for day in range(days):
                month_index_for_day.append(y * 12 + m)
                month_in_year_for_day.append(m)
                year_index_for_day.append(y)
                day_in_month_for_day.append(day)
                weekday_for_day.append(weekday)
                weekday = (weekday + 1) % 7

    return (
        np.array(month_index_for_day, dtype=int),
        np.array(month_in_year_for_day, dtype=int),
        np.array(year_index_for_day, dtype=int),
        np.array(day_in_month_for_day, dtype=int),
        np.array(weekday_for_day, dtype=int),
    )
