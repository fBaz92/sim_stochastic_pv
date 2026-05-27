"""
Weekly-pattern load profile implementation.

Extends the monthly-average baseline with a 7×24 modulation matrix that
captures weekday/weekend differences without breaking the monthly energy
budget preserved by the baseline.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .base import LoadProfile

# ---------------------------------------------------------------------------
# Preset weekly patterns
# ---------------------------------------------------------------------------
# Each preset is a (7, 24) array of relative weights, where row 0 = Monday,
# row 6 = Sunday, and column h = hour h.  Exact absolute values are
# irrelevant — they are column-normalised before use so that the mean weight
# across all 7 days for each hour equals 1.0, preserving the monthly baseline.
#
# Design rationale for each preset:
#   - residential_typical : standard 3-person family whose adults commute to
#     work Mon–Fri.  Weekday daytime (09–17h) is low (empty house); weekends
#     are high throughout the day.
#   - smart_worker : one or two adults working from home Mon–Fri.  Weekday
#     daytime is medium-high (home office equipment, kettle, etc.); weekends
#     are slightly lower (out more often).
#   - commuter : long-commute worker who leaves early and returns late.
#     Weekday daytime is very low; the evening peak is pushed to 20–22h.
#     Weekends compensate with high daytime loads.
#
# Hour convention: column h represents the hour interval [h, h+1).
# So h=0 = midnight–01:00, h=12 = noon–13:00, h=22 = 22:00–23:00.

def _make_daily_pattern(*values: float) -> list[float]:
    """
    Convenience wrapper that ensures exactly 24 values.

    Args:
        values: Exactly 24 numeric values, one per hour.

    Returns:
        list[float]: The same values as a plain list.

    Raises:
        ValueError: If the number of values is not 24.
    """
    if len(values) != 24:
        raise ValueError(f"Expected 24 hourly values, got {len(values)}")
    return list(values)


# fmt: off
_RESIDENTIAL_TYPICAL_WEEKDAY = _make_daily_pattern(
     5,  4,  3,  3,  4, 15, 40, 55, 35, 25, 22, 25,
    28, 25, 22, 22, 25, 55, 70, 65, 55, 45, 35, 22,
)
_RESIDENTIAL_TYPICAL_WEEKEND = _make_daily_pattern(
     5,  4,  3,  3,  4, 12, 25, 40, 55, 62, 65, 68,
    65, 60, 60, 62, 62, 68, 72, 68, 60, 52, 42, 28,
)
_SMART_WORKER_WEEKDAY = _make_daily_pattern(
     5,  4,  3,  3,  4, 15, 35, 55, 65, 62, 58, 65,
    70, 62, 58, 58, 62, 72, 80, 72, 65, 55, 42, 25,
)
_SMART_WORKER_WEEKEND = _make_daily_pattern(
     5,  4,  3,  3,  4, 12, 25, 40, 50, 55, 58, 60,
    58, 55, 55, 58, 58, 62, 68, 62, 55, 48, 38, 22,
)
_COMMUTER_WEEKDAY = _make_daily_pattern(
     5,  4,  3,  3,  4, 12, 30, 55, 22, 15, 12, 15,
    18, 15, 12, 12, 18, 38, 60, 78, 75, 60, 45, 28,
)
_COMMUTER_WEEKEND = _make_daily_pattern(
     5,  4,  3,  3,  4, 12, 28, 42, 58, 65, 70, 72,
    68, 65, 60, 62, 62, 68, 78, 72, 60, 52, 40, 28,
)
# fmt: on

WEEKLY_PRESETS: Dict[str, np.ndarray] = {
    "residential_typical": np.array(
        [_RESIDENTIAL_TYPICAL_WEEKDAY] * 5 + [_RESIDENTIAL_TYPICAL_WEEKEND] * 2,
        dtype=float,
    ),
    "smart_worker": np.array(
        [_SMART_WORKER_WEEKDAY] * 5 + [_SMART_WORKER_WEEKEND] * 2,
        dtype=float,
    ),
    "commuter": np.array(
        [_COMMUTER_WEEKDAY] * 5 + [_COMMUTER_WEEKEND] * 2,
        dtype=float,
    ),
}
"""
Pre-defined weekly modulation patterns.

Each entry maps a preset name to a NumPy array of shape ``(7, 24)`` where
row 0 = Monday and row 6 = Sunday. The values are *relative weights*: they
will be column-normalised by :class:`WeeklyPatternLoadProfile` so that the
mean weight across all 7 days equals 1.0 for each hour, preserving the
monthly energy budget.

Available presets:

``"residential_typical"``
    Standard family whose adults commute to work Mon–Fri.  Weekday daytime
    is low (empty house, 09–17h ≈ half the weekend value); evening peak
    18–22h is prominent.  Weekends show higher and broader daytime
    consumption.

``"smart_worker"``
    One or two adults working from home Mon–Fri.  Weekday daytime (09–17h)
    is medium-high (office equipment, heating/cooling, breaks); weekends are
    slightly lower because the occupants are out more.

``"commuter"``
    Long-commute worker who leaves before 08h and returns after 19h.  Weekday
    daytime load is very low (essentially standby only); the evening peak is
    pushed to 20–22h.  Weekends compensate with high all-day occupancy.
"""


# ---------------------------------------------------------------------------
# WeeklyPatternLoadProfile
# ---------------------------------------------------------------------------

class WeeklyPatternLoadProfile(LoadProfile):
    """
    Load profile that modulates a monthly baseline with a 7×24 weekly pattern.

    Combines two sources of information to produce a realistic hourly load
    that respects:

    1. **Seasonal variation** — encoded in the ``monthly_profiles_w`` baseline
       (shape ``(12, 24)``), identical to :class:`MonthlyAverageLoadProfile`.
    2. **Intra-week variation** — encoded in ``weekly_pattern_w``
       (shape ``(7, 24)``), capturing weekday vs. weekend differences that
       the simple monthly average ignores.

    The two are combined by *column-normalising* the weekly pattern before
    use, so that for each hour ``h`` the mean weight across all 7 days of the
    week equals exactly 1.0:

    .. code-block::

        col_mean[h]        = mean(weekly_pattern_w[:, h])   # scalar per hour
        weight[d, h]       = weekly_pattern_w[d, h] / col_mean[h]
        load(m, d, h) kW   = baseline_kw[m, h] * weight[d, h]

    This normalisation **preserves the monthly energy budget**: the average
    load over a uniformly-distributed week (equal probability of each weekday)
    equals the monthly baseline exactly.

    Attributes:
        monthly_profiles_kw: ``np.ndarray`` of shape ``(12, 24)``, the monthly
            baseline in kW (converted from the Watt input on construction).
        weekly_pattern_w: ``np.ndarray`` of shape ``(7, 24)``, the raw weekly
            pattern as supplied (Watts; *not* normalised — for reference only).
        _weekly_weights: ``np.ndarray`` of shape ``(7, 24)``, the normalised
            weights used internally by :meth:`get_hourly_load_kw`.

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.load_profiles import (
            WeeklyPatternLoadProfile,
            WEEKLY_PRESETS,
            make_flat_monthly_load_profiles,
        )

        baseline_w = make_flat_monthly_load_profiles(base_load_w=250.0)  # 250 W flat
        pattern    = WEEKLY_PRESETS["residential_typical"]                # (7, 24)

        load = WeeklyPatternLoadProfile(baseline_w, pattern)

        # Monday 12:00 (weekday daytime — low for residential_typical)
        mon_noon = load.get_hourly_load_kw(0, 5, 14, 12, 0)   # weekday=0 (Mon)
        # Saturday 12:00 (weekend daytime — high)
        sat_noon = load.get_hourly_load_kw(0, 5, 17, 12, 5)   # weekday=5 (Sat)
        print(f"Mon noon: {mon_noon:.3f} kW, Sat noon: {sat_noon:.3f} kW")
        # Mon noon: ~0.160 kW,  Sat noon: ~0.295 kW  (exact values depend on baseline)

        # Confirm that the weekly mean equals the baseline (hour 12)
        weights_h12 = [load._weekly_weights[d, 12] for d in range(7)]
        print(f"Mean weight hour 12: {sum(weights_h12)/7:.4f}")  # → 1.0000
        ```

    Notes:
        - Deterministic: no stochastic state — :meth:`reset_for_run` is a no-op.
        - The ``weekday`` parameter is actually *used* (unlike most other
          profiles before Phase 5) — it selects the row of ``_weekly_weights``.
        - If all 7 values in a column are zero (degenerate pattern), the
          normalisation defaults to 1.0 for that column, returning the
          baseline unchanged.
        - Suitable for use as a sub-profile inside :class:`HomeAwayLoadProfile`
          (home or away side) or as a standalone load profile.
        - For the preset patterns see :data:`WEEKLY_PRESETS`.
    """

    def __init__(
        self,
        monthly_profiles_w: np.ndarray,
        weekly_pattern_w: np.ndarray,
    ) -> None:
        """
        Initialise the weekly-pattern load profile.

        Validates shapes, stores the baseline in kW, and pre-computes the
        normalised weight matrix so that :meth:`get_hourly_load_kw` is an O(1)
        lookup (no division at query time).

        Args:
            monthly_profiles_w: Hourly load baseline in **Watts**, shape
                ``(12, 24)``.  Rows = months (Jan=0 … Dec=11), columns =
                hours (0–23).  Values must be non-negative; negative values
                will produce negative loads, which is physically nonsensical.
            weekly_pattern_w: Weekly modulation matrix in arbitrary positive
                units, shape ``(7, 24)``.  Row 0 = Monday, row 6 = Sunday.
                The column-wise mean is used for normalisation, so only the
                *relative* values within each column (hour) matter — multiplying
                all values by a constant has no effect on the output.
                Supplied as ``weekly_pattern_w`` for serialisation round-trips;
                not used directly after construction.

        Raises:
            ValueError: If ``monthly_profiles_w`` does not have shape ``(12, 24)``.
            ValueError: If ``weekly_pattern_w`` does not have shape ``(7, 24)``.

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.load_profiles import (
                WeeklyPatternLoadProfile, WEEKLY_PRESETS,
            )
            baseline = np.full((12, 24), 200.0)   # 200 W flat for all months/hours
            pattern  = WEEKLY_PRESETS["commuter"]  # low weekday daytime, high evening
            load = WeeklyPatternLoadProfile(baseline, pattern)
            ```

        Notes:
            - Baseline stored in kW internally (divided by 1000) for consistency
              with the rest of the load profile ecosystem.
            - ``weekly_pattern_w`` is stored as-is (in original units) for
              serialisation and inspection; the normalised weights live in
              ``_weekly_weights``.
        """
        if np.asarray(monthly_profiles_w).shape != (12, 24):
            raise ValueError(
                f"monthly_profiles_w must have shape (12, 24), "
                f"got {np.asarray(monthly_profiles_w).shape}"
            )
        if np.asarray(weekly_pattern_w).shape != (7, 24):
            raise ValueError(
                f"weekly_pattern_w must have shape (7, 24), "
                f"got {np.asarray(weekly_pattern_w).shape}"
            )

        self.monthly_profiles_kw: np.ndarray = np.asarray(monthly_profiles_w, dtype=float) / 1000.0
        self.weekly_pattern_w: np.ndarray = np.asarray(weekly_pattern_w, dtype=float).copy()

        # Pre-compute column-normalised weights.
        # For each hour h: weight[d, h] = pattern[d, h] / mean_d(pattern[:, h])
        # If the column mean is zero (degenerate), default to 1.0 (pass-through).
        col_means = self.weekly_pattern_w.mean(axis=0)          # shape (24,)
        safe_means = np.where(col_means > 0.0, col_means, 1.0)  # avoid /0
        self._weekly_weights: np.ndarray = self.weekly_pattern_w / safe_means[np.newaxis, :]

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Return the modulated hourly load in kW.

        Combines the monthly baseline with the pre-computed weekly weight for
        the given weekday and hour.  This is the *only* ``LoadProfile`` method
        in the codebase that actually reads the ``weekday`` argument — all
        prior implementations (``MonthlyAverageLoadProfile``, ``AreraLoadProfile``,
        etc.) ignored it.

        Args:
            year_index: Ignored — the profile is year-stationary.
            month_in_year: Month index 0–11 (Jan=0, Dec=11).  Selects the
                row of ``monthly_profiles_kw`` for the seasonal baseline.
            day_in_month: Ignored — day-of-month has no effect (only the
                weekday matters).
            hour_in_day: Hour index 0–23.  Selects the column in both the
                baseline and the weight matrix.
            weekday: Day of week 0–6 (Mon=0, Sun=6).  Selects the row of
                the normalised weight matrix.

        Returns:
            float: Load in kW = ``monthly_baseline_kw[month, hour]`` ×
                ``weekly_weight[weekday, hour]``.  Always non-negative when
                inputs are non-negative.

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.load_profiles import (
                WeeklyPatternLoadProfile, WEEKLY_PRESETS,
            )
            baseline = np.full((12, 24), 300.0)   # 300 W all months/hours
            load     = WeeklyPatternLoadProfile(baseline, WEEKLY_PRESETS["smart_worker"])

            # Query weekday morning (Mon, 08:00)
            kw = load.get_hourly_load_kw(0, 0, 0, 8, 0)  # Mon=0
            print(f"{kw:.3f} kW")
            # Saturday morning, same hour
            kw_sat = load.get_hourly_load_kw(0, 0, 5, 8, 5)  # Sat=5
            print(f"{kw_sat:.3f} kW")
            ```

        Notes:
            - O(1) lookup: both lookups are direct array indexing.
            - Deterministic within a run; ``reset_for_run`` is a no-op.
        """
        baseline_kw = self.monthly_profiles_kw[month_in_year, hour_in_day]
        weight = self._weekly_weights[weekday, hour_in_day]
        return float(baseline_kw * weight)
