"""
Presence calendar — the occupancy backbone of home/away load profiles.

A :class:`HomeAwayLoadProfile` switches between a *home* profile (occupied
days) and an *away* profile (standby days). Until now the only way to drive
that switch was to hand-author the two arrays ``min_days_home`` /
``max_days_home`` (12 integers each). Those arrays are powerful but opaque:
"23 days in March" says nothing about *which* days, and gives the user no
mental model for a vacation home that is visited every weekend plus the
whole of August.

This module introduces a small, human-meaningful description of *when a
building is occupied* and converts it into exactly those two arrays. The
runtime model is unchanged — the calendar is a friendlier front-end to the
same ``min_days_home`` / ``max_days_home`` knobs:

    PresenceCalendar  ──to_min_max_days_home()──▶  (min_days, max_days)
                                                         │
                                                         ▼
                                              HomeAwayLoadProfile

The per-month description is intentionally coarse — *counts of days*, not a
day-by-day schedule — because the load model itself only samples a count per
month and then scatters those days at random. Declaring more structure than
the model consumes would be a false precision.

It also ships:

- :class:`HouseTypePreset` / :data:`HOUSE_TYPE_PRESETS` — reference data for
  the "quick fit from a bill" entry level (floor area + a typical annual
  consumption per dwelling archetype). Used by the UI to pre-fill a sensible
  annual-kWh starting point; it does not feed the occupancy maths.
- :data:`DEFAULT_PRESENCE_CALENDAR` — a typical primary residence
  (~23 occupied days per month).
- :data:`PRESENCE_CALENDAR_PRESETS` — three named starting points
  (primary residence, summer vacation home, weekend house).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Any, Dict, List, Mapping, Tuple

from ...calendar_utils import MONTH_LENGTHS

# A 7-day week has 2 weekend days (Saturday, Sunday). Used to estimate how
# many of a month's "loose" days (those not inside a full week-block) fall on
# a weekend, without committing to a specific calendar alignment.
WEEKEND_DAYS_PER_WEEK: float = 2.0
DAYS_PER_WEEK: int = 7

# Validation bounds for a single month's pattern. They keep the editor honest
# (a month cannot contain six full weeks) without the runtime having to defend
# against nonsense later — to_min_max_days_home() still clamps to the real
# month length as a final safety net.
MAX_FULL_WEEKS: int = 5
MAX_EXTRA_WEEKDAYS: int = 7

# Number of calendar days in a non-leap year (sum of MONTH_LENGTHS). Cached
# here so annual_presence_fraction() does not recompute it on every call.
DAYS_PER_YEAR: int = sum(MONTH_LENGTHS)


@dataclass(frozen=True)
class MonthPresencePattern:
    """
    How often a building is occupied during one calendar month.

    Describes occupancy as a small set of human-meaningful counts rather than
    a day-by-day schedule, because the downstream :class:`HomeAwayLoadProfile`
    only consumes a *count* of home days per month (it then scatters them at
    random). The four fields decompose that count into parts a non-expert can
    reason about:

        deterministic_home_days ≈ full_weeks * 7
                                + (weekend days of the leftover, if weekends)
                                + extra_weekdays

    The ``visit_probability`` then turns the *remaining* (otherwise-away) days
    into a soft upper bound: the month's home-day count is sampled uniformly
    between the deterministic floor and that widened ceiling.

    Attributes:
        weekends: Whether weekends count as occupied days. ``True`` for any
            building visited on Saturdays/Sundays (the common case);
            ``False`` for e.g. a weekday-only city pied-à-terre.
        full_weeks: Number of whole Mon–Sun weeks spent at the building
            (integer, 0–5). Each contributes 7 occupied days.
        extra_weekdays: Additional isolated weekdays occupied on top of the
            full weeks and weekends (integer, 0–7). Models the odd extra day.
        visit_probability: Chance, per otherwise-away day, of a short visit
            (float, 0.0–1.0). Widens the sampled home-day band upward; 0.0
            means the deterministic floor is also the ceiling (no surprise
            visits).

    Raises:
        ValueError: If ``full_weeks`` ∉ [0, 5], ``extra_weekdays`` ∉ [0, 7],
            or ``visit_probability`` ∉ [0.0, 1.0].

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import MonthPresencePattern

        # A summer month at a vacation home: there the whole time.
        august = MonthPresencePattern(full_weeks=4, weekends=True)

        # A winter month at the same home: only the odd weekend, plus a
        # chance of a quick midweek check.
        january = MonthPresencePattern(
            full_weeks=0, weekends=True, visit_probability=0.1
        )
        ```

    Notes:
        - The fields are *counts*, not specific dates; the simulator chooses
          which days are home at random each Monte Carlo path.
        - Frozen so a calendar can be hashed and shared safely across paths.
        - The weekend-day estimate uses the average 2-in-7 ratio rather than a
          concrete calendar, so the result is independent of which weekday the
          month happens to start on.
    """

    weekends: bool = True
    full_weeks: int = 4
    extra_weekdays: int = 0
    visit_probability: float = 0.0

    def __post_init__(self) -> None:
        if not (0 <= self.full_weeks <= MAX_FULL_WEEKS):
            raise ValueError(
                f"full_weeks must be in [0, {MAX_FULL_WEEKS}], got {self.full_weeks}"
            )
        if not (0 <= self.extra_weekdays <= MAX_EXTRA_WEEKDAYS):
            raise ValueError(
                f"extra_weekdays must be in [0, {MAX_EXTRA_WEEKDAYS}], "
                f"got {self.extra_weekdays}"
            )
        if not (0.0 <= self.visit_probability <= 1.0):
            raise ValueError(
                f"visit_probability must be in [0.0, 1.0], got {self.visit_probability}"
            )

    def deterministic_home_days(self, days_in_month: int) -> int:
        """
        Number of days that are *certainly* occupied this month.

        This is the floor of the home-day band — the count reached even with
        ``visit_probability == 0``. It sums the full-week days, the weekend
        days of whatever portion of the month is left over (when ``weekends``
        is set), and the loose extra weekdays, then clamps to the month length.

        Args:
            days_in_month: Calendar length of the month (28–31).

        Returns:
            int: Occupied-day floor in ``[0, days_in_month]``.

        Example:
            ```python
            p = MonthPresencePattern(full_weeks=3, extra_weekdays=2, weekends=False)
            p.deterministic_home_days(31)  # 3*7 + 2 = 23
            ```

        Notes:
            - Weekend days are estimated as ``2/7`` of the leftover days, so a
              full-of-weekends month is not double-counted against the full
              weeks already claimed.
        """
        full_days = min(self.full_weeks * DAYS_PER_WEEK, days_in_month)
        leftover = days_in_month - full_days
        weekend_days = 0
        if self.weekends:
            weekend_days = int(round(leftover * WEEKEND_DAYS_PER_WEEK / DAYS_PER_WEEK))
        total = full_days + weekend_days + self.extra_weekdays
        return int(min(max(total, 0), days_in_month))

    def min_max_home_days(self, days_in_month: int) -> Tuple[int, int]:
        """
        Lower and upper bounds on occupied days for one month.

        The lower bound is :meth:`deterministic_home_days`; the upper bound
        adds the expected number of opportunistic visits among the remaining
        away days (``floor(remaining * visit_probability)``). Both are clamped
        to ``[0, days_in_month]`` and ``max >= min`` always holds.

        Args:
            days_in_month: Calendar length of the month (28–31).

        Returns:
            tuple[int, int]: ``(min_days, max_days)`` for this month, both in
            ``[0, days_in_month]``.

        Example:
            ```python
            p = MonthPresencePattern(full_weeks=0, weekends=True, visit_probability=0.5)
            p.min_max_home_days(30)  # e.g. (9, 19): ~9 weekend days, up to ~10 visits
            ```
        """
        min_days = self.deterministic_home_days(days_in_month)
        remaining = days_in_month - min_days
        extra = floor(remaining * self.visit_probability)
        max_days = int(min(min_days + extra, days_in_month))
        return min_days, max_days

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this pattern to a JSON-friendly dict.

        Returns:
            dict: ``{"weekends", "full_weeks", "extra_weekdays",
            "visit_probability"}`` — round-trips through :meth:`from_dict`.
        """
        return {
            "weekends": bool(self.weekends),
            "full_weeks": int(self.full_weeks),
            "extra_weekdays": int(self.extra_weekdays),
            "visit_probability": float(self.visit_probability),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MonthPresencePattern":
        """
        Build a pattern from a JSON-decoded dict, applying defaults.

        Missing keys fall back to the field defaults (a fully home-all-weeks
        month with no visits), so a partial dict is accepted. Out-of-range
        values raise via :meth:`__post_init__`.

        Args:
            raw: Mapping with any subset of the four field keys.

        Returns:
            MonthPresencePattern: Validated instance.

        Raises:
            ValueError: If a present value is outside its allowed range.
        """
        return cls(
            weekends=bool(raw.get("weekends", True)),
            full_weeks=int(raw.get("full_weeks", 4)),
            extra_weekdays=int(raw.get("extra_weekdays", 0)),
            visit_probability=float(raw.get("visit_probability", 0.0)),
        )


@dataclass(frozen=True)
class PresenceCalendar:
    """
    Twelve-month occupancy description for a building.

    Wraps one :class:`MonthPresencePattern` per calendar month (January =
    index 0) and converts the whole year into the ``min_days_home`` /
    ``max_days_home`` arrays that :class:`HomeAwayLoadProfile` consumes. This
    is the single source of truth for *when* a building is occupied; the load
    model only knows about the resulting day counts.

    Attributes:
        months: Exactly 12 :class:`MonthPresencePattern` entries, Jan–Dec.

    Raises:
        ValueError: If ``months`` does not contain exactly 12 entries.

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import (
            PresenceCalendar, MonthPresencePattern, PRESENCE_CALENDAR_PRESETS,
        )

        cal = PRESENCE_CALENDAR_PRESETS["summer_vacation"]
        min_days, max_days = cal.to_min_max_days_home()
        # min_days[6] ≈ 31 (July: there all month), min_days[0] small (January)

        # Feed straight into the runtime model:
        # HomeAwayLoadProfile(home, away, min_days, max_days)
        ```

    Notes:
        - ``to_min_max_days_home`` is the only method the simulator path needs;
          the rest support the UI (expected value, annual fraction, JSON).
    """

    months: Tuple[MonthPresencePattern, ...]

    def __post_init__(self) -> None:
        if len(self.months) != 12:
            raise ValueError(
                f"PresenceCalendar requires exactly 12 month patterns, "
                f"got {len(self.months)}"
            )

    def to_min_max_days_home(self) -> Tuple[List[int], List[int]]:
        """
        Convert the calendar into the runtime occupancy arrays.

        Returns:
            tuple[list[int], list[int]]: ``(min_days_home, max_days_home)``,
            each a 12-element list (Jan–Dec) in ``[0, days_in_month]`` with
            ``max[m] >= min[m]``. Directly usable as the
            :class:`HomeAwayLoadProfile` bounds.

        Example:
            ```python
            cal = PRESENCE_CALENDAR_PRESETS["weekend_house"]
            min_days, max_days = cal.to_min_max_days_home()
            len(min_days)  # 12
            ```
        """
        min_days: List[int] = []
        max_days: List[int] = []
        for month_index, pattern in enumerate(self.months):
            lo, hi = pattern.min_max_home_days(MONTH_LENGTHS[month_index])
            min_days.append(lo)
            max_days.append(hi)
        return min_days, max_days

    def expected_days_home(self) -> List[float]:
        """
        Expected number of occupied days per month.

        Because :class:`HomeAwayLoadProfile` samples the count uniformly in
        ``[min, max]``, the expectation per month is ``(min + max) / 2``. This
        is the figure the UI shows as "~X days" and the basis of
        :meth:`annual_presence_fraction`.

        Returns:
            list[float]: 12 expected day counts (Jan–Dec), each ≥ 0.

        Example:
            ```python
            cal = PRESENCE_CALENDAR_PRESETS["primary_residence"]
            exp = cal.expected_days_home()
            sum(exp)  # ≈ annual occupied days
            ```
        """
        min_days, max_days = self.to_min_max_days_home()
        return [(lo + hi) / 2.0 for lo, hi in zip(min_days, max_days)]

    def annual_presence_fraction(self) -> float:
        """
        Fraction of the year the building is expected to be occupied.

        Computed as ``sum(expected_days_home) / 365``. This is the ``f`` used
        by the bill-based auto-fit to split annual energy between home and
        away regimes.

        Returns:
            float: Occupancy fraction in ``[0.0, 1.0]``.

        Example:
            ```python
            PRESENCE_CALENDAR_PRESETS["primary_residence"].annual_presence_fraction()
            # ~0.76 (occupied roughly three weeks a month)
            PRESENCE_CALENDAR_PRESETS["weekend_house"].annual_presence_fraction()
            # ~0.29 (weekends only)
            ```

        Notes:
            - Uses a 365-day year (no leap-year handling), consistent with
              :data:`MONTH_LENGTHS` and the rest of the simulator.
        """
        fraction = sum(self.expected_days_home()) / float(DAYS_PER_YEAR)
        return float(min(max(fraction, 0.0), 1.0))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the calendar to a JSON-friendly dict.

        Returns:
            dict: ``{"months": [<12 pattern dicts>]}`` — round-trips through
            :meth:`from_dict`.
        """
        return {"months": [m.to_dict() for m in self.months]}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PresenceCalendar":
        """
        Build a calendar from a JSON-decoded dict.

        Args:
            raw: Mapping with a ``"months"`` key holding a 12-element list of
                pattern dicts (see :meth:`MonthPresencePattern.from_dict`).

        Returns:
            PresenceCalendar: Validated instance.

        Raises:
            ValueError: If ``"months"`` is absent or not a 12-element list, or
                if any month pattern is out of range.
        """
        months_raw = raw.get("months")
        if not isinstance(months_raw, (list, tuple)) or len(months_raw) != 12:
            raise ValueError(
                "PresenceCalendar.from_dict requires a 'months' list of 12 entries"
            )
        months = tuple(MonthPresencePattern.from_dict(m) for m in months_raw)
        return cls(months=months)


@dataclass(frozen=True)
class HouseTypePreset:
    """
    Reference archetype for the "quick fit from a bill" entry level.

    Pairs a dwelling type with a plausible floor area and a typical annual
    electricity consumption. The UI uses ``baseline_annual_kwh`` to pre-fill
    the annual-kWh field when the user picks a house type, giving a sensible
    starting point before they type their own bill figure. It is *reference
    data only* — it never enters the occupancy or scaling maths, which are
    driven entirely by the user's annual kWh and presence calendar.

    Attributes:
        label_it: Italian display label for the dropdown (audience language).
        floor_area_m2: Representative heated floor area (m²).
        baseline_annual_kwh: Typical annual electricity consumption (kWh/year)
            for this archetype, used to pre-fill the bill field.

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import HOUSE_TYPE_PRESETS

        std = HOUSE_TYPE_PRESETS["apartment_standard"]
        std.baseline_annual_kwh  # ~2700 kWh/year
        ```
    """

    label_it: str
    floor_area_m2: float
    baseline_annual_kwh: float


# Typical Italian residential archetypes. The annual figures are deliberately
# round, order-of-magnitude defaults (ISTAT/ARERA residential averages sit
# around 2700 kWh/year for a standard household) — they only seed the UI; the
# user's real bill overrides them.
HOUSE_TYPE_PRESETS: Dict[str, HouseTypePreset] = {
    "apartment_small": HouseTypePreset(
        label_it="Appartamento piccolo",
        floor_area_m2=55.0,
        baseline_annual_kwh=1800.0,
    ),
    "apartment_standard": HouseTypePreset(
        label_it="Appartamento standard",
        floor_area_m2=90.0,
        baseline_annual_kwh=2700.0,
    ),
    "house_standard": HouseTypePreset(
        label_it="Casa indipendente",
        floor_area_m2=130.0,
        baseline_annual_kwh=3600.0,
    ),
    "house_large": HouseTypePreset(
        label_it="Casa grande",
        floor_area_m2=200.0,
        baseline_annual_kwh=5000.0,
    ),
    "vacation_home": HouseTypePreset(
        label_it="Casa vacanze",
        floor_area_m2=80.0,
        baseline_annual_kwh=1500.0,
    ),
}


def _uniform_calendar(pattern: MonthPresencePattern) -> PresenceCalendar:
    """
    Build a calendar that repeats one month pattern across all 12 months.

    Args:
        pattern: The :class:`MonthPresencePattern` to repeat.

    Returns:
        PresenceCalendar: Twelve identical months.
    """
    return PresenceCalendar(months=tuple(pattern for _ in range(12)))


# A typical primary residence: occupied ~23 days/month (≈76% of the year),
# the odd few days away for travel. Used as the default when no calendar is
# supplied for a home/away profile.
DEFAULT_PRESENCE_CALENDAR: PresenceCalendar = _uniform_calendar(
    MonthPresencePattern(full_weeks=3, extra_weekdays=2, weekends=False, visit_probability=0.0)
)


def _build_summer_vacation_calendar() -> PresenceCalendar:
    """
    Build the "summer vacation home" preset (full June–August, weekends else).

    Returns:
        PresenceCalendar: There all summer (months 5–7), only weekends with a
        small visit chance the rest of the year.
    """
    summer = MonthPresencePattern(full_weeks=4, weekends=True, visit_probability=0.0)
    off_season = MonthPresencePattern(
        full_weeks=0, weekends=True, visit_probability=0.1
    )
    months = []
    for month_index in range(12):
        if month_index in (5, 6, 7):  # June, July, August
            months.append(summer)
        else:
            months.append(off_season)
    return PresenceCalendar(months=tuple(months))


# Three named starting points for the presence-calendar editor. The UI offers
# them as one-click presets; the user then fine-tunes individual months.
PRESENCE_CALENDAR_PRESETS: Dict[str, PresenceCalendar] = {
    "primary_residence": _uniform_calendar(
        MonthPresencePattern(full_weeks=3, extra_weekdays=2, weekends=True, visit_probability=0.0)
    ),
    "summer_vacation": _build_summer_vacation_calendar(),
    "weekend_house": _uniform_calendar(
        MonthPresencePattern(full_weeks=0, extra_weekdays=0, weekends=True, visit_probability=0.0)
    ),
}
