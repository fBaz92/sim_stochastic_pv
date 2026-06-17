"""
Bill-based auto-fit — the simplest load-profile entry level.

The user has two numbers from an electricity bill ("I consume ~2400 kWh a
year") and a rough idea of when the building is occupied. From just those,
this module reconstructs a runnable home/away load profile by *scaling the
ARERA residential baseline*:

    - the **away** regime is the plain ARERA baseline (standby: fridge,
      routers — the building's irreducible draw);
    - the **home** regime is the same ARERA shape multiplied by a single
      ``home_scale_factor`` chosen so the whole-year energy matches the bill,
      given the expected fraction ``f`` of days the building is occupied.

The energy bookkeeping is a one-line mixture. If ``B`` is the annual energy of
the unscaled ARERA baseline run for a full year, the home days contribute
``f * scale * B`` and the away days ``(1 - f) * B``:

    target = f * scale * B + (1 - f) * B
    ⇒ scale = (target - (1 - f) * B) / (f * B)

That ``scale`` is everything the runtime needs: the home factory is
``AreraLoadProfile(bl_table = BL_TABLE * scale)`` and the away factory is a
plain :class:`AreraLoadProfile`. Occupancy (the ``f`` and the per-month day
counts) comes from a :class:`PresenceCalendar`.

The fit is intentionally crude — it is the "two numbers, one graph" tier. The
Guidato and Esperto levels exist for users who want appliance detail, HVAC,
or hand-tuned hourly shapes. What this tier guarantees is that the resulting
profile's expected annual energy equals the bill (whenever physically
feasible) while preserving the realistic intra-day ARERA shape.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ...calendar_utils import MONTH_LENGTHS
from .arera import AreraLoadProfile
from .base import BL_TABLE, LoadProfile
from .helpers import _get_band_arera
from .presence import PresenceCalendar

# Number of bimonthly billing periods in a year (Italian bills are usually
# issued every two months — six readings per year).
BIMONTHLY_PERIODS_PER_YEAR: int = 6

# When the requested annual energy is below what the away (standby) baseline
# alone consumes, the home scale would go non-positive. We floor it at a small
# positive value so the profile stays runnable; the UI can flag the mismatch.
MIN_HOME_SCALE_FACTOR: float = 0.05

# Below this presence fraction the building is treated as effectively never
# occupied: there are no home days to scale, so the home factor is irrelevant
# and reported as 1.0 to avoid a division by ~0.
MIN_PRESENCE_FRACTION: float = 1e-9


def _band_hours_per_month(days_in_month: int) -> tuple[int, int, int]:
    """
    Count F1/F2/F3 hours in one month under a canonical Monday start.

    The ARERA band of an hour depends only on its weekday and hour. To get a
    deterministic, calendar-alignment-independent annual baseline we treat
    every month as starting on a Monday and count how many of its hours fall
    in each tariff band.

    Args:
        days_in_month: Calendar length of the month (28–31).

    Returns:
        tuple[int, int, int]: Hour counts ``(n_F1, n_F2, n_F3)``; they sum to
        ``24 * days_in_month``.

    Notes:
        - Internal helper for :func:`compute_arera_baseline_annual_kwh`.
        - Using a fixed Monday start makes the baseline a stable reference
          constant rather than a function of an arbitrary calendar offset.
    """
    counts = {"F1": 0, "F2": 0, "F3": 0}
    for day in range(days_in_month):
        weekday = day % 7  # day 0 → Monday
        for hour in range(24):
            counts[_get_band_arera(weekday, hour)] += 1
    return counts["F1"], counts["F2"], counts["F3"]


def compute_arera_baseline_annual_kwh(bl_table: np.ndarray = BL_TABLE) -> float:
    """
    Annual energy of the unscaled ARERA baseline run for a full year.

    Sums the base-load table weighted by how many hours each month spends in
    each tariff band (F1/F2/F3). This is the reference energy ``B`` against
    which a bill is fitted: a household consuming exactly this much, while
    home all year, needs ``home_scale_factor == 1``.

    Args:
        bl_table: Base-load table in Watts, shape ``(12, 3)`` (months × ARERA
            bands). Defaults to the packaged :data:`BL_TABLE`.

    Returns:
        float: Annual electricity consumption in kWh for the (unscaled)
        baseline. With the default table this is ≈ 1196 kWh/year.

    Raises:
        ValueError: If ``bl_table`` is not shape ``(12, 3)``.

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import (
            compute_arera_baseline_annual_kwh, BL_TABLE,
        )

        compute_arera_baseline_annual_kwh()              # ≈ 1196.0
        compute_arera_baseline_annual_kwh(BL_TABLE * 2)  # ≈ 2392.0 (linear)
        ```

    Notes:
        - Deterministic and linear in ``bl_table``.
        - The module-level :data:`ARERA_BASELINE_ANNUAL_KWH` caches the
          default-table result so callers need not recompute it.
    """
    if bl_table.shape != (12, 3):
        raise ValueError(f"bl_table must have shape (12, 3), got {bl_table.shape}")

    total_wh = 0.0
    for month in range(12):
        n_f1, n_f2, n_f3 = _band_hours_per_month(MONTH_LENGTHS[month])
        total_wh += (
            n_f1 * float(bl_table[month, 0])
            + n_f2 * float(bl_table[month, 1])
            + n_f3 * float(bl_table[month, 2])
        )
    return total_wh / 1000.0


# Cached annual energy of the default ARERA baseline (≈ 1196 kWh/year). Used as
# the reference ``B`` in the bill fit so it is computed only once at import.
ARERA_BASELINE_ANNUAL_KWH: float = compute_arera_baseline_annual_kwh()


def annual_kwh_from_bimonthly(bimonthly_kwh: Sequence[float]) -> float:
    """
    Sum six bimonthly bill readings into an annual total.

    Args:
        bimonthly_kwh: Exactly 6 consumption readings (kWh), one per two-month
            Italian billing period.

    Returns:
        float: Annual consumption in kWh.

    Raises:
        ValueError: If the sequence does not have exactly 6 entries, or any
            value is negative.

    Example:
        ```python
        annual_kwh_from_bimonthly([400, 350, 300, 280, 320, 450])  # 2100.0
        ```
    """
    if len(bimonthly_kwh) != BIMONTHLY_PERIODS_PER_YEAR:
        raise ValueError(
            f"bimonthly_kwh must have exactly {BIMONTHLY_PERIODS_PER_YEAR} "
            f"entries, got {len(bimonthly_kwh)}"
        )
    if any(v < 0 for v in bimonthly_kwh):
        raise ValueError("bimonthly_kwh values must be non-negative")
    return float(sum(bimonthly_kwh))


def build_scaled_arera_factory(
    scale_factor: float,
) -> Callable[[], LoadProfile]:
    """
    Build a zero-arg factory producing a scaled ARERA load profile.

    The returned callable yields a fresh ``AreraLoadProfile`` whose base-load
    table is :data:`BL_TABLE` multiplied by ``scale_factor`` — the home regime
    of a bill-fitted profile. A factory (rather than a single instance) is
    returned because the blueprint/Monte-Carlo machinery expects to mint a new
    profile per build.

    Args:
        scale_factor: Multiplier on the ARERA base-load table (> 0). ``1.0``
            reproduces the standard baseline; ``2.0`` doubles every hour's
            load.

    Returns:
        Callable[[], LoadProfile]: Factory returning a scaled
        :class:`AreraLoadProfile`.

    Raises:
        ValueError: If ``scale_factor <= 0``.

    Example:
        ```python
        factory = build_scaled_arera_factory(1.8)
        profile = factory()
        # profile.get_hourly_load_kw(...) is 1.8× the default ARERA value
        ```
    """
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be > 0, got {scale_factor}")
    scaled_table = BL_TABLE * float(scale_factor)
    return lambda table=scaled_table: AreraLoadProfile(bl_table=table)


def fit_bolletta_profile(
    target_annual_kwh: float,
    presence_calendar: PresenceCalendar,
    house_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fit a home/away ARERA profile to an annual bill + a presence calendar.

    Solves for the single ``home_scale_factor`` that makes the profile's
    expected annual energy equal ``target_annual_kwh`` (see the module
    docstring for the derivation). The away regime is the unscaled ARERA
    baseline; the home regime is ARERA scaled by the returned factor. The
    presence calendar supplies both the occupancy fraction ``f`` (the
    home/away energy split) and the per-month ``min/max`` day counts.

    Args:
        target_annual_kwh: Desired whole-year consumption (kWh, > 0), e.g.
            taken straight from the bill.
        presence_calendar: Occupancy description of the building. Its
            :meth:`PresenceCalendar.annual_presence_fraction` drives the split.
        house_type: Optional :data:`HOUSE_TYPE_PRESETS` key. Advisory metadata
            only (echoed back for the UI); it does not affect the fit.

    Returns:
        dict: Fit result with keys:

        - ``home_scale_factor`` (float): multiplier on the ARERA table for the
          home regime (≥ :data:`MIN_HOME_SCALE_FACTOR`).
        - ``estimated_home_kwh`` (float): annual energy attributed to home days.
        - ``estimated_away_kwh`` (float): annual energy attributed to away days.
        - ``annual_presence_fraction`` (float): the occupancy fraction used.
        - ``min_days_home`` / ``max_days_home`` (list[int]): 12-month occupancy
          bounds derived from the calendar.
        - ``derived_profile_data`` (dict): a ready-to-save load-profile ``data``
          block (``input_level="bolletta"`` + presence calendar + bill echo +
          a ``_derived`` block) that the scenario builder and preview pipeline
          understand directly.

    Raises:
        ValueError: If ``target_annual_kwh <= 0``.

    Example:
        ```python
        from sim_stochastic_pv.simulation.load_profiles import (
            fit_bolletta_profile, DEFAULT_PRESENCE_CALENDAR,
        )

        fit = fit_bolletta_profile(2400.0, DEFAULT_PRESENCE_CALENDAR)
        fit["home_scale_factor"]  # ~2.4 (home days well above standby)
        round(fit["estimated_home_kwh"] + fit["estimated_away_kwh"])  # ≈ 2400
        ```

    Notes:
        - When ``target_annual_kwh`` is below the away baseline's own
          contribution (an implausibly low bill for the declared presence) the
          scale is floored at :data:`MIN_HOME_SCALE_FACTOR` and the reported
          energies will no longer sum to the target — the UI surfaces that.
        - The fit matches the *expected* annual energy; individual Monte Carlo
          paths vary around it via the home-day sampling.
    """
    if target_annual_kwh <= 0:
        raise ValueError(
            f"target_annual_kwh must be > 0, got {target_annual_kwh}"
        )

    base = ARERA_BASELINE_ANNUAL_KWH
    fraction = presence_calendar.annual_presence_fraction()
    away_kwh = base * (1.0 - fraction)

    if fraction <= MIN_PRESENCE_FRACTION:
        # Building effectively never occupied: no home days exist to scale.
        home_scale = 1.0
        estimated_home_kwh = 0.0
        estimated_away_kwh = base
    else:
        home_scale = (target_annual_kwh - away_kwh) / (base * fraction)
        home_scale = max(home_scale, MIN_HOME_SCALE_FACTOR)
        estimated_home_kwh = fraction * home_scale * base
        estimated_away_kwh = away_kwh

    min_days, max_days = presence_calendar.to_min_max_days_home()

    derived_profile_data: Dict[str, Any] = {
        # ``kind`` lets the full-scenario builder route this block (the scenario
        # only sees ``data``); ``input_level`` records which editor produced it.
        "kind": "bolletta",
        "input_level": "bolletta",
        "presence_calendar": presence_calendar.to_dict(),
        "bolletta": {
            "annual_kwh": float(target_annual_kwh),
            "house_type": house_type,
        },
        "_derived": {
            "home_scale_factor": float(home_scale),
            "estimated_home_kwh": float(estimated_home_kwh),
            "estimated_away_kwh": float(estimated_away_kwh),
            "annual_presence_fraction": float(fraction),
        },
    }

    return {
        "home_scale_factor": float(home_scale),
        "estimated_home_kwh": float(estimated_home_kwh),
        "estimated_away_kwh": float(estimated_away_kwh),
        "annual_presence_fraction": float(fraction),
        "min_days_home": min_days,
        "max_days_home": max_days,
        "derived_profile_data": derived_profile_data,
    }
