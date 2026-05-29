"""
Phase 17-bis — Event-based discrete appliance load decorator.

Phase 17 added intra-day variability (LogN/AR(1) multiplier) and HVAC
additive load to the deterministic baseline profile. What it could
*not* capture, by construction, is the **bimodal kW signature** of
real household consumption: discrete appliances turning on and off
(washing machine, oven, dishwasher, EV charger). The Itō-corrected
multiplier smooths everything; it cannot produce the spike-and-fall
pattern that drives PV self-consumption and battery sizing.

This module provides the missing piece — a Poisson-scheduled
**event-based** load profile that adds rectangular power pulses on
top of the (already stochastic) baseline:

    total_load(h) = base_load(h) * eps(h)        # Phase 17 multiplier
                  + appliance_load(h)            # this module
                  + hvac_load(h)                 # Phase 17 HVAC

The multiplicative-first/additive-second ordering matters: applying
the LogN multiplier to a 1.5 kW washing-machine event would produce
1.275 kW or 1.7 kW — an appliance turns ON or OFF, it does not modulate.
The events therefore live *outside* the stochastic decorator.

Out of scope: demand-response price-responsive scheduling, weekday vs
weekend frequency differentiation, internal appliance thermal ramps,
inter-appliance coordination (contract-kW clipping), EV V2H. The
``smart_pv`` mode supplied here biases event start times toward the
solar production hours via the deterministic hourly shape of
:class:`~sim_stochastic_pv.simulation.solar.SolarModel`; it is the
simplest meaningful "intelligence" the user can opt into to test the
economic value of timer-based scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np

from .base import LoadProfile


# ---------------------------------------------------------------------------
# Single appliance specification
# ---------------------------------------------------------------------------

_HOUR_OF_DAY: tuple[int, ...] = tuple(range(24))


@dataclass(frozen=True)
class ApplianceEvent:
    """
    Discrete appliance event specification.

    Models one *type* of appliance — the per-event timing is the
    responsibility of :class:`EventBasedApplianceProfile`, which draws
    a Poisson number of events per month from ``monthly_frequency`` and
    picks each event's start hour according to ``allowed_hours`` and
    ``hour_of_day_weights``.

    Attributes:
        name: Human-readable identifier (also the key in the per-name
            KPI dict). Two events with the same name in the same
            scenario will be aggregated by the simulator's KPI layer
            into a single entry — keep names unique when in doubt.
        p_kw: Instantaneous power draw of the appliance while running
            (kW). Real appliances vary a bit during the cycle (a
            washing machine peaks at the heater stage and drops to ~0
            during spin); we use the *cycle-average* electrical power
            so the energy-per-cycle works out correctly.
        duration_hours: Cycle duration (h). Need not be an integer:
            0.75 h for a small oven, 8.0 h for slow EV charging. The
            decorator handles fractional last-hour energy accounting
            so the total energy per event is exactly ``p_kw *
            duration_hours``.
        monthly_frequency: Expected number of events per calendar month,
            one float per month (Jan..Dec). The decorator draws the
            actual count from ``Poisson(monthly_frequency[m])`` for
            each (month, year) pair so the long-run mean matches the
            input while still yielding path-by-path variability.
        allowed_hours: Subset of ``range(24)`` of hours where the event
            is *allowed* to start. The hour of day must already be a
            sensible start time: an 8 h EV charge starting at 22:00 is
            fine even though it continues past midnight.
        hour_of_day_weights: 24-length non-negative array of soft
            preferences across hours. Multiplied element-wise by the
            ``allowed_hours`` mask before being normalised. Default
            ``None`` → uniform within the allowed window.
        schedule_mode: ``"naive_timer"`` (default) uses
            ``hour_of_day_weights`` as-is; ``"smart_pv"`` reweights the
            start-hour probability by the deterministic solar hourly
            shape (peak at noon), biasing events toward the sun. The
            user opts in per appliance — the EV charger may run
            "smart_pv" while the oven stays "naive_timer".

    Notes:
        - All fields are *constants for a scenario*. The
          per-path stochasticity comes entirely from the Poisson event
          count and the start-hour draw — the appliance itself has no
          memory across events.
        - The dataclass is frozen so it can be hashed and shared
          safely across paths.
    """

    name: str
    p_kw: float
    duration_hours: float
    monthly_frequency: tuple[float, ...]
    allowed_hours: tuple[int, ...]
    hour_of_day_weights: tuple[float, ...] | None = None
    schedule_mode: Literal["naive_timer", "smart_pv"] = "naive_timer"

    def __post_init__(self) -> None:
        if self.p_kw <= 0:
            raise ValueError(f"p_kw must be > 0, got {self.p_kw}")
        if self.duration_hours <= 0:
            raise ValueError(
                f"duration_hours must be > 0, got {self.duration_hours}"
            )
        if len(self.monthly_frequency) != 12:
            raise ValueError(
                "monthly_frequency must be a length-12 sequence (one per "
                f"calendar month), got length {len(self.monthly_frequency)}"
            )
        if any(f < 0 for f in self.monthly_frequency):
            raise ValueError("monthly_frequency entries must be >= 0")
        if not self.allowed_hours:
            raise ValueError("allowed_hours must be a non-empty subset of 0..23")
        for h in self.allowed_hours:
            if not (0 <= int(h) < 24):
                raise ValueError(
                    f"allowed_hours entries must be in [0, 23], got {h}"
                )
        if self.hour_of_day_weights is not None:
            if len(self.hour_of_day_weights) != 24:
                raise ValueError(
                    "hour_of_day_weights must have length 24, got "
                    f"{len(self.hour_of_day_weights)}"
                )
            if any(w < 0 for w in self.hour_of_day_weights):
                raise ValueError("hour_of_day_weights must be non-negative")
        if self.schedule_mode not in ("naive_timer", "smart_pv"):
            raise ValueError(
                "schedule_mode must be 'naive_timer' or 'smart_pv', "
                f"got {self.schedule_mode!r}"
            )

    def expected_kwh_annual(self) -> float:
        """
        Analytic expected energy consumption (kWh/year) per appliance.

        Returns:
            Sum of monthly frequencies times power times duration.
            Useful as a UI-side predictive readout: the wizard can
            show "you should expect ≈ N kWh/year for this appliance"
            before launching the simulator. Matches the long-run mean
            of the Monte Carlo sampler within the law of large numbers.
        """
        return float(sum(self.monthly_frequency)) * self.p_kw * self.duration_hours


# ---------------------------------------------------------------------------
# Catalog of realistic Italian residential appliances
# ---------------------------------------------------------------------------


def _flat_freq(events_per_month: float) -> tuple[float, ...]:
    """Convenience: build a 12-tuple of identical monthly frequencies."""
    return tuple([float(events_per_month)] * 12)


def _hours_range(start: int, end: int) -> tuple[int, ...]:
    """Hours in ``[start, end)`` modulo 24 (handles wrap-around midnight)."""
    if start <= end:
        return tuple(range(start, end))
    return tuple(list(range(start, 24)) + list(range(0, end)))


#: Pre-built realistic appliance specifications for Italian residential
#: users. Calibration from ISTAT energy-consumption surveys + RSE
#: residential appliance studies. The user references these by string
#: in the scenario JSON; the lookup happens in :class:`ApplianceCatalog`.
APPLIANCE_PRESETS: dict[str, ApplianceEvent] = {
    "washing_machine": ApplianceEvent(
        name="washing_machine",
        p_kw=1.5,
        duration_hours=1.5,
        monthly_frequency=_flat_freq(12.0),
        allowed_hours=_hours_range(9, 18),
    ),
    "dishwasher": ApplianceEvent(
        name="dishwasher",
        p_kw=1.2,
        duration_hours=1.0,
        monthly_frequency=_flat_freq(15.0),
        allowed_hours=_hours_range(13, 22),
    ),
    "oven": ApplianceEvent(
        name="oven",
        p_kw=2.5,
        duration_hours=0.75,
        monthly_frequency=_flat_freq(8.0),
        allowed_hours=tuple([11, 12, 18, 19]),
    ),
    "dryer": ApplianceEvent(
        name="dryer",
        p_kw=2.2,
        duration_hours=1.0,
        monthly_frequency=_flat_freq(6.0),
        allowed_hours=_hours_range(10, 14),
    ),
    "ev_charger_slow": ApplianceEvent(
        name="ev_charger_slow",
        p_kw=2.3,
        duration_hours=8.0,
        monthly_frequency=_flat_freq(20.0),
        allowed_hours=_hours_range(22, 6),  # 22..23, 0..5 (overnight)
    ),
    "ev_charger_fast": ApplianceEvent(
        name="ev_charger_fast",
        p_kw=7.4,
        duration_hours=2.5,
        monthly_frequency=_flat_freq(15.0),
        allowed_hours=_hours_range(22, 6),
    ),
    "induction_cooktop": ApplianceEvent(
        name="induction_cooktop",
        p_kw=1.8,
        duration_hours=0.5,
        monthly_frequency=_flat_freq(30.0),
        allowed_hours=tuple([11, 12, 19, 20]),
    ),
    "dhw_heat_pump_cycle": ApplianceEvent(
        name="dhw_heat_pump_cycle",
        p_kw=1.8,
        duration_hours=0.5,
        monthly_frequency=_flat_freq(30.0),
        allowed_hours=tuple([7, 8, 17, 18]),
    ),
}


def get_preset(name: str) -> ApplianceEvent:
    """
    Resolve a preset name to its :class:`ApplianceEvent`.

    Args:
        name: Catalog key (case-insensitive whitespace-trimmed).

    Returns:
        Fresh :class:`ApplianceEvent` instance (frozen — safe to share).

    Raises:
        KeyError: When the name is not in :data:`APPLIANCE_PRESETS`.
    """
    key = name.strip().lower()
    if key not in APPLIANCE_PRESETS:
        raise KeyError(
            f"Appliance preset {name!r} not found. "
            f"Available presets: {sorted(APPLIANCE_PRESETS)}"
        )
    return APPLIANCE_PRESETS[key]


# ---------------------------------------------------------------------------
# Scenario-level configuration + KPI containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApplianceProfileConfig:
    """
    Container for the ``load_profile.appliances`` scenario block.

    Attributes:
        enabled: Toggle. When False the simulator does not build the
            decorator — the legacy load path stays untouched.
        smart_pv_default: Global default for ``schedule_mode`` —
            applied to every appliance that does NOT specify its own
            ``schedule_mode`` explicitly. The "user-friendly" UI knob.
        appliances: List of fully-resolved :class:`ApplianceEvent`
            instances (the scenario_builder has already merged preset
            data with any per-item override).

    Notes:
        Smart-PV evaluation is per appliance; the global
        ``smart_pv_default`` only changes the *default* picked by the
        builder when no explicit ``schedule_mode`` is provided. Once
        the dataclass is built, each appliance's mode is fixed.
    """

    enabled: bool = False
    smart_pv_default: bool = False
    appliances: tuple[ApplianceEvent, ...] = field(default_factory=tuple)


@dataclass
class AppliancesKPIs:
    """
    Per-path KPI container for the appliances feature.

    All counters are normalised so the Monte Carlo aggregator can take
    arithmetic means across paths.

    Attributes:
        total_appliance_kwh_annual: Total electrical energy drawn by
            appliances over the year (kWh/yr). Sum across all
            appliances.
        appliance_kwh_annual_by_name: Per-appliance breakdown for the
            Dashboard bar chart. Keys are
            :attr:`ApplianceEvent.name`.
        peak_simultaneous_kw: Maximum total appliance kW seen in any
            single hour of the path — used for sanity-checking the
            inverter / contract-kW dimensioning.
        share_of_total_load_pct: Fraction (0..100) of the total
            household load attributable to appliances. Finalised by
            the simulator after the path totals are known.
        smart_pv_self_consumption_pct: Among the *appliance* energy
            specifically (not the whole household), fraction (0..100)
            that fell under the hourly PV production curve of the
            path. ``0`` when there is no PV system or when no
            appliance ran in ``smart_pv`` mode. KPI of scheduling
            effectiveness.
    """

    total_appliance_kwh_annual: float = 0.0
    appliance_kwh_annual_by_name: dict[str, float] = field(default_factory=dict)
    peak_simultaneous_kw: float = 0.0
    share_of_total_load_pct: float = 0.0
    smart_pv_self_consumption_pct: float = 0.0


# ---------------------------------------------------------------------------
# Decorator: schedules events + serves hourly kW
# ---------------------------------------------------------------------------


class EventBasedApplianceProfile(LoadProfile):
    """
    Additive event-based appliance load.

    Conforms to the :class:`LoadProfile` interface so it can be queried
    by the simulator inside the hourly loop. The returned kW is the
    **additional** load due to appliances *only*; the simulator adds it
    on top of the baseline (stochastic-decorated) load.

    The decorator pre-computes the entire per-path hourly contribution
    in :meth:`reset_for_run`, so :meth:`get_hourly_load_kw` is O(1).

    Attributes:
        appliances: Tuple of :class:`ApplianceEvent` to schedule.
        solar_hourly_shape: Optional 24-length non-negative array
            (sums to 1) representing the deterministic solar production
            shape. Required when any appliance is in ``"smart_pv"``
            mode; ignored otherwise. The simulator passes
            :attr:`~sim_stochastic_pv.simulation.solar.SolarModel.hourly_shape`.
        _hourly_kw: Pre-computed total appliance kW per hour. Shape
            ``(n_years * 12 * 30 * 24,)`` aligned to the mock calendar
            used by :func:`~sim_stochastic_pv.calendar_utils.build_calendar`
            (12 months × 30 days). ``None`` until
            :meth:`reset_for_run` is called.
        _kwh_by_name: Pre-computed kWh per appliance name (used to
            populate the KPIs without re-scanning the hourly array).
    """

    def __init__(
        self,
        appliances: Sequence[ApplianceEvent],
        solar_hourly_shape: np.ndarray | None = None,
    ) -> None:
        self.appliances = tuple(appliances)
        if solar_hourly_shape is not None:
            arr = np.asarray(solar_hourly_shape, dtype=float)
            if arr.shape != (24,):
                raise ValueError(
                    "solar_hourly_shape must have shape (24,), got "
                    f"{arr.shape}"
                )
            if (arr < 0).any():
                raise ValueError("solar_hourly_shape must be non-negative")
            self.solar_hourly_shape = arr
        else:
            self.solar_hourly_shape = None
        # Any smart_pv appliance requires the solar shape — fail loud.
        if any(a.schedule_mode == "smart_pv" for a in self.appliances):
            if self.solar_hourly_shape is None:
                raise ValueError(
                    "EventBasedApplianceProfile requires solar_hourly_shape "
                    "when at least one appliance has schedule_mode='smart_pv'."
                )
        self._hourly_kw: np.ndarray | None = None
        self._n_hours: int = 0
        self._kwh_by_name: dict[str, float] = {}

    # ------------------------------------------------------------------
    # LoadProfile interface
    # ------------------------------------------------------------------

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Sample the full event schedule for the path and rasterise it.

        The simulator always passes ``rng`` and ``n_years``. Each call
        clears any prior state and starts fresh, so the decorator is
        safe to reuse across paths.
        """
        if rng is None:
            rng = np.random.default_rng()
        years = n_years if n_years is not None else 1
        # The simulator's `build_calendar` uses 30-day months and 12-month
        # years — mirror that calendar so the hourly index returned by
        # ``get_hourly_load_kw`` lines up byte-for-byte.
        self._n_hours = int(years * 12 * 30 * 24)
        self._hourly_kw = np.zeros(self._n_hours, dtype=float)
        self._kwh_by_name = {a.name: 0.0 for a in self.appliances}

        for appliance in self.appliances:
            weights = self._effective_start_weights(appliance)
            for y in range(years):
                for m in range(12):
                    self._schedule_appliance_month(
                        appliance, y, m, weights, rng
                    )

    def get_hourly_load_kw(
        self,
        year_index: int,
        month_in_year: int,
        day_in_month: int,
        hour_in_day: int,
        weekday: int,
    ) -> float:
        """
        Return the additive appliance contribution for the given hour.

        The hour is looked up in the pre-rasterised path. When called
        before :meth:`reset_for_run` the result is 0.0 — the legacy
        baseline survives even if the integration is partial.
        """
        if self._hourly_kw is None:
            return 0.0
        idx = (
            year_index * (12 * 30 * 24)
            + month_in_year * (30 * 24)
            + day_in_month * 24
            + hour_in_day
        )
        if idx < 0 or idx >= self._n_hours:
            return 0.0
        return float(self._hourly_kw[idx])

    # ------------------------------------------------------------------
    # KPI helpers (called by EnergySystemSimulator after the path runs)
    # ------------------------------------------------------------------

    def hourly_array(self) -> np.ndarray:
        """
        Return the path's full hourly appliance kW array.

        Returns:
            A copy so callers cannot mutate the simulator's state.
            Empty array when the decorator hasn't been reset yet.
        """
        if self._hourly_kw is None:
            return np.zeros(0, dtype=float)
        return self._hourly_kw.copy()

    def kpis_for_path(
        self,
        n_years: int,
        pv_hourly_kw: np.ndarray | None,
    ) -> AppliancesKPIs:
        """
        Build the per-path KPI bundle.

        Args:
            n_years: Simulation horizon (years), used to normalise
                kWh totals to per-year averages.
            pv_hourly_kw: Optional hourly PV DC production array for
                the path (kW). When provided, the function computes
                ``smart_pv_self_consumption_pct`` as the fraction of
                appliance energy that fell within PV production.

        Returns:
            Fresh :class:`AppliancesKPIs` instance.
        """
        if self._hourly_kw is None or self._hourly_kw.size == 0:
            return AppliancesKPIs()
        n_years = max(1, int(n_years))
        total_kwh = float(self._hourly_kw.sum())
        kpis = AppliancesKPIs(
            total_appliance_kwh_annual=total_kwh / n_years,
            appliance_kwh_annual_by_name={
                name: kwh / n_years for name, kwh in self._kwh_by_name.items()
            },
            peak_simultaneous_kw=float(self._hourly_kw.max()),
        )
        if (
            pv_hourly_kw is not None
            and pv_hourly_kw.size == self._hourly_kw.size
            and total_kwh > 0
        ):
            # Smart-PV self-consumption = appliance energy that fell
            # within an hour with non-zero PV production, weighted by
            # min(appliance, PV). Pure energy match, not optimisation.
            covered = np.minimum(self._hourly_kw, pv_hourly_kw)
            kpis.smart_pv_self_consumption_pct = (
                100.0 * float(covered.sum()) / total_kwh
            )
        return kpis

    # ------------------------------------------------------------------
    # Internal scheduling
    # ------------------------------------------------------------------

    def _effective_start_weights(self, appliance: ApplianceEvent) -> np.ndarray:
        """
        Compute the 24-length start-hour weights for ``appliance``.

        Combines (a) the user-provided ``hour_of_day_weights`` (or
        uniform when ``None``), (b) the ``allowed_hours`` mask, and
        (c) the solar reweighting in ``"smart_pv"`` mode. Returns a
        non-normalised array; the caller draws via
        :meth:`numpy.random.Generator.choice` which normalises
        internally.
        """
        weights = np.zeros(24, dtype=float)
        for h in appliance.allowed_hours:
            weights[h] = 1.0
        if appliance.hour_of_day_weights is not None:
            weights *= np.asarray(appliance.hour_of_day_weights, dtype=float)
        if appliance.schedule_mode == "smart_pv":
            assert self.solar_hourly_shape is not None  # checked in __init__
            weights *= self.solar_hourly_shape
        return weights

    def _schedule_appliance_month(
        self,
        appliance: ApplianceEvent,
        year_index: int,
        month_in_year: int,
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """
        Schedule one (appliance, month, year) chunk of events.

        Draws the event count from a Poisson distribution, picks each
        event's day (uniform over the mock 30-day month) and start
        hour (categorical with the given weights), then rasterises the
        rectangular pulse into ``self._hourly_kw`` with fractional
        last-hour accounting to preserve total energy.
        """
        lam = float(appliance.monthly_frequency[month_in_year])
        if lam <= 0:
            return
        n_events = int(rng.poisson(lam))
        if n_events <= 0:
            return
        total_weight = float(weights.sum())
        if total_weight <= 0:
            return
        probs = weights / total_weight
        start_hours = rng.choice(24, size=n_events, replace=True, p=probs)
        days = rng.integers(0, 30, size=n_events)
        duration = float(appliance.duration_hours)
        full_hours = int(np.floor(duration))
        remainder = duration - full_hours
        p_kw = float(appliance.p_kw)
        year_offset = year_index * (12 * 30 * 24)
        month_offset = month_in_year * (30 * 24)
        for d, sh in zip(days, start_hours):
            base_idx = year_offset + month_offset + int(d) * 24 + int(sh)
            # Full-power hours.
            for k in range(full_hours):
                idx = base_idx + k
                if 0 <= idx < self._n_hours:
                    self._hourly_kw[idx] += p_kw
                    self._kwh_by_name[appliance.name] += p_kw
            # Fractional remainder hour (only if duration is non-integer).
            if remainder > 0:
                idx = base_idx + full_hours
                if 0 <= idx < self._n_hours:
                    contribution = p_kw * remainder
                    self._hourly_kw[idx] += contribution
                    self._kwh_by_name[appliance.name] += contribution


# ---------------------------------------------------------------------------
# Aggregation across MC paths (called by application._build_*_summary)
# ---------------------------------------------------------------------------


def aggregate_appliances_kpis(per_path: Sequence[AppliancesKPIs]) -> dict:
    """
    Aggregate per-path :class:`AppliancesKPIs` into a summary dict.

    All means are arithmetic. The per-name dict gets the mean of each
    appliance's kWh/yr across the paths; appliances that don't appear
    in a path contribute 0 to that path's term.

    Returns:
        Dict ready to be serialised inside ``summary["appliances"]``.
        Empty input → all-zero structure with no per-name breakdown.
    """
    if not per_path:
        return {
            "total_appliance_kwh_annual_mean": 0.0,
            "appliance_kwh_annual_by_name_mean": {},
            "peak_simultaneous_kw_mean": 0.0,
            "share_of_total_load_pct_mean": 0.0,
            "smart_pv_self_consumption_pct_mean": 0.0,
        }
    total = float(np.mean([k.total_appliance_kwh_annual for k in per_path]))
    peak = float(np.mean([k.peak_simultaneous_kw for k in per_path]))
    share = float(np.mean([k.share_of_total_load_pct for k in per_path]))
    spv = float(np.mean([k.smart_pv_self_consumption_pct for k in per_path]))
    # Per-name: union of all names seen, then mean treating missing as 0.
    names: set[str] = set()
    for k in per_path:
        names.update(k.appliance_kwh_annual_by_name.keys())
    by_name = {
        name: float(
            np.mean([k.appliance_kwh_annual_by_name.get(name, 0.0) for k in per_path])
        )
        for name in sorted(names)
    }
    return {
        "total_appliance_kwh_annual_mean": total,
        "appliance_kwh_annual_by_name_mean": by_name,
        "peak_simultaneous_kw_mean": peak,
        "share_of_total_load_pct_mean": share,
        "smart_pv_self_consumption_pct_mean": spv,
    }
