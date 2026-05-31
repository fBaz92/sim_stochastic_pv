"""
Reliability models for transmission assets (interconnections, lines, cables).

This module generates stochastic availability paths ``a(t) ∈ [0, 1]`` that
multiplicatively derate a nominal capacity. It is used by the
interconnection layer to simulate random forced outages, but is written
generically and can be reused for any asset with a two-state (up/down)
reliability profile.

Model
-----

The canonical repairable-system model is the **two-state alternating
renewal process** (up ↔ down), a continuous-time Markov chain:

- Time between failures (up duration) is exponential with rate
  ``λ = 1 / MTTF`` — memoryless fault arrivals.
- Repair durations are drawn from a **lognormal** distribution
  (not exponential) because empirical transmission-asset outage data
  exhibits a heavy right tail: most repairs are short (hours), a few
  drag on for months (submarine cable replacements, converter station
  rebuilds). An exponential repair time would badly underestimate tail
  risk.
- Faults can be **total** (link fully tripped) or **partial** (e.g. one
  pole of a bipolar HVDC tripped → 50% derate). Sampled per event.

The availability process is generated as a sequence of
(up-interval, down-interval) pairs until the simulation horizon is
reached. Overlapping faults cannot occur under this model (we do not
start a new fault until the current repair ends).

Three construction modes
------------------------

To support diverse user skill levels, :class:`TwoStateMarkovReliability`
can be built in three ways:

1. **Explicit**: the user specifies all parameters directly.
2. **By availability**: the user specifies target availability ``A`` and
   median MTTR; MTTF is derived from ``A = MTTF / (MTTF + MTTR_mean)``,
   accounting for the mean-vs-median conversion of the lognormal.
3. **By technology preset**: lookup in :data:`TECHNOLOGY_PRESETS` keyed
   by line type (``'overhead_ac'``, ``'hvdc_back_to_back'``, etc.).

The factory :func:`build_reliability_model` dispatches on a config dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sim_stochastic_pv.market.config import QUARTERS_PER_HOUR
from sim_stochastic_pv.market.grid import TimeGrid


# ── Protocol ──────────────────────────────────────────────────────────────


class ReliabilityModel(Protocol):
    """Structural interface for reliability models.

    Implementations must provide a single method that returns a multiplier
    path in ``[0, 1]`` with the same length as the time grid. A value of
    1.0 means fully available; 0.0 means fully derated (outage).
    """

    def sample_availability_path(
        self, time_grid: TimeGrid,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate one year of availability multipliers."""
        ...


# ── Implementations ───────────────────────────────────────────────────────


class PerfectReliability:
    """Null-model reliability: ``a(t) ≡ 1`` at every time step.

    Used when the global switch :data:`~sim_stochastic_pv.market.config.ENABLE_NTC_FAULTS`
    is ``False``, and for sensitivity runs isolating the price effect of
    faults from other sources of variability.
    """

    def sample_availability_path(self, time_grid: TimeGrid,
                                  rng: np.random.Generator) -> np.ndarray:
        """Return an array of ones.

        Args:
            time_grid: Temporal backbone (used only for its length).
            rng: Unused. Accepted for interface compatibility.

        Returns:
            np.ndarray: Ones array of shape ``(time_grid.n,)``.
        """
        return np.ones(time_grid.n)


@dataclass
class TwoStateMarkovReliability:
    """Two-state Markov reliability with exponential MTTF and lognormal MTTR.

    Implements the alternating renewal process described in the module
    docstring. Each generated availability path is a single Monte Carlo
    realization; aggregating many runs recovers the stationary availability.

    Attributes:
        mttf_hours (float): Mean time to failure (hours). Controls fault
            arrival rate: ``λ = 1 / mttf_hours``.
        mttr_median_hours (float): Median time to repair (hours). Note: for
            a lognormal this is ``exp(μ_log)``; the *mean* repair time is
            ``median · exp(σ_log² / 2)``, larger than the median when
            ``σ_log > 0``.
        mttr_sigma_log (float): Shape parameter of the lognormal repair
            distribution. Larger = heavier tail. Typical values: 0.8-1.0
            for overhead AC lines, 1.3-1.8 for submarine cables.
        total_fault_probability (float): Probability that a fault is total
            (derate to 0). With complement probability, the fault is
            partial and derates to :attr:`partial_derate_factor`.
        partial_derate_factor (float): Availability multiplier during a
            partial fault (e.g. 0.5 for loss of one pole in a bipolar
            HVDC link).
    """

    mttf_hours: float
    mttr_median_hours: float
    mttr_sigma_log: float = 1.0
    total_fault_probability: float = 0.6
    partial_derate_factor: float = 0.5

    def __post_init__(self) -> None:
        """Validate parameter ranges."""
        if self.mttf_hours <= 0:
            raise ValueError(f"mttf_hours must be > 0, got {self.mttf_hours}")
        if self.mttr_median_hours <= 0:
            raise ValueError(
                f"mttr_median_hours must be > 0, got {self.mttr_median_hours}")
        if self.mttr_sigma_log < 0:
            raise ValueError(
                f"mttr_sigma_log must be >= 0, got {self.mttr_sigma_log}")
        if not 0.0 <= self.total_fault_probability <= 1.0:
            raise ValueError(
                "total_fault_probability must be in [0,1], "
                f"got {self.total_fault_probability}")
        if not 0.0 <= self.partial_derate_factor <= 1.0:
            raise ValueError(
                "partial_derate_factor must be in [0,1], "
                f"got {self.partial_derate_factor}")

    # ── Alternative constructors ──

    @classmethod
    def from_availability(
        cls,
        availability: float,
        mttr_median_hours: float,
        mttr_sigma_log: float = 1.0,
        total_fault_probability: float = 0.6,
        partial_derate_factor: float = 0.5,
    ) -> "TwoStateMarkovReliability":
        """Build a model from target availability and median MTTR.

        Derives MTTF from the stationary-availability relation
        ``A = MTTF / (MTTF + MTTR_mean)``, giving
        ``MTTF = MTTR_mean · A / (1 - A)``. The conversion from median
        to mean of the lognormal uses ``mean = median · exp(σ² / 2)``,
        which matters for heavy-tailed distributions: with
        ``σ_log = 1.5`` the mean is ``3.1×`` the median.

        Args:
            availability: Target steady-state availability, in ``(0, 1)``.
                Typical values: 0.99 for overhead AC, 0.96 for submarine.
            mttr_median_hours: Median repair time (hours).
            mttr_sigma_log: Lognormal shape parameter. Defaults to 1.0.
            total_fault_probability: Forwarded to the constructor.
            partial_derate_factor: Forwarded to the constructor.

        Returns:
            TwoStateMarkovReliability: Instance with derived ``mttf_hours``.

        Raises:
            ValueError: If ``availability`` is not in ``(0, 1)``.
        """
        if not 0.0 < availability < 1.0:
            raise ValueError(
                f"availability must be in (0, 1), got {availability}")
        mttr_mean = mttr_median_hours * np.exp(mttr_sigma_log ** 2 / 2.0)
        mttf = mttr_mean * availability / (1.0 - availability)
        return cls(
            mttf_hours=mttf,
            mttr_median_hours=mttr_median_hours,
            mttr_sigma_log=mttr_sigma_log,
            total_fault_probability=total_fault_probability,
            partial_derate_factor=partial_derate_factor,
        )

    @classmethod
    def from_technology(cls, tech: str,
                        **overrides) -> "TwoStateMarkovReliability":
        """Build a model from a technology preset, with optional overrides.

        Looks up ``tech`` in :data:`TECHNOLOGY_PRESETS` and instantiates
        with those defaults. Any keyword arguments in ``overrides`` replace
        the preset values (useful to tweak a single parameter without
        copying the whole dict).

        Args:
            tech: Technology key. See :data:`TECHNOLOGY_PRESETS`.
            **overrides: Fields to override on top of the preset.

        Returns:
            TwoStateMarkovReliability: Configured instance.

        Raises:
            ValueError: If ``tech`` is not a known preset key.
        """
        if tech not in TECHNOLOGY_PRESETS:
            raise ValueError(
                f"Unknown technology preset '{tech}'. "
                f"Available: {sorted(TECHNOLOGY_PRESETS.keys())}")
        params = {**TECHNOLOGY_PRESETS[tech], **overrides}
        return cls(**params)

    # ── Path generation ──

    def sample_availability_path(self, time_grid: TimeGrid,
                                  rng: np.random.Generator) -> np.ndarray:
        """Simulate one year of availability via alternating renewal.

        Walks forward in time in hours:

        1. Sample next fault arrival: ``t += Exp(1/mttf_hours)``.
        2. Sample fault severity (total vs partial) and repair duration
           from the lognormal.
        3. Apply the derate to the affected quarter-hour slice.
        4. Advance past the repair and loop.

        The lognormal repair duration is parameterized so that
        ``exp(location) = median`` and ``shape = sigma_log``. Extremely
        long repairs are truncated at the simulation horizon.

        Args:
            time_grid: Temporal backbone (uses ``n`` quarter-hours).
            rng: NumPy random generator.

        Returns:
            np.ndarray: Availability path of shape ``(time_grid.n,)``
                with values in ``[0, 1]``.
        """
        T = time_grid.n
        dt_hours = 1.0 / QUARTERS_PER_HOUR          # 0.25 h
        horizon_hours = T * dt_hours

        a = np.ones(T)
        t_hours = 0.0
        mu_log = float(np.log(self.mttr_median_hours))

        while True:
            # 1. Next fault arrival
            t_hours += float(rng.exponential(scale=self.mttf_hours))
            if t_hours >= horizon_hours:
                break

            # 2. Repair duration (lognormal) and severity
            repair_hours = float(
                rng.lognormal(mean=mu_log, sigma=self.mttr_sigma_log))
            if rng.random() < self.total_fault_probability:
                derate = 0.0
            else:
                derate = float(self.partial_derate_factor)

            # 3. Apply to the affected quarter-hour slice
            start_idx = int(t_hours / dt_hours)
            end_idx = min(int((t_hours + repair_hours) / dt_hours), T)
            if end_idx > start_idx:
                # minimum handles rare overlapping faults (pre-existing derate)
                a[start_idx:end_idx] = np.minimum(a[start_idx:end_idx], derate)

            # 4. Advance past the repair
            t_hours += repair_hours

        return a


# ── Technology presets ────────────────────────────────────────────────────

TECHNOLOGY_PRESETS: dict[str, dict] = {
    'overhead_ac': {
        # Conventional AC overhead line. Frequent trips (weather, faults)
        # but fast isolation and repair.
        'mttf_hours': 8760 * 0.5,         # ~2 events/year
        'mttr_median_hours': 12.0,        # ~half a day
        'mttr_sigma_log': 0.8,
        'total_fault_probability': 0.7,
        'partial_derate_factor': 0.5,
    },
    'hvdc_back_to_back': {
        # Converter station on land, bipolar configuration.
        'mttf_hours': 8760 * 0.8,         # ~1.25 events/year
        'mttr_median_hours': 72.0,        # ~3 days median
        'mttr_sigma_log': 1.0,
        'total_fault_probability': 0.4,   # partial (one pole) is common
        'partial_derate_factor': 0.5,
    },
    'submarine_cable_hvdc': {
        # Submarine HVDC cable. Rare events but catastrophic MTTR —
        # ship campaigns, seasonal weather windows, spare-part lead times.
        'mttf_hours': 8760 * 2.0,         # 1 event per 2 years
        'mttr_median_hours': 120.0,       # 5 days median...
        'mttr_sigma_log': 1.5,            # ...but very heavy right tail
        'total_fault_probability': 0.6,
        'partial_derate_factor': 0.5,
    },
    'submarine_cable_ac': {
        # Submarine AC cable (less common, shorter distances).
        'mttf_hours': 8760 * 1.5,
        'mttr_median_hours': 168.0,       # ~1 week
        'mttr_sigma_log': 1.4,
        'total_fault_probability': 0.8,
        'partial_derate_factor': 0.5,
    },
}
"""Reliability parameter presets keyed by transmission technology.

Numbers are order-of-magnitude defaults derived from ENTSO-E outage
statistics and CIGRE reliability surveys; they should be overridden by
operator-specific data when available. The presets are intentionally
conservative (slightly pessimistic) for use in energy-security studies.

Keys (same meaning as :class:`TwoStateMarkovReliability` fields):
    mttf_hours, mttr_median_hours, mttr_sigma_log,
    total_fault_probability, partial_derate_factor.
"""


# ── Factory ───────────────────────────────────────────────────────────────


def build_reliability_model(cfg: dict) -> ReliabilityModel:
    """Build a :class:`ReliabilityModel` from a plain-dict specification.

    This is the single entry point used by the interconnection layer to
    resolve the ``reliability`` field of a config entry. The ``type`` key
    selects the construction mode; the remaining keys are forwarded as
    keyword arguments to the matching constructor.

    Supported ``type`` values:

    * ``'perfect'`` — :class:`PerfectReliability`. No other keys.
    * ``'explicit'`` — direct :class:`TwoStateMarkovReliability`
      construction. Required keys: ``mttf_hours``, ``mttr_median_hours``.
      Optional: ``mttr_sigma_log``, ``total_fault_probability``,
      ``partial_derate_factor``.
    * ``'availability'`` — :meth:`TwoStateMarkovReliability.from_availability`.
      Required keys: ``availability``, ``mttr_median_hours``. Optional:
      ``mttr_sigma_log``, ``total_fault_probability``,
      ``partial_derate_factor``.
    * ``'technology'`` — :meth:`TwoStateMarkovReliability.from_technology`.
      Required key: ``tech``. Any other keys are forwarded as overrides.

    Args:
        cfg: Configuration dict with a ``type`` key.

    Returns:
        ReliabilityModel: Configured instance ready for
            :meth:`sample_availability_path`.

    Raises:
        ValueError: If ``type`` is missing or unknown.
    """
    if 'type' not in cfg:
        raise ValueError(f"Reliability config missing 'type': {cfg}")
    params = {k: v for k, v in cfg.items() if k != 'type'}
    kind = cfg['type']

    if kind == 'perfect':
        if params:
            raise ValueError(
                f"'perfect' reliability takes no parameters, got {params}")
        return PerfectReliability()
    if kind == 'explicit':
        return TwoStateMarkovReliability(**params)
    if kind == 'availability':
        return TwoStateMarkovReliability.from_availability(**params)
    if kind == 'technology':
        return TwoStateMarkovReliability.from_technology(**params)
    raise ValueError(
        f"Unknown reliability type '{kind}'. "
        f"Expected one of: perfect, explicit, availability, technology.")
