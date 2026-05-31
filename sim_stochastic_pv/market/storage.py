"""
Battery storage (BESS) — utility-scale aggregate model.

Models a grid-scale battery with inter-temporal state (state of charge),
AC-side round-trip efficiency, self-discharge, and synthetic-inertia
capability. The battery is grid-following (its inverter does not provide
rotational inertia directly) but can be programmed to emulate an inertial
response — hence the :attr:`StorageUnit.h_synthetic` parameter.

The class is deliberately kept free of any dispatch policy: it only
exposes physical constraints (SOC evolution, power limits, inertia
contribution). The actual charge/discharge decision (e.g. the rolling
percentile arbitrage strategy) lives in
:mod:`~sim_stochastic_pv.market.dispatch`, so that alternative policies can be swapped
in without touching the physical model.

Units convention:
    - Power is in per-unit of :data:`~sim_stochastic_pv.market.config.P_BASE` everywhere.
    - SOC is a **fraction** in [0, 1] of :attr:`StorageUnit.energy_capacity_gwh`.
    - Self-discharge is expressed as a fraction lost per 24 h of idle
      storage; the equivalent per-quarter-hour rate is derived by
      96-compounding.
    - ``h_synthetic`` is the emulated inertia constant in seconds (so that
      a BESS with ``h_synthetic = 4`` contributes to the weighted-H sum
      like a turbine with ``H = 4`` and the same MVA rating).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim_stochastic_pv.market.config import P_PEAK_GW, QUARTERS_PER_DAY


@dataclass
class StorageUnit:
    """Aggregated utility-scale battery energy storage system (BESS).

    Physical model only: exposes SOC evolution, AC-side power limits, and
    synthetic-inertia contribution. The dispatch policy (when to charge,
    when to discharge) is orthogonal and lives in the dispatch engine.

    The round-trip efficiency is split **symmetrically** into
    ``sqrt(η_rt)`` on each leg, which is the industry-standard convention
    for AC-AC BESS energy accounting. With ``η_rt = 0.88`` this gives
    ``η_charge = η_discharge ≈ 0.938``: storing 1 MWh_AC consumes
    1.066 MWh_AC from the grid, and retrieving it delivers 0.938 MWh_AC
    back — net round-trip 0.88.

    Synthetic inertia is modelled as a *potential* contribution to the
    weighted system-H sum, gated by SOC: a BESS sitting at the extreme
    of its operational band (SOC ≤ SOC_min + margin, or
    SOC ≥ SOC_max - margin) cannot provide bidirectional response, so
    its contribution drops to zero for that quarter-hour. Inside the
    band the contribution is ``h_synthetic × power_capacity_pu``,
    independent of whether the unit happens to be charging, discharging,
    or idling at the current timestep — the inverter only needs headroom
    in MVA terms to respond to df/dt events, not to be actively
    exchanging power.

    Attributes:
        name: Human-readable identifier for plots and aggregates.
        energy_capacity_gwh: Nameplate energy capacity (GWh, electrical
            AC-side). Effective usable capacity is
            ``(soc_max_frac - soc_min_frac) × energy_capacity_gwh``.
        power_capacity_gw: Nameplate charge/discharge power (GW, AC-side
            at the point of common coupling).
        efficiency_roundtrip: Round-trip AC-AC efficiency in ``(0, 1]``.
            Split symmetrically into :attr:`eta_charge` and
            :attr:`eta_discharge`.
        soc_min_frac: Minimum operational SOC as a fraction of
            :attr:`energy_capacity_gwh`. Typically 0.05–0.1 to avoid
            accelerated ageing at deep discharge.
        soc_max_frac: Maximum operational SOC as a fraction. Typically
            0.9–0.95 to avoid accelerated ageing at full charge.
        initial_soc_frac: Starting SOC at ``t = 0`` of each simulated
            year. Must lie inside the operational band.
        self_discharge_per_day: Energy fraction lost per 24 h of idle
            storage (leakage + auxiliaries + BMS consumption). Applied
            compoundly at the quarter-hour resolution.
        h_synthetic: Emulated inertia constant in seconds. Typical BESS
            range 2–6 s; set to 0 to disable synthetic-inertia support
            (pure grid-following inverter).
        inertia_soc_margin: Extra headroom above SOC_min and below
            SOC_max required for the unit to provide synthetic inertia.
            Outside this band the inverter cannot respond in both
            directions, so the contribution is zeroed. Default 0.02
            (2 % of nameplate capacity).
    """

    name: str
    energy_capacity_gwh: float
    power_capacity_gw: float
    efficiency_roundtrip: float = 0.88
    soc_min_frac: float = 0.1
    soc_max_frac: float = 0.9
    initial_soc_frac: float = 0.5
    self_discharge_per_day: float = 0.001
    h_synthetic: float = 4.0
    inertia_soc_margin: float = 0.02

    def __post_init__(self) -> None:
        """Validate the parameter set at construction time.

        Raises:
            ValueError: If any parameter is outside its physical range or
                if the SOC band is degenerate.
        """
        if not 0.0 < self.efficiency_roundtrip <= 1.0:
            raise ValueError(
                "efficiency_roundtrip must be in (0, 1]; "
                f"got {self.efficiency_roundtrip!r}")
        if self.power_capacity_gw <= 0:
            raise ValueError(
                "power_capacity_gw must be positive; "
                f"got {self.power_capacity_gw!r}")
        if self.energy_capacity_gwh <= 0:
            raise ValueError(
                "energy_capacity_gwh must be positive; "
                f"got {self.energy_capacity_gwh!r}")
        if not 0.0 <= self.soc_min_frac < self.soc_max_frac <= 1.0:
            raise ValueError(
                "Operational SOC band invalid: require "
                f"0 <= soc_min_frac ({self.soc_min_frac}) "
                f"< soc_max_frac ({self.soc_max_frac}) <= 1")
        if not (self.soc_min_frac
                <= self.initial_soc_frac
                <= self.soc_max_frac):
            raise ValueError(
                f"initial_soc_frac ({self.initial_soc_frac}) must lie "
                f"inside [{self.soc_min_frac}, {self.soc_max_frac}]")
        if self.self_discharge_per_day < 0.0:
            raise ValueError(
                "self_discharge_per_day must be non-negative; "
                f"got {self.self_discharge_per_day!r}")
        if self.h_synthetic < 0.0:
            raise ValueError(
                f"h_synthetic must be non-negative; got {self.h_synthetic!r}")
        if self.inertia_soc_margin < 0.0:
            raise ValueError(
                "inertia_soc_margin must be non-negative; "
                f"got {self.inertia_soc_margin!r}")

    # ── Derived unit-conversion properties ────────────────────────────

    @property
    def power_capacity_pu(self) -> float:
        """Power capacity in per-unit of :data:`P_PEAK_GW`."""
        return self.power_capacity_gw / P_PEAK_GW

    @property
    def energy_capacity_pu_h(self) -> float:
        """Energy capacity expressed as per-unit-hours.

        1 pu · h = :data:`P_PEAK_GW` GWh, so dividing the nameplate GWh by
        ``P_PEAK_GW`` yields the correct integrated-energy unit that pairs
        directly with power values in p.u.
        """
        return self.energy_capacity_gwh / P_PEAK_GW

    @property
    def eta_charge(self) -> float:
        """One-way charging efficiency = ``sqrt(efficiency_roundtrip)``."""
        return float(np.sqrt(self.efficiency_roundtrip))

    @property
    def eta_discharge(self) -> float:
        """One-way discharging efficiency = ``sqrt(efficiency_roundtrip)``."""
        return float(np.sqrt(self.efficiency_roundtrip))

    @property
    def self_discharge_per_qh(self) -> float:
        """Self-discharge fraction per quarter-hour, 96-compounded.

        The per-qh rate ``λ_qh`` is chosen so that compounding over a full
        day reproduces the per-day rate exactly:
        ``(1 − λ_qh)^96 = 1 − self_discharge_per_day``.
        """
        if self.self_discharge_per_day <= 0.0:
            return 0.0
        return 1.0 - (1.0 - self.self_discharge_per_day) ** (
            1.0 / QUARTERS_PER_DAY)

    # ── Inertia contribution ─────────────────────────────────────────

    def inertia_contribution(self, soc_frac: float) -> tuple[float, float]:
        """Return the (H·S, S) contribution for the system-inertia sum.

        The system inertia constant is
        ``H_system = Σ(H_i · S_i) / Σ(S_i)`` over all units providing
        inertia. For a synchronous turbine the pair is
        ``(H_nom · capacity_pu, capacity_pu)``; for a BESS the same
        structure is used with :attr:`h_synthetic` replacing ``H_nom``,
        but the contribution is gated on SOC being sufficiently far from
        the operational bounds.

        Args:
            soc_frac: Current SOC as a fraction of
                :attr:`energy_capacity_gwh`.

        Returns:
            ``(weighted_H, weight)``: both zero if the BESS cannot provide
            bidirectional response at this timestep (SOC at the bounds,
            or ``h_synthetic == 0``); otherwise
            ``(h_synthetic × power_capacity_pu, power_capacity_pu)``.
        """
        if self.h_synthetic <= 0.0:
            return 0.0, 0.0
        lo = self.soc_min_frac + self.inertia_soc_margin
        hi = self.soc_max_frac - self.inertia_soc_margin
        if not (lo <= soc_frac <= hi):
            return 0.0, 0.0
        return self.h_synthetic * self.power_capacity_pu, self.power_capacity_pu


def build_storage_units(
    storage_cfg: dict[str, dict] | None,
) -> list[StorageUnit]:
    """Build :class:`StorageUnit` instances from a configuration mapping.

    Skips entries with non-positive energy or power capacity so that the
    caller can temporarily disable a unit by zeroing one of its fields
    without having to remove it from the configuration dictionary.

    Args:
        storage_cfg: Mapping ``name -> parameter_dict``. When ``None`` or
            empty, returns an empty list — the caller can then run a
            storage-free dispatch exactly as before.

    Returns:
        List of :class:`StorageUnit` objects, in the iteration order of
        ``storage_cfg``. The iteration order is stable in modern Python
        (3.7+), so callers get a reproducible unit ordering for the
        per-unit aggregation arrays downstream.
    """
    if not storage_cfg:
        return []
    units: list[StorageUnit] = []
    for name, params in storage_cfg.items():
        if params.get('energy_capacity_gwh', 0.0) <= 0.0:
            continue
        if params.get('power_capacity_gw', 0.0) <= 0.0:
            continue
        units.append(StorageUnit(name=name, **params))
    return units
