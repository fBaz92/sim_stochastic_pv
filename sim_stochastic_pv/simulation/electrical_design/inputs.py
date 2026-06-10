"""
Input dataclasses for the electrical designer: installation site and
design requirements.

The datasheet side (panel + inverter) reuses the Phase-16
:class:`~sim_stochastic_pv.simulation.electrical.PanelElectricalSpecs`
and :class:`~sim_stochastic_pv.simulation.electrical.InverterElectricalSpecs`
extended with the designer-only fields (α coefficient, system voltage,
max series fuse; per-MPPT current limits, full-load MPPT window).
"""

from __future__ import annotations

from dataclasses import dataclass


# Default cell over-temperature in full sun (°C): T_cell = T_amb + ΔT.
# Typical range 25–35 °C, derivable from the panel NOCT.
DEFAULT_DELTA_T_CELL_C = 30.0

# Safety factor on I_sc for cable/protection sizing (NEC/IEC practice):
# covers irradiance above 1000 W/m² (snow albedo, high altitude).
DEFAULT_SAFETY_FACTOR_ISC = 1.25

# Target maximum DC cable loss as a fraction of the installed DC power.
DEFAULT_MAX_CABLE_LOSS_FRACTION = 0.01

# CEI EN 62548 string-fuse window: I_n in [1.5, 2.4] × I_sc(STC).
DEFAULT_FUSE_FACTOR_MIN = 1.5
DEFAULT_FUSE_FACTOR_MAX = 2.4


@dataclass(frozen=True)
class DesignSite:
    """
    Thermal envelope of the installation site for sizing purposes.

    The two extreme corners drive the whole string sizing:

    - **Cold case** — a cold, sunny dawn with the module still at
      ambient temperature: maximum V_oc, checked against the inverter's
      absolute DC limit. Assumption: ``T_cell = t_min_c``.
    - **Hot case** — peak summer afternoon in full sun:
      ``T_cell = t_max_c + delta_t_cell_c``; minimum V_mp, checked
      against the MPPT lower bound, and maximum currents.

    Attributes:
        t_min_c: Record minimum ambient temperature (°C). For a site
            with a calibrated climate profile, use an extreme quantile
            of the simulated minima (the extremes backtest exposes
            them) rather than a guess.
        t_max_c: Record maximum ambient temperature (°C).
        delta_t_cell_c: Cell over-temperature in full sun (°C),
            default :data:`DEFAULT_DELTA_T_CELL_C`.

    Example:
        ```python
        site = DesignSite(t_min_c=-20.0, t_max_c=45.0)
        ```
    """

    t_min_c: float
    t_max_c: float
    delta_t_cell_c: float = DEFAULT_DELTA_T_CELL_C

    def __post_init__(self) -> None:
        if self.t_min_c >= self.t_max_c:
            raise ValueError(
                f"t_min_c ({self.t_min_c}) must be below t_max_c ({self.t_max_c})"
            )
        if self.delta_t_cell_c < 0:
            raise ValueError("delta_t_cell_c must be >= 0")

    @property
    def t_cell_cold_c(self) -> float:
        """Design cold-case cell temperature (= ambient minimum)."""
        return self.t_min_c

    @property
    def t_cell_hot_c(self) -> float:
        """Design hot-case cell temperature (= ambient max + ΔT)."""
        return self.t_max_c + self.delta_t_cell_c


@dataclass(frozen=True)
class DesignRequirements:
    """
    Project requirements and verification parameters.

    Attributes:
        p_ac_required_kw: AC power the plant must deliver (kW). The
            plant sizing picks the number of strings that reaches
            ``p_ac_required_kw × target_dc_ac_ratio`` of DC power.
        target_dc_ac_ratio: Desired DC oversizing over the AC rating
            (typical residential range 1.1–1.3; default 1.2).
        n_panels_per_string: Chosen modules per string. Must fall in
            the admissible range computed by
            :func:`~.sizing.compute_string_bounds` — the evaluation
            reports the violation instead of raising, so the UI can
            show "FUORI RANGE" while the user types.
        safety_factor_isc: Multiplier on the hot-corner I_sc for cable
            and protection sizing (default
            :data:`DEFAULT_SAFETY_FACTOR_ISC`).
        max_cable_loss_fraction: Acceptance threshold for the total DC
            cable loss as a fraction of the installed DC power
            (default :data:`DEFAULT_MAX_CABLE_LOSS_FRACTION` = 1 %).
        fuse_factor_min: Lower CEI EN 62548 factor on I_sc(STC) for the
            string fuse (default 1.5).
        fuse_factor_max: Upper factor (default 2.4).
    """

    p_ac_required_kw: float
    target_dc_ac_ratio: float = 1.2
    n_panels_per_string: int = 1
    safety_factor_isc: float = DEFAULT_SAFETY_FACTOR_ISC
    max_cable_loss_fraction: float = DEFAULT_MAX_CABLE_LOSS_FRACTION
    fuse_factor_min: float = DEFAULT_FUSE_FACTOR_MIN
    fuse_factor_max: float = DEFAULT_FUSE_FACTOR_MAX

    def __post_init__(self) -> None:
        if self.p_ac_required_kw <= 0:
            raise ValueError("p_ac_required_kw must be > 0")
        if self.target_dc_ac_ratio <= 0:
            raise ValueError("target_dc_ac_ratio must be > 0")
        if self.n_panels_per_string < 1:
            raise ValueError("n_panels_per_string must be >= 1")
        if self.safety_factor_isc < 1.0:
            raise ValueError("safety_factor_isc must be >= 1")
        if not 0.0 < self.max_cable_loss_fraction < 1.0:
            raise ValueError("max_cable_loss_fraction must be in (0, 1)")
        if not 0.0 < self.fuse_factor_min <= self.fuse_factor_max:
            raise ValueError(
                "fuse factors must satisfy 0 < min <= max "
                f"(got {self.fuse_factor_min}, {self.fuse_factor_max})"
            )
