"""
I-V / P-V curve families for a PV module at arbitrary irradiance and
cell temperature, built on the single-diode model.

The five single-diode parameters are fitted **once at STC** from the
datasheet points by
:class:`~sim_stochastic_pv.simulation.pv_model.PVModelSingleDiode`,
then *scaled physically* to each (G, T) condition (simplified De Soto /
IEC 60891 behaviour):

    Iph(G, T) = Iph_stc · g · (1 + α/100 · ΔT)            g = G / 1000
    Voc(G, T) = Voc_stc · (1 + β/100 · ΔT) + n·Ns·Vt(T) · ln(g)
    Is(G, T)  = (Iph − Voc/Rsh) / (exp(Voc / (n·Ns·Vt(T))) − 1)

with ``ΔT = T − 25``; ``n``, ``Rs`` and ``Rsh`` are kept from the STC
fit. Solving ``Is`` in closed form from the open-circuit condition makes
every curve cross its β-corrected V_oc exactly without a per-condition
re-fit (numerically fragile at extreme temperatures). The curve is then
evaluated on a voltage grid with a vectorised damped-Newton iteration
(the per-point ``least_squares`` of ``i_of_v`` would be too slow for
interactive product sheets).

Notes:
    Below ~50 W/m² the logarithmic V_oc shift degrades — the public
    helper clamps irradiance at 50 W/m².
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .pv_model import PVModelSingleDiode


# Lowest irradiance accepted for a curve (W/m²): below this the log
# translation of Voc is unreliable and the curve has no practical use.
MIN_IRRADIANCE_W_M2 = 50.0

# Default condition families for the product sheet.
DEFAULT_IRRADIANCES_W_M2: tuple[float, ...] = (200.0, 400.0, 600.0, 800.0, 1000.0)
DEFAULT_TEMPERATURES_C: tuple[float, ...] = (-10.0, 25.0, 45.0, 70.0)

STC_IRRADIANCE_W_M2 = 1000.0
STC_TEMPERATURE_C = 25.0


@dataclass(frozen=True)
class PanelCurve:
    """
    One I-V / P-V curve at a fixed (irradiance, cell temperature).

    Attributes:
        irradiance_w_m2: Plane-of-array irradiance (W/m²).
        t_cell_c: Cell temperature (°C).
        v: Voltage grid (V), ascending, ``n_points`` long.
        i: Current at each grid voltage (A), clipped at 0.
        p: Power ``v · i`` (W).
        mpp_v: Voltage of the maximum-power point on the grid (V).
        mpp_i: Current at the MPP (A).
        mpp_p: Power at the MPP (W).
    """

    irradiance_w_m2: float
    t_cell_c: float
    v: list[float] = field(default_factory=list)
    i: list[float] = field(default_factory=list)
    p: list[float] = field(default_factory=list)
    mpp_v: float = 0.0
    mpp_i: float = 0.0
    mpp_p: float = 0.0


def _evaluate_iv(model: PVModelSingleDiode, v: np.ndarray) -> np.ndarray:
    """
    Vectorised Newton solution of the implicit single-diode equation.

    Solves ``f(I) = Iph − Is·(exp((V+I·Rs)/(n·Vt·Ns)) − 1) − (V+I·Rs)/Rsh − I``
    for every grid voltage simultaneously. 40 damped Newton steps from
    ``I₀ = Isc`` converge far below float precision for physical
    parameter sets; the result is clipped at 0 (the product sheet does
    not display the negative-current quadrant).

    Args:
        model: A *solved* :class:`PVModelSingleDiode`.
        v: Voltage grid (V), shape ``(n,)``.

    Returns:
        Currents (A), shape ``(n,)``, clipped at 0.
    """
    iph, isat, n, rs, rsh = model.Iph, model.Is, model.n, model.Rs, model.Rsh
    nvt = n * model.Vt_mod
    current = np.full_like(v, model.Isc, dtype=float)
    for _ in range(40):
        arg = np.clip((v + current * rs) / nvt, -50.0, 80.0)
        exp_term = np.exp(arg)
        f = iph - isat * (exp_term - 1.0) - (v + current * rs) / rsh - current
        df = -isat * exp_term * (rs / nvt) - rs / rsh - 1.0
        step = f / df
        current = current - 0.8 * step  # damped for robustness near Voc
    return np.clip(current, 0.0, None)


def _curve_at(
    stc_model: PVModelSingleDiode,
    isc_stc: float,
    voc_stc: float,
    alpha_pct: float,
    beta_pct: float,
    irradiance_w_m2: float,
    t_cell_c: float,
    n_points: int,
) -> PanelCurve:
    """
    Scale the STC-fitted parameters to one (G, T) condition and evaluate.

    Instead of re-fitting five parameters per condition (numerically
    fragile at extreme temperatures), the STC fit is scaled physically:

    - ``Iph' = Iph · g · (1 + α/100·ΔT)`` — photocurrent linear in
      irradiance and α-corrected;
    - the diode ideality ``n``, ``Rs`` and ``Rsh`` are kept from STC;
    - ``Is'`` is obtained in closed form from the open-circuit condition
      at the *translated* Voc target
      (``Is' = (Iph' − Voc'/Rsh) / (exp(Voc'/(n·Vt'(T))) − 1)``),
      so every curve crosses its β-corrected open-circuit voltage
      exactly and inherits the temperature dependence of the diode.
    """
    g = max(irradiance_w_m2, MIN_IRRADIANCE_W_M2) / STC_IRRADIANCE_W_M2
    dt = t_cell_c - STC_TEMPERATURE_C

    from scipy.constants import Boltzmann, elementary_charge

    vt_mod = Boltzmann * (t_cell_c + 273.15) / elementary_charge * stc_model.Ns
    n_vt = stc_model.n * vt_mod

    iph = stc_model.Iph * g * (1.0 + alpha_pct / 100.0 * dt)
    voc = voc_stc * (1.0 + beta_pct / 100.0 * dt) + n_vt * float(np.log(g))
    is_scaled = (iph - voc / stc_model.Rsh) / (np.exp(voc / n_vt) - 1.0)

    scaled = PVModelSingleDiode(
        Isc=isc_stc * g * (1.0 + alpha_pct / 100.0 * dt),
        Voc=voc,
        Imp=stc_model.Imp,  # informational only — not used by evaluation
        Vmp=stc_model.Vmp,
        Ns=stc_model.Ns,
        Tcell=t_cell_c,
    )
    scaled.Iph = float(iph)
    scaled.Is = float(max(is_scaled, 1e-15))
    scaled.n = stc_model.n
    scaled.Rs = stc_model.Rs
    scaled.Rsh = stc_model.Rsh

    v = np.linspace(0.0, voc, n_points)
    i = _evaluate_iv(scaled, v)
    p = v * i
    mpp_idx = int(np.argmax(p))
    return PanelCurve(
        irradiance_w_m2=float(irradiance_w_m2),
        t_cell_c=float(t_cell_c),
        v=[float(x) for x in v],
        i=[float(x) for x in i],
        p=[float(x) for x in p],
        mpp_v=float(v[mpp_idx]),
        mpp_i=float(i[mpp_idx]),
        mpp_p=float(p[mpp_idx]),
    )


def compute_panel_curve_families(
    isc_stc: float,
    voc_stc: float,
    imp_stc: float,
    vmp_stc: float,
    n_cells_series: int,
    alpha_isc_pct_per_c: float,
    beta_voc_pct_per_c: float,
    irradiances_w_m2: Sequence[float] = DEFAULT_IRRADIANCES_W_M2,
    temperatures_c: Sequence[float] = DEFAULT_TEMPERATURES_C,
    n_points: int = 80,
) -> tuple[list[PanelCurve], list[PanelCurve]]:
    """
    Build the two standard curve families of a module product sheet.

    Family 1 sweeps irradiance at the STC cell temperature (25 °C);
    family 2 sweeps cell temperature at full sun (1000 W/m²). The
    single-diode model is fitted once at STC and physically scaled to
    each condition (see the module docstring). γ(P_max) is not needed:
    the power-temperature behaviour emerges from the diode physics via
    the α and β corrections.

    Args:
        isc_stc: Short-circuit current at STC (A).
        voc_stc: Open-circuit voltage at STC (V).
        imp_stc: MPP current at STC (A).
        vmp_stc: MPP voltage at STC (V).
        n_cells_series: Cells in series inside the module.
        alpha_isc_pct_per_c: I_sc temperature coefficient (%/°C, > 0).
        beta_voc_pct_per_c: V_oc temperature coefficient (%/°C, < 0).
        irradiances_w_m2: Irradiance family values (clamped at
            :data:`MIN_IRRADIANCE_W_M2`).
        temperatures_c: Temperature family values.
        n_points: Grid points per curve (default 80 — smooth enough for
            charts while keeping the payload small).

    Returns:
        Tuple ``(irradiance_family, temperature_family)`` of
        :class:`PanelCurve` lists.

    Raises:
        ValueError: Non-physical datasheet values (non-positive, or
            MPP outside the (0, Isc)×(0, Voc) box).
        RuntimeError: The STC single-diode fit failed to converge.

    Example:
        ```python
        irr, temp = compute_panel_curve_families(
            isc_stc=15.88, voc_stc=40.14, imp_stc=14.9, vmp_stc=33.9,
            n_cells_series=108, alpha_isc_pct_per_c=0.045,
            beta_voc_pct_per_c=-0.25,
        )
        irr[-1].mpp_p   # ≈ 505 W at STC
        ```
    """
    if min(isc_stc, voc_stc, imp_stc, vmp_stc) <= 0 or n_cells_series <= 0:
        raise ValueError("datasheet values must be positive")
    if not (imp_stc < isc_stc and vmp_stc < voc_stc):
        raise ValueError(
            "MPP must lie strictly inside the (Isc, Voc) box: "
            f"got Imp={imp_stc} vs Isc={isc_stc}, Vmp={vmp_stc} vs Voc={voc_stc}"
        )

    stc_model = PVModelSingleDiode(
        Isc=isc_stc, Voc=voc_stc, Imp=imp_stc, Vmp=vmp_stc,
        Ns=n_cells_series, Tcell=STC_TEMPERATURE_C,
    )
    stc_model.solve()

    common = dict(
        stc_model=stc_model,
        isc_stc=isc_stc,
        voc_stc=voc_stc,
        alpha_pct=alpha_isc_pct_per_c,
        beta_pct=beta_voc_pct_per_c,
        n_points=n_points,
    )
    irradiance_family = [
        _curve_at(irradiance_w_m2=g, t_cell_c=STC_TEMPERATURE_C, **common)
        for g in irradiances_w_m2
    ]
    temperature_family = [
        _curve_at(irradiance_w_m2=STC_IRRADIANCE_W_M2, t_cell_c=t, **common)
        for t in temperatures_c
    ]
    return irradiance_family, temperature_family
