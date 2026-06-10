"""
Production preview for the electrical designer: a fast hourly Monte
Carlo that turns a static design into expected annual energy with
uncertainty bands.

This is the step the reference spreadsheet explicitly leaves out
("valutare il clipping con una simulazione oraria"): given the chosen
DC/AC ratio, string layout and cable section, the preview quantifies

- the **AC energy** actually delivered per year (mean + p05/p95 band
  across stochastic weather paths);
- the **inverter clipping** loss implied by the DC/AC ratio;
- the **ohmic cable loss** integrated hour by hour (∝ I²,
  so a static "x% at STC" figure underestimates sunny-hour losses);
- the **electrical-model derating** (MPPT window + temperature) when a
  calibrated climate profile is available.

Economics are deliberately out of scope here — the designer hands the
finished design to the standard analysis flow for NPV/IRR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..electrical import ElectricalModel
from ..solar import SolarModel
from ..thermal import ThermalModel, month_of_year


@dataclass(frozen=True)
class CableLossSpec:
    """
    Parameters of the per-hour ohmic cable-loss computation.

    The hourly string current is approximated as
    ``I(h) = P_string(h) / V_mp_string_stc`` — voltage held at the STC
    operating point. The approximation slightly overestimates the
    current (and the loss) on hot afternoons when the true voltage is
    lower, which is the conservative direction for a loss estimate.

    Attributes:
        resistance_per_string_ohm: Out-and-return conductor resistance
            of one string run (Ω) at the cable operating temperature.
        n_strings: Number of parallel strings (each with its own run).
        v_mp_string_stc_v: String MPP voltage at STC (V).
    """

    resistance_per_string_ohm: float
    n_strings: int
    v_mp_string_stc_v: float

    def __post_init__(self) -> None:
        if self.resistance_per_string_ohm < 0:
            raise ValueError("resistance_per_string_ohm must be >= 0")
        if self.n_strings < 1:
            raise ValueError("n_strings must be >= 1")
        if self.v_mp_string_stc_v <= 0:
            raise ValueError("v_mp_string_stc_v must be > 0")


@dataclass
class ProductionPreviewResult:
    """
    Aggregated output of :func:`simulate_production_preview`.

    All energies are annual values (kWh/year). Band percentiles are
    computed across the Monte Carlo paths.

    Attributes:
        annual_dc_kwh_mean: PV DC energy before any loss.
        annual_ac_kwh_mean: Delivered AC energy, mean across paths.
        annual_ac_kwh_p05: 5th percentile of the delivered AC energy.
        annual_ac_kwh_p95: 95th percentile.
        clipping_kwh_mean: Energy lost to the inverter AC cap.
        clipping_fraction: ``clipping / dc`` (0–1).
        cable_loss_kwh_mean: Ohmic cable loss.
        cable_loss_fraction: ``cable_loss / dc`` (0–1).
        electrical_derating_kwh_mean: Energy removed by the MPPT-window
            + temperature model (0 when no climate profile was wired).
        inverter_efficiency: Conversion efficiency applied to the AC
            side (echoed input).
        hours_outside_mppt_per_year_mean: Electrical-model KPI, 0 when
            the model was off.
        hours_dc_overvoltage_per_year_mean: Same.
        n_paths: Paths simulated.
    """

    annual_dc_kwh_mean: float = 0.0
    annual_ac_kwh_mean: float = 0.0
    annual_ac_kwh_p05: float = 0.0
    annual_ac_kwh_p95: float = 0.0
    clipping_kwh_mean: float = 0.0
    clipping_fraction: float = 0.0
    cable_loss_kwh_mean: float = 0.0
    cable_loss_fraction: float = 0.0
    electrical_derating_kwh_mean: float = 0.0
    inverter_efficiency: float = 1.0
    hours_outside_mppt_per_year_mean: float = 0.0
    hours_dc_overvoltage_per_year_mean: float = 0.0
    n_paths: int = 0


def simulate_production_preview(
    solar_model: SolarModel,
    p_ac_max_kw: float,
    inverter_efficiency: float = 1.0,
    n_paths: int = 30,
    seed: int = 42,
    thermal_model: ThermalModel | None = None,
    electrical_model: ElectricalModel | None = None,
    cable: CableLossSpec | None = None,
) -> ProductionPreviewResult:
    """
    Simulate one year of hourly production over ``n_paths`` weather paths.

    Loss chain per hour (in physical order):

    1. stochastic PV DC power from the solar model;
    2. optional electrical-model adjustment (MPPT window shutdown /
       derating + γ temperature derating) using hourly T_ambient from
       the thermal model;
    3. optional ohmic cable loss ``n_strings · R · I(h)²``;
    4. inverter AC cap (clipping) at ``p_ac_max_kw``;
    5. inverter efficiency.

    Args:
        solar_model: Configured :class:`SolarModel` for the site and the
            installed DC power.
        p_ac_max_kw: Inverter AC nameplate (kW).
        inverter_efficiency: Conversion efficiency (0–1], default 1.
        n_paths: Monte Carlo paths (each one synthetic weather year).
        seed: Master seed; path ``p`` uses ``seed + p``.
        thermal_model: Optional calibrated ambient-temperature model —
            required for the electrical model to run.
        electrical_model: Optional Phase-16 MPPT-window model.
        cable: Optional :class:`CableLossSpec`; ``None`` skips step 3.

    Returns:
        :class:`ProductionPreviewResult`.

    Raises:
        ValueError: Non-positive ``p_ac_max_kw`` / ``n_paths`` or an
            ``electrical_model`` without a ``thermal_model``.
    """
    if p_ac_max_kw <= 0:
        raise ValueError("p_ac_max_kw must be > 0")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")
    if not 0.0 < inverter_efficiency <= 1.0:
        raise ValueError("inverter_efficiency must be in (0, 1]")
    if electrical_model is not None and thermal_model is None:
        raise ValueError(
            "electrical_model requires a thermal_model for hourly T_ambient"
        )

    n_days = 365
    days = np.arange(n_days)
    months = np.asarray(month_of_year(days), dtype=np.int64)
    year_index = np.zeros(n_days, dtype=np.int64)

    annual_dc = np.zeros(n_paths)
    annual_ac = np.zeros(n_paths)
    clipping = np.zeros(n_paths)
    cable_loss = np.zeros(n_paths)
    electrical_loss = np.zeros(n_paths)
    hours_off_mppt = np.zeros(n_paths)
    hours_overvoltage = np.zeros(n_paths)

    for p in range(n_paths):
        rng = np.random.default_rng(seed + p)
        daily_kwh = solar_model.simulate_daily_energy(
            n_years=1,
            month_in_year_for_day=months,
            year_index_for_day=year_index,
            rng=rng,
        )
        hourly_kw = np.zeros(n_days * 24)
        for d in range(n_days):
            hourly_kw[d * 24:(d + 1) * 24] = solar_model.daily_profile_kwh(
                daily_kwh[d]
            )
        annual_dc[p] = hourly_kw.sum()

        power = hourly_kw
        if electrical_model is not None and thermal_model is not None:
            t_amb = thermal_model.to_hourly(
                thermal_model.simulate_daily_means(n_days, rng)
            )
            adjusted, kpis = electrical_model.apply_to_pv_dc(power, t_amb)
            electrical_loss[p] = float(power.sum() - adjusted.sum())
            hours_off_mppt[p] = kpis.hours_outside_mppt_per_year
            hours_overvoltage[p] = kpis.hours_dc_overvoltage_per_year
            power = adjusted

        if cable is not None:
            i_per_string_a = (power * 1000.0) / (
                cable.n_strings * cable.v_mp_string_stc_v
            )
            loss_kw = (
                cable.n_strings
                * cable.resistance_per_string_ohm
                * i_per_string_a**2
                / 1000.0
            )
            loss_kw = np.minimum(loss_kw, power)
            cable_loss[p] = float(loss_kw.sum())
            power = power - loss_kw

        ac_power = np.minimum(power, p_ac_max_kw)
        clipping[p] = float((power - ac_power).sum())
        annual_ac[p] = float(ac_power.sum() * inverter_efficiency)

    dc_mean = float(annual_dc.mean())
    return ProductionPreviewResult(
        annual_dc_kwh_mean=dc_mean,
        annual_ac_kwh_mean=float(annual_ac.mean()),
        annual_ac_kwh_p05=float(np.percentile(annual_ac, 5)),
        annual_ac_kwh_p95=float(np.percentile(annual_ac, 95)),
        clipping_kwh_mean=float(clipping.mean()),
        clipping_fraction=float(clipping.mean() / dc_mean) if dc_mean > 0 else 0.0,
        cable_loss_kwh_mean=float(cable_loss.mean()),
        cable_loss_fraction=(
            float(cable_loss.mean() / dc_mean) if dc_mean > 0 else 0.0
        ),
        electrical_derating_kwh_mean=float(electrical_loss.mean()),
        inverter_efficiency=inverter_efficiency,
        hours_outside_mppt_per_year_mean=float(hours_off_mppt.mean()),
        hours_dc_overvoltage_per_year_mean=float(hours_overvoltage.mean()),
        n_paths=n_paths,
    )
