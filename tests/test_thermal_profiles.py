"""
Tests for the Phase-2 Dashboard plumbing: hourly mean monthly profiles of
consumption and indoor temperature (with cross-path bands) and the
heating/cooling active-hours KPIs.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.simulation import (
    EconomicConfig,
    EnergySystemConfig,
    EnergySystemSimulator,
    EscalatingPriceModel,
    MonteCarloSimulator,
    MonthlyAverageLoadProfile,
    SolarModel,
)
from sim_stochastic_pv.simulation.solar import make_default_solar_params_for_pavullo
from sim_stochastic_pv.simulation.thermal import (
    HarmonicSeasonalMean,
    ThermalModel,
    ThermalMonthParams,
)
from sim_stochastic_pv.simulation.thermal_load import (
    HeatPumpConfig,
    HouseThermalConfig,
    HvacController,
    SetpointConfig,
    ThermalLoadConfig,
    ThermalLoadKPIs,
    aggregate_thermal_kpis,
)

_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def _solar(pv_kwp=3.0):
    return SolarModel(
        pv_kwp=pv_kwp,
        month_params=make_default_solar_params_for_pavullo(),
        degradation_per_year=0.0,
    )


def _thermal_model(a0: float) -> ThermalModel:
    return ThermalModel(
        harmonic=HarmonicSeasonalMean(a0=a0, a1=0.0, a2=0.0),
        monthly_params=[
            ThermalMonthParams(
                t_std_residual_c=0.5, persistence_phi=0.6, t_amplitude_c=1.0
            )
            for _ in range(12)
        ],
        climate_trend_c_per_year=0.0,
    )


def _thermal_load_cfg(dynamic=True) -> ThermalLoadConfig:
    return ThermalLoadConfig(
        enabled=True,
        dynamic=dynamic,
        house=HouseThermalConfig(floor_area_m2=100.0, insulation_preset="standard"),
        heat_pump=HeatPumpConfig(cop_heating=3.5, cop_cooling=3.0, p_elec_max_kw=5.0),
        setpoint=SetpointConfig(t_setpoint_heating_c=20.0, t_setpoint_cooling_c=26.0),
    )


# ── HVAC heating/cooling active hours ───────────────────────────────────────


def test_hvac_kpis_count_heating_hours_in_cold_climate():
    ctrl = HvacController(_thermal_load_cfg(dynamic=True))
    t_amb = np.full(8760, 2.0)  # cold all year → heating every hour
    _p, kpis = ctrl.compute_hourly_p_elec_kw(t_amb)
    assert kpis.heating_hours_per_year > 8000
    assert kpis.cooling_hours_per_year == pytest.approx(0.0)


def test_hvac_kpis_count_cooling_hours_in_hot_climate():
    ctrl = HvacController(_thermal_load_cfg(dynamic=True))
    t_amb = np.full(8760, 36.0)  # hot all year → cooling every hour
    _p, kpis = ctrl.compute_hourly_p_elec_kw(t_amb)
    assert kpis.cooling_hours_per_year > 8000
    assert kpis.heating_hours_per_year == pytest.approx(0.0)


def test_aggregate_thermal_kpis_exposes_heating_cooling_hours():
    per_path = [
        ThermalLoadKPIs(heating_hours_per_year=1200.0, cooling_hours_per_year=300.0),
        ThermalLoadKPIs(heating_hours_per_year=1400.0, cooling_hours_per_year=200.0),
    ]
    agg = aggregate_thermal_kpis(per_path)
    assert agg["heating_hours_per_year_mean"] == pytest.approx(1300.0)
    assert agg["cooling_hours_per_year_mean"] == pytest.approx(250.0)
    # Empty input still yields the keys (stable schema).
    assert "heating_hours_per_year_mean" in aggregate_thermal_kpis([])


# ── consumption profile (no thermal) ────────────────────────────────────────


def test_consumption_profile_reconstructs_monthly_total():
    load = MonthlyAverageLoadProfile(monthly_avg_kwh=[150.0] * 12)
    cfg = EnergySystemConfig(n_years=2, pv_kwp=3.0, n_batteries=0, inverter_p_ac_max_kw=3.0)
    esim = EnergySystemSimulator(cfg, _solar(), load)
    out = esim.run_one_path(np.random.default_rng(0))

    lp = esim.last_load_kwh_by_year_month_hour
    assert lp.shape == (2, 12, 24)
    assert esim.last_indoor_temp_c_by_year_month_hour is None  # no HVAC

    # mean hourly profile × days-in-month, summed over hours == monthly load.
    recon = np.array(
        [[lp[y, m, :].sum() * _DAYS[m] for m in range(12)] for y in range(2)]
    ).reshape(-1)
    assert np.allclose(recon, out[4], rtol=1e-6)


def test_mc_surfaces_consumption_profile_bands():
    load = MonthlyAverageLoadProfile(monthly_avg_kwh=[150.0] * 12)
    cfg = EnergySystemConfig(n_years=2, pv_kwp=3.0, n_batteries=0, inverter_p_ac_max_kw=3.0)
    esim = EnergySystemSimulator(cfg, _solar(), load)
    econ = EconomicConfig(investment_eur=5000.0, n_mc=4, inflation_rate=0.0)
    res = MonteCarloSimulator(
        esim, EscalatingPriceModel(use_stochastic_escalation=False), econ
    ).run(seed=1, show_progress=False)
    assert res.load_profile_mean_kwh.shape == (2, 12, 24)
    assert res.indoor_temp_profile_mean_c is None
    assert np.all(res.load_profile_p05_kwh <= res.load_profile_mean_kwh + 1e-9)
    assert np.all(res.load_profile_mean_kwh <= res.load_profile_p95_kwh + 1e-9)


# ── indoor-temperature profile (dynamic thermal) ────────────────────────────


def test_mc_surfaces_indoor_temperature_profile_in_dynamic_mode():
    load = MonthlyAverageLoadProfile(monthly_avg_kwh=[150.0] * 12)
    cfg = EnergySystemConfig(
        n_years=1,
        pv_kwp=3.0,
        n_batteries=0,
        inverter_p_ac_max_kw=5.0,
        thermal_model=_thermal_model(a0=2.0),  # cold → heating dominant
        thermal_load_config=_thermal_load_cfg(dynamic=True),
    )
    esim = EnergySystemSimulator(cfg, _solar(), load)
    econ = EconomicConfig(investment_eur=5000.0, n_mc=3, inflation_rate=0.0)
    res = MonteCarloSimulator(
        esim, EscalatingPriceModel(use_stochastic_escalation=False), econ
    ).run(seed=2, show_progress=False)

    assert res.indoor_temp_profile_mean_c is not None
    assert res.indoor_temp_profile_mean_c.shape == (1, 12, 24)
    # Indoor temperature is held near the heating setpoint (~20 °C).
    assert 15.0 < float(np.nanmean(res.indoor_temp_profile_mean_c)) < 25.0
    # The thermal KPIs report heating-dominated operation.
    assert res.thermal_kpis_summary["heating_hours_per_year_mean"] > 0.0
