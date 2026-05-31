"""Tests for PV surplus capture (export vs curtailment).

Surplus PV — energy neither self-consumed nor stored — used to be discarded
silently. The inverter now splits it into an exportable part (bounded by the
AC ceiling, to be valued under dedicated withdrawal) and a curtailed part.
These tests pin the per-hour energy balance at the inverter level and the
horizon-level accounting surfaced by the energy simulator.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.simulation.battery import BatteryBank, BatterySpecs
from sim_stochastic_pv.simulation.inverter import InverterAC
from sim_stochastic_pv.simulation.energy_simulator import (
    EnergySystemConfig,
    EnergySystemSimulator,
)
from sim_stochastic_pv.simulation.solar import (
    SolarModel,
    make_default_solar_params_for_pavullo,
)
from sim_stochastic_pv.simulation.load_profiles import MonthlyAverageLoadProfile


# ── Inverter-level balance (deterministic) ────────────────────────────────

def _full_battery(capacity_kwh: float = 1.0) -> BatteryBank:
    """A battery at the top of its window (no charge headroom)."""
    bank = BatteryBank(specs=BatterySpecs(capacity_kwh=capacity_kwh),
                       n_batteries=1)
    bank.reset(soc_init=1.0)
    return bank


def _empty_room_battery(capacity_kwh: float = 10.0) -> BatteryBank:
    """A large, empty battery with ample charge headroom."""
    bank = BatteryBank(specs=BatterySpecs(capacity_kwh=capacity_kwh),
                       n_batteries=1, max_charge_kw=None)
    bank.reset(soc_init=0.0)
    return bank


def test_dispatch_returns_seven_terms_and_balances() -> None:
    """e_pv_prod == direct + to_batt + to_grid + curtailed, every term >= 0."""
    inv = InverterAC(p_ac_max_kw=3.0)
    out = inv.dispatch(p_pv_dc_kw=5.0, p_load_kw=1.0, battery=_full_battery())
    assert len(out) == 7
    (e_prod, e_direct, e_batt_dis, e_grid_imp, e_to_batt,
     e_to_grid, e_curtailed) = out
    assert all(v >= 0.0 for v in out)
    assert e_prod == pytest.approx(e_direct + e_to_batt + e_to_grid + e_curtailed)


def test_export_capped_by_ac_headroom_rest_curtailed() -> None:
    """With a full battery the surplus splits at the inverter AC ceiling."""
    # 5 kW PV, 1 kW load, full battery, 3 kW AC ceiling.
    # PV-direct = 1 kW (the load). AC headroom for export = 3 - 1 = 2 kW.
    # Surplus = 5 - 1 = 4 kWh; 2 kWh exportable, 2 kWh curtailed.
    inv = InverterAC(p_ac_max_kw=3.0)
    (_, e_direct, _, _, e_to_batt, e_to_grid, e_curtailed) = inv.dispatch(
        p_pv_dc_kw=5.0, p_load_kw=1.0, battery=_full_battery())
    assert e_direct == pytest.approx(1.0)
    assert e_to_batt == pytest.approx(0.0)
    assert e_to_grid == pytest.approx(2.0)
    assert e_curtailed == pytest.approx(2.0)


def test_ample_inverter_exports_all_surplus() -> None:
    """A generous AC ceiling exports the whole surplus, nothing curtailed."""
    inv = InverterAC(p_ac_max_kw=100.0)
    (_, e_direct, _, _, e_to_batt, e_to_grid, e_curtailed) = inv.dispatch(
        p_pv_dc_kw=5.0, p_load_kw=1.0, battery=_full_battery())
    assert e_to_grid == pytest.approx(4.0)
    assert e_curtailed == pytest.approx(0.0)


def test_surplus_charges_battery_before_export() -> None:
    """An empty battery soaks up the surplus first; export only on the remainder."""
    inv = InverterAC(p_ac_max_kw=100.0)
    bank = _empty_room_battery(capacity_kwh=10.0)
    (_, e_direct, _, _, e_to_batt, e_to_grid, e_curtailed) = inv.dispatch(
        p_pv_dc_kw=5.0, p_load_kw=1.0, battery=bank)
    # Battery had room, so it must take some surplus before any export.
    assert e_to_batt > 0.0
    assert e_to_batt + e_to_grid + e_curtailed == pytest.approx(4.0)


def test_no_surplus_no_export() -> None:
    """When PV does not exceed the load there is nothing to export or curtail."""
    inv = InverterAC(p_ac_max_kw=3.0)
    (_, e_direct, _, e_grid_imp, e_to_batt, e_to_grid, e_curtailed) = inv.dispatch(
        p_pv_dc_kw=0.5, p_load_kw=2.0, battery=_full_battery())
    assert e_to_grid == pytest.approx(0.0)
    assert e_curtailed == pytest.approx(0.0)
    assert e_grid_imp > 0.0  # load partly served from the grid


# ── Simulator-level accounting ────────────────────────────────────────────

def _make_simulator(n_years=1, pv_kwp=3.0, inverter_p_ac_max_kw=3.0, **kwargs):
    solar_model = SolarModel(
        pv_kwp=pv_kwp,
        month_params=make_default_solar_params_for_pavullo(),
        degradation_per_year=0.0,
    )
    load_profile = MonthlyAverageLoadProfile(monthly_avg_kwh=[200.0] * 12)
    config = EnergySystemConfig(
        n_years=n_years,
        pv_kwp=pv_kwp,
        n_batteries=2,
        inverter_p_ac_max_kw=inverter_p_ac_max_kw,
        **kwargs,
    )
    return EnergySystemSimulator(config, solar_model, load_profile)


def test_simulator_surfaces_surplus_arrays_with_shapes() -> None:
    """run_one_path populates the surplus attributes with the right shapes."""
    sim = _make_simulator(n_years=2)
    sim.run_one_path(np.random.default_rng(0))
    n_months = 2 * 12
    assert sim.last_monthly_export_kwh.shape == (n_months,)
    assert sim.last_monthly_curtailed_kwh.shape == (n_months,)
    assert sim.last_monthly_pv_to_batt_kwh.shape == (n_months,)
    assert sim.last_export_kwh_by_year_month_hour.shape == (2, 12, 24)
    assert (sim.last_monthly_export_kwh >= 0).all()
    assert (sim.last_monthly_curtailed_kwh >= 0).all()


def test_undersized_inverter_produces_surplus() -> None:
    """A big array on a tiny inverter exports and/or curtails real energy."""
    # 4 kWp on a 0.8 kW AC inverter: heavy midday surplus that used to vanish.
    sim = _make_simulator(pv_kwp=4.0, inverter_p_ac_max_kw=0.8)
    sim.run_one_path(np.random.default_rng(1))
    total_surplus = (sim.last_monthly_export_kwh.sum()
                     + sim.last_monthly_curtailed_kwh.sum())
    assert total_surplus > 0.0


def test_simulator_energy_balance_over_horizon() -> None:
    """PV production equals direct + to-battery + export + curtailed (summed)."""
    sim = _make_simulator(pv_kwp=4.0, inverter_p_ac_max_kw=0.8)
    (monthly_pv_prod, monthly_pv_direct, _batt_to_load, _grid_imp,
     _load, _soh, _soc) = sim.run_one_path(np.random.default_rng(2))
    lhs = monthly_pv_prod.sum()
    rhs = (monthly_pv_direct.sum()
           + sim.last_monthly_pv_to_batt_kwh.sum()
           + sim.last_monthly_export_kwh.sum()
           + sim.last_monthly_curtailed_kwh.sum())
    assert lhs == pytest.approx(rhs)


def test_export_grid_matches_monthly_export_total() -> None:
    """The (year, month, hour) export grid sums to the monthly export total."""
    sim = _make_simulator(n_years=2, pv_kwp=4.0, inverter_p_ac_max_kw=0.8)
    sim.run_one_path(np.random.default_rng(3))
    assert sim.last_export_kwh_by_year_month_hour.sum() == pytest.approx(
        sim.last_monthly_export_kwh.sum())


def test_no_export_at_night() -> None:
    """Export is zero at deep-night hours where the PV array produces nothing."""
    sim = _make_simulator(pv_kwp=4.0, inverter_p_ac_max_kw=0.8)
    sim.run_one_path(np.random.default_rng(4))
    grid = sim.last_export_kwh_by_year_month_hour  # (n_years, 12, 24)
    night = grid[:, :, 0:4]  # 00:00-04:00
    assert np.all(night == 0.0)
    assert grid[:, :, 11:15].sum() > 0.0  # midday exports
