from __future__ import annotations

import warnings

import numpy as np
import pytest

from sim_stochastic_pv.simulation.battery import BatteryBank, BatterySpecs
from sim_stochastic_pv.simulation.monte_carlo import _npv
from sim_stochastic_pv.simulation.solar import SolarModel, SolarMonthParams


def test_battery_bank_charge_and_discharge_updates_soh() -> None:
    specs = BatterySpecs(capacity_kwh=10.0, cycles_life=5)
    bank = BatteryBank(
        specs=specs,
        n_batteries=1,
        eta_charge=1.0,
        eta_discharge=1.0,
        max_charge_kw=None,
        max_discharge_kw=None,
        dt_hours=1.0,
    )

    used = bank.charge(5.0)
    assert used == pytest.approx(5.0)
    assert bank.soc_kwh == pytest.approx(5.0)

    delivered = bank.discharge(3.0)
    assert delivered == pytest.approx(3.0)
    assert bank.soc_kwh == pytest.approx(2.0)
    assert bank.soh < 1.0  # degradation applied after discharge
    assert bank.capacity_bank_kwh < bank.capacity_nominal_kwh
    assert bank.eta_charge < 1.0


def test_solar_model_applies_degradation() -> None:
    params = [SolarMonthParams(5.0, 0.5, 1.0, 1.0) for _ in range(12)]
    model = SolarModel(pv_kwp=1.0, month_params=params, degradation_per_year=0.02)

    n_years = 2
    n_days = 24
    month_in_year_for_day = np.zeros(n_days, dtype=int)
    year_index_for_day = np.concatenate([np.zeros(n_days // 2, dtype=int), np.ones(n_days // 2, dtype=int)])

    rng = np.random.default_rng(0)
    energy = model.simulate_daily_energy(
        n_years=n_years,
        month_in_year_for_day=month_in_year_for_day,
        year_index_for_day=year_index_for_day,
        rng=rng,
    )

    first_year = energy[: n_days // 2].mean()
    second_year = energy[n_days // 2 :].mean()
    expected_ratio = (1.0 - model.degradation_per_year) ** 1
    assert second_year == pytest.approx(first_year * expected_ratio, rel=0.1)


def test_npv_handles_extreme_negative_rates_without_runtime_warning() -> None:
    cashflows = np.array([-1000.0, 200.0, 400.0, 600.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("error", RuntimeWarning)
        value = _npv(-2.0, cashflows)

    assert np.isfinite(value)
    assert caught == []
