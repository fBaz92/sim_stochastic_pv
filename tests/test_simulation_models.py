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
        soc_init=0,
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


# ---------------------------------------------------------------------------
# Markov-chain weather model (Phase 1 of the roadmap)
# ---------------------------------------------------------------------------


def _simulate_sunny_states(
    p_sunny: float,
    weather_persistence: float,
    n_days: int = 50_000,
    seed: int = 12345,
) -> np.ndarray:
    """
    Helper: simulate one long single-month time series and return the
    inferred sunny/cloudy indicator for each day.

    A sunny day is detected by the fact that production was multiplied by
    ``sunny_factor`` (=2.0 in the helper) instead of ``cloudy_factor`` (=1.0).
    """
    params = [
        SolarMonthParams(
            avg_daily_kwh_per_kwp=1.0,
            p_sunny=p_sunny,
            sunny_factor=2.0,
            cloudy_factor=1.0,
            weather_persistence=weather_persistence,
        )
        for _ in range(12)
    ]
    # All days belong to the same (year=0, month=0) to keep the chain running.
    month_in_year_for_day = np.zeros(n_days, dtype=int)
    year_index_for_day = np.zeros(n_days, dtype=int)

    model = SolarModel(pv_kwp=1.0, month_params=params, degradation_per_year=0.0)
    rng = np.random.default_rng(seed)
    energy = model.simulate_daily_energy(
        n_years=1,
        month_in_year_for_day=month_in_year_for_day,
        year_index_for_day=year_index_for_day,
        rng=rng,
    )
    # sunny_factor=2.0, cloudy_factor=1.0, base=1.0 → energy>1.5 → sunny
    return energy > 1.5


def test_markov_chain_preserves_marginal_p_sunny() -> None:
    """
    The two-state Markov chain is built so that, irrespective of the
    persistence parameter, the stationary distribution of sunny days equals
    the configured ``p_sunny``. Verify this property empirically.
    """
    rng_seed = 42
    n_days = 80_000
    p_sunny_target = 0.6

    for persistence in (0.0, 0.3, 0.7):
        states = _simulate_sunny_states(
            p_sunny=p_sunny_target,
            weather_persistence=persistence,
            n_days=n_days,
            seed=rng_seed,
        )
        empirical = states.mean()
        # 3-sigma tolerance for Bernoulli with p=0.6, n=80000:
        # std = sqrt(0.6*0.4/80000) ≈ 0.0017 → tol ~ 0.005
        assert abs(empirical - p_sunny_target) < 0.01, (
            f"persistence={persistence}: marginal {empirical:.4f} "
            f"deviates too much from target {p_sunny_target}"
        )


def test_markov_chain_persistence_produces_autocorrelation() -> None:
    """
    Increasing ``weather_persistence`` must increase the lag-1
    autocorrelation of the sunny/cloudy indicator. With persistence=0 the
    chain is iid and the autocorrelation should be ~0; with persistence=0.8
    it should be clearly positive.
    """
    n_days = 60_000

    def lag1_autocorr(x: np.ndarray) -> float:
        x = x.astype(float)
        x = x - x.mean()
        denom = (x * x).sum()
        if denom == 0:
            return 0.0
        return float((x[:-1] * x[1:]).sum() / denom)

    states_iid = _simulate_sunny_states(0.6, 0.0, n_days=n_days, seed=7)
    states_mid = _simulate_sunny_states(0.6, 0.4, n_days=n_days, seed=7)
    states_hi = _simulate_sunny_states(0.6, 0.8, n_days=n_days, seed=7)

    ac_iid = lag1_autocorr(states_iid)
    ac_mid = lag1_autocorr(states_mid)
    ac_hi = lag1_autocorr(states_hi)

    # The theoretical lag-1 autocorrelation of this chain equals exactly
    # the persistence parameter, so we expect each value to be close to it.
    assert abs(ac_iid - 0.0) < 0.02
    assert abs(ac_mid - 0.4) < 0.03
    assert abs(ac_hi - 0.8) < 0.03


def test_markov_chain_defaults_to_iid_when_persistence_omitted() -> None:
    """
    ``SolarMonthParams.weather_persistence`` defaults to 0.0. When omitted
    the chain must reproduce the legacy iid Bernoulli behaviour, i.e. the
    lag-1 autocorrelation must be statistically indistinguishable from 0.
    """
    params = [SolarMonthParams(1.0, 0.5, 2.0, 1.0) for _ in range(12)]
    model = SolarModel(pv_kwp=1.0, month_params=params, degradation_per_year=0.0)

    n_days = 60_000
    rng = np.random.default_rng(2024)
    energy = model.simulate_daily_energy(
        n_years=1,
        month_in_year_for_day=np.zeros(n_days, dtype=int),
        year_index_for_day=np.zeros(n_days, dtype=int),
        rng=rng,
    )
    sunny = (energy > 1.5).astype(float)
    centred = sunny - sunny.mean()
    denom = (centred * centred).sum()
    autocorr = float((centred[:-1] * centred[1:]).sum() / denom)

    assert abs(autocorr) < 0.02
