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


# ---------------------------------------------------------------------------
# Stochastic price models (Phase 2 of the roadmap)
# ---------------------------------------------------------------------------

from sim_stochastic_pv.simulation.prices import (
    GBMPriceModel,
    MeanRevertingPriceModel,
)


def _collect_gbm_terminal_prices(
    model: GBMPriceModel,
    n_years: int,
    n_paths: int,
    base_seed: int = 0,
) -> np.ndarray:
    """Run ``n_paths`` GBM resets and return terminal-month prices."""
    finals = np.empty(n_paths, dtype=float)
    for i in range(n_paths):
        model.reset_for_run(rng=np.random.default_rng(base_seed + i), n_years=n_years)
        finals[i] = model.get_price(n_years - 1, 11)
    return finals


def test_gbm_with_zero_volatility_is_deterministic() -> None:
    """
    With volatility = 0 the GBM degenerates to ``P_t = P_0 · exp(μ · t)``.
    Verify against the closed-form for both intermediate and terminal months.
    """
    mu = 0.025
    p0 = 0.20
    n_years = 10
    model = GBMPriceModel(
        base_price_eur_per_kwh=p0,
        drift_annual=mu,
        volatility_annual=0.0,
    )
    model.reset_for_run(rng=np.random.default_rng(0), n_years=n_years)

    # Final month (year 9, month 11) → t = (9 * 12 + 12) / 12 = 10 years
    expected_final = p0 * np.exp(mu * n_years)
    actual_final = model.get_price(n_years - 1, 11)
    assert actual_final == pytest.approx(expected_final, rel=1e-9)

    # Intermediate month (year 5, month 0) → t = 5 + 0/12 = 5 + 1/12 years
    # (the path point at (y=5, m=0) is the (5*12+0+1)=61st step → 61/12 years)
    expected_y5 = p0 * np.exp(mu * 61.0 / 12.0)
    actual_y5 = model.get_price(5, 0)
    assert actual_y5 == pytest.approx(expected_y5, rel=1e-9)


def test_gbm_mean_path_matches_theoretical_expectation() -> None:
    """
    Across many paths, the Monte Carlo average of P_t must converge to the
    theoretical expectation ``E[P_t] = P_0 · exp(μ · t)``.
    The Itō correction ``-σ²/2`` is exactly what makes this hold.
    """
    p0 = 0.25
    mu = 0.04
    sigma = 0.20  # large σ amplifies the test
    n_years = 5
    n_paths = 5000

    model = GBMPriceModel(
        base_price_eur_per_kwh=p0,
        drift_annual=mu,
        volatility_annual=sigma,
    )
    finals = _collect_gbm_terminal_prices(model, n_years, n_paths, base_seed=1000)

    expected = p0 * np.exp(mu * n_years)
    empirical = finals.mean()
    # SE of the mean: σ_path / sqrt(n) where σ_path = expected*sqrt(exp(σ²t)-1)
    se_factor = np.sqrt(np.exp(sigma * sigma * n_years) - 1.0)
    se = expected * se_factor / np.sqrt(n_paths)
    assert abs(empirical - expected) < 4.0 * se, (
        f"Empirical mean {empirical:.5f} too far from theoretical {expected:.5f} "
        f"(SE={se:.5f})"
    )


def test_gbm_log_variance_grows_linearly_with_time() -> None:
    """
    The key property of GBM: ``Var[log P_t] = σ² · t``.
    Verify by sampling at two different horizons and checking the ratio.
    """
    sigma = 0.15
    p0 = 0.20

    def empirical_log_var(n_years: int, n_paths: int) -> float:
        model = GBMPriceModel(
            base_price_eur_per_kwh=p0,
            drift_annual=0.0,         # drift irrelevant for variance
            volatility_annual=sigma,
        )
        finals = _collect_gbm_terminal_prices(model, n_years, n_paths, base_seed=42)
        return float(np.var(np.log(finals), ddof=1))

    n_paths = 6000
    var_5y = empirical_log_var(5, n_paths)
    var_20y = empirical_log_var(20, n_paths)

    expected_5y = sigma * sigma * 5
    expected_20y = sigma * sigma * 20

    # 10% tolerance for empirical variance estimates with n=6000
    assert var_5y == pytest.approx(expected_5y, rel=0.10)
    assert var_20y == pytest.approx(expected_20y, rel=0.10)


def test_gbm_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        GBMPriceModel(base_price_eur_per_kwh=-0.10)
    with pytest.raises(ValueError):
        GBMPriceModel(volatility_annual=-0.01)
    with pytest.raises(ValueError):
        GBMPriceModel(seasonal_factors=[1.0] * 11)
    with pytest.raises(ValueError):
        GBMPriceModel(seasonal_factors=[1.0] * 6 + [-0.5] + [1.0] * 5)


def test_ou_reverts_to_long_term_price() -> None:
    """
    For long horizons and finite κ, the mean of the OU process converges to
    the equilibrium ``long_term_price``. Average across many paths.
    """
    p0 = 0.20
    long_term = 0.30
    kappa = 0.5    # ≈ 1.4 years half-life
    sigma = 0.10
    n_years = 25  # ~18 half-lives → effectively stationary
    n_paths = 2000

    model = MeanRevertingPriceModel(
        base_price_eur_per_kwh=p0,
        long_term_price_eur_per_kwh=long_term,
        mean_reversion_speed_annual=kappa,
        volatility_annual=sigma,
    )
    finals = np.empty(n_paths)
    for i in range(n_paths):
        model.reset_for_run(rng=np.random.default_rng(5000 + i), n_years=n_years)
        finals[i] = model.get_price(n_years - 1, 11)

    # Stationary mean of log P is log(long_term).
    # Stationary mean of P is long_term · exp(σ²/(4κ)) due to log-normality.
    stationary_mean_logprice = np.log(long_term)
    log_finals = np.log(finals)
    se = np.std(log_finals, ddof=1) / np.sqrt(n_paths)
    assert abs(log_finals.mean() - stationary_mean_logprice) < 5.0 * se


def test_ou_log_variance_is_bounded() -> None:
    """
    Unlike GBM, the OU log-variance saturates at ``σ²/(2κ)`` for large t.
    Verify the empirical 20-year variance is close to that bound, NOT
    growing linearly like ``σ² · t``.
    """
    kappa = 0.5
    sigma = 0.10
    p0 = 0.25
    n_paths = 3000

    def empirical_log_var(n_years: int) -> float:
        model = MeanRevertingPriceModel(
            base_price_eur_per_kwh=p0,
            long_term_price_eur_per_kwh=p0,
            mean_reversion_speed_annual=kappa,
            volatility_annual=sigma,
        )
        finals = np.empty(n_paths)
        for i in range(n_paths):
            model.reset_for_run(rng=np.random.default_rng(9000 + i), n_years=n_years)
            finals[i] = model.get_price(n_years - 1, 11)
        return float(np.var(np.log(finals), ddof=1))

    stationary_var = sigma * sigma / (2.0 * kappa)  # = 0.01
    naive_gbm_var_20y = sigma * sigma * 20          # = 0.20

    var_20y = empirical_log_var(20)

    # Empirical variance should be near the stationary bound, NOT the
    # linearly-growing GBM value: enforce both directions.
    assert var_20y == pytest.approx(stationary_var, rel=0.20)
    assert var_20y < 0.30 * naive_gbm_var_20y, (
        "OU log-variance should be much smaller than the naive GBM upper "
        f"bound but got {var_20y:.4f} vs {naive_gbm_var_20y:.4f}"
    )


def test_ou_rejects_zero_mean_reversion() -> None:
    """``κ = 0`` is rejected with a hint to use GBMPriceModel instead."""
    with pytest.raises(ValueError, match="GBMPriceModel"):
        MeanRevertingPriceModel(
            base_price_eur_per_kwh=0.25,
            long_term_price_eur_per_kwh=0.25,
            mean_reversion_speed_annual=0.0,
            volatility_annual=0.10,
        )


# ---------------------------------------------------------------------------
# Scenario builder dispatcher tests
# ---------------------------------------------------------------------------

from sim_stochastic_pv.scenario_builder import build_default_price_model
from sim_stochastic_pv.simulation.prices import EscalatingPriceModel


def _scenario_with_price(price_block: dict) -> dict:
    """Wrap a price block into the minimal scenario_data dict required by
    ``build_default_price_model`` (it only reads the ``price`` key)."""
    return {"price": price_block}


def test_dispatcher_defaults_to_escalating_model() -> None:
    model = build_default_price_model(
        scenario_data=_scenario_with_price(
            {"base_price_eur_per_kwh": 0.22, "annual_escalation": 0.02}
        )
    )
    assert isinstance(model, EscalatingPriceModel)


def test_dispatcher_returns_gbm_when_requested() -> None:
    model = build_default_price_model(
        scenario_data=_scenario_with_price(
            {
                "model_type": "gbm",
                "base_price_eur_per_kwh": 0.25,
                "drift_annual": 0.03,
                "volatility_annual": 0.12,
            }
        )
    )
    assert isinstance(model, GBMPriceModel)
    assert model.drift_annual == pytest.approx(0.03)
    assert model.volatility_annual == pytest.approx(0.12)


def test_dispatcher_returns_mean_reverting() -> None:
    model = build_default_price_model(
        scenario_data=_scenario_with_price(
            {
                "model_type": "mean_reverting",
                "base_price_eur_per_kwh": 0.20,
                "long_term_price_eur_per_kwh": 0.30,
                "mean_reversion_speed_annual": 0.4,
                "volatility_annual": 0.10,
            }
        )
    )
    assert isinstance(model, MeanRevertingPriceModel)
    assert model.long_term_price == pytest.approx(0.30)
    assert model.mean_reversion_speed_annual == pytest.approx(0.4)


def test_dispatcher_raises_on_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unknown price model_type"):
        build_default_price_model(
            scenario_data=_scenario_with_price({"model_type": "garch"})
        )


# ---------------------------------------------------------------------------
# Phase 3 — price paths in Monte Carlo results
# ---------------------------------------------------------------------------

from sim_stochastic_pv.simulation.monte_carlo import MonteCarloSimulator
from sim_stochastic_pv.simulation.energy_simulator import (
    EnergySystemConfig,
    EnergySystemSimulator,
)
from sim_stochastic_pv.simulation.battery import BatterySpecs as _BS
from sim_stochastic_pv.simulation.load_profiles import (
    MonthlyAverageLoadProfile,
    make_flat_monthly_load_profiles,
)
from sim_stochastic_pv.simulation.monte_carlo import EconomicConfig


def _make_minimal_mc(n_years: int = 2, n_mc: int = 10) -> MonteCarloSimulator:
    """Tiny Monte Carlo setup used by Phase 3 tests."""
    solar = SolarModel(
        pv_kwp=2.0,
        month_params=[SolarMonthParams(3.0, 0.6, 1.2, 0.3) for _ in range(12)],
        degradation_per_year=0.0,
    )
    load = MonthlyAverageLoadProfile(make_flat_monthly_load_profiles(200.0))
    energy_cfg = EnergySystemConfig(
        n_years=n_years,
        pv_kwp=2.0,
        battery_specs=_BS(capacity_kwh=2.0, cycles_life=5000),
        n_batteries=0,
        inverter_p_ac_max_kw=2.0,
    )
    energy_sim = EnergySystemSimulator(
        config=energy_cfg, solar_model=solar, load_profile=load
    )
    return MonteCarloSimulator(
        energy_simulator=energy_sim,
        price_model=GBMPriceModel(
            base_price_eur_per_kwh=0.25,
            drift_annual=0.025,
            volatility_annual=0.15,
        ),
        economic_config=EconomicConfig(investment_eur=5000, n_mc=n_mc),
    )


def test_monte_carlo_results_expose_price_paths() -> None:
    """Phase 3.1: MonteCarloResults must contain df_price and price_paths."""
    n_years, n_mc = 2, 8
    mc = _make_minimal_mc(n_years=n_years, n_mc=n_mc)
    results = mc.run(seed=7, show_progress=False)

    assert results.df_price is not None
    expected_cols = {
        "month_index",
        "year",
        "month_in_year",
        "price_mean_eur_per_kwh",
        "price_p05_eur_per_kwh",
        "price_p95_eur_per_kwh",
    }
    assert expected_cols.issubset(set(results.df_price.columns))
    assert len(results.df_price) == n_years * 12

    paths = results.price_paths_eur_per_kwh
    assert paths.shape == (n_mc, n_years * 12)
    # All prices strictly positive
    assert (paths > 0).all()


def test_monte_carlo_price_band_consistent_with_paths() -> None:
    """
    The df_price aggregates must match the per-path arrays statistically:
    mean ≈ paths.mean(axis=0), p05 < mean < p95 once volatility is non-zero.
    """
    mc = _make_minimal_mc(n_years=3, n_mc=80)
    results = mc.run(seed=11, show_progress=False)

    mean_from_paths = results.price_paths_eur_per_kwh.mean(axis=0)
    mean_from_df = results.df_price["price_mean_eur_per_kwh"].values
    np.testing.assert_allclose(mean_from_df, mean_from_paths, rtol=1e-12)

    p05 = results.df_price["price_p05_eur_per_kwh"].values
    p95 = results.df_price["price_p95_eur_per_kwh"].values
    # First month: very narrow band but already > 0
    assert (p95 >= p05).all()
    # By the last month the band must have widened materially
    assert (p95[-1] - p05[-1]) > (p95[0] - p05[0]) * 1.5


def test_application_summary_exposes_price_block() -> None:
    """
    Phase 3.2: the application response must include a `price` block in
    plots_data with mean/p05/p95 + sample_paths capped to 20 entries.
    """
    from sim_stochastic_pv.application import _build_price_plot_payload

    mc = _make_minimal_mc(n_years=2, n_mc=40)
    results = mc.run(seed=3, show_progress=False)

    payload = _build_price_plot_payload(results)
    assert set(payload.keys()) == {
        "months",
        "mean_eur_per_kwh",
        "p05_eur_per_kwh",
        "p95_eur_per_kwh",
        "sample_paths",
    }
    assert len(payload["months"]) == 24
    # Sample paths capped to the documented limit (20)
    assert 1 <= len(payload["sample_paths"]) <= 20
    # Each sample path covers the whole horizon
    for path in payload["sample_paths"]:
        assert len(path) == 24


# ---------------------------------------------------------------------------
# Phase 4 — break-even KPIs in application summary
# ---------------------------------------------------------------------------


def test_application_summary_exposes_break_even_kpis() -> None:
    """
    Phase 4.1: ``SimulationApplication.run_analysis`` must include the
    break-even KPI keys in the top-level summary dict.

    Checked keys:
        - ``prob_break_even_within_horizon``  – fraction of paths in [0, 1]
        - ``break_even_month_median``          – None or non-negative float
        - ``break_even_month_p05``             – None or non-negative float
        - ``break_even_month_p95``             – None or non-negative float
        - ``npv_median_eur``                   – finite float
        - ``irr_mean``                         – None or finite float
    """
    from sim_stochastic_pv.application import SimulationApplication

    app = SimulationApplication(save_outputs=False, persistence=None)
    scenario = {
        "scenario_name": "phase4_test",
        "load_profile": {"home_profile_type": "arera"},
        "solar": {
            "solar_profile_id": None,
            "pv_kwp": 2.0,
            "degradation_per_year": 0.007,
        },
        "energy": {
            "n_years": 3,
            "pv_kwp": 2.0,
            "battery_specs": {"capacity_kwh": 1.0, "cycles_life": 1000},
            "n_batteries": 0,
            "inverter_p_ac_max_kw": 2.0,
        },
        "price": {
            "base_price_eur_per_kwh": 0.25,
            "annual_escalation": 0.02,
            "use_stochastic_escalation": False,
        },
        "economic": {"investment_eur": 5000.0, "n_mc": 10},
    }
    summary = app.run_analysis(scenario_data=scenario, seed=42)

    # Top-level break-even KPIs must be present
    assert "prob_break_even_within_horizon" in summary
    assert "break_even_month_median" in summary
    assert "break_even_month_p05" in summary
    assert "break_even_month_p95" in summary
    assert "npv_median_eur" in summary
    assert "irr_mean" in summary

    # Value ranges
    pbe = summary["prob_break_even_within_horizon"]
    if pbe is not None:
        assert 0.0 <= pbe <= 1.0

    npv = summary["npv_median_eur"]
    assert npv is None or (isinstance(npv, float) and np.isfinite(npv))

    irr = summary["irr_mean"]
    assert irr is None or (isinstance(irr, float) and np.isfinite(irr))


def test_profit_plots_data_contains_break_even_fields() -> None:
    """
    Phase 4.2: the ``plots_data.profit`` block must include the three
    break-even annotation fields used by the Dashboard chart plugin.

    Checked keys:
        - ``break_even_month_median``
        - ``break_even_month_p05``
        - ``break_even_month_p95``
    """
    from sim_stochastic_pv.application import SimulationApplication

    app = SimulationApplication(save_outputs=False, persistence=None)
    scenario = {
        "scenario_name": "phase4_chart_test",
        "load_profile": {"home_profile_type": "arera"},
        "solar": {
            "solar_profile_id": None,
            "pv_kwp": 2.0,
            "degradation_per_year": 0.007,
        },
        "energy": {
            "n_years": 2,
            "pv_kwp": 2.0,
            "battery_specs": {"capacity_kwh": 1.0, "cycles_life": 1000},
            "n_batteries": 0,
            "inverter_p_ac_max_kw": 2.0,
        },
        "price": {
            "base_price_eur_per_kwh": 0.25,
            "annual_escalation": 0.0,
            "use_stochastic_escalation": False,
        },
        "economic": {"investment_eur": 100.0, "n_mc": 10},
    }
    summary = app.run_analysis(scenario_data=scenario, seed=7)

    profit_block = summary["plots_data"]["profit"]
    assert "break_even_month_median" in profit_block
    assert "break_even_month_p05" in profit_block
    assert "break_even_month_p95" in profit_block


# ---------------------------------------------------------------------------
# Phase 8 — load profile with kind:"home_away" + scenario-level min/max days
# ---------------------------------------------------------------------------

from sim_stochastic_pv.scenario_builder import build_default_load_profile


def test_load_profile_home_away_shape_builds_home_away_profile() -> None:
    """
    The new ``kind: 'home_away'`` schema must produce a HomeAwayLoadProfile
    whose home/away sides are populated from the sub-profile dicts.
    """
    from sim_stochastic_pv.simulation.load_profiles import HomeAwayLoadProfile

    scenario = {
        "load_profile": {
            "kind": "home_away",
            "home": {"monthly_w": [300] * 12},
            "away": {"monthly_w": [100] * 12},
        },
        "min_days_home": [20] * 12,
        "max_days_home": [25] * 12,
    }
    profile = build_default_load_profile(scenario)
    assert isinstance(profile, HomeAwayLoadProfile)
    # The min/max days arrays must flow through from scenario root.
    assert list(profile.min_days_home) == [20] * 12
    assert list(profile.max_days_home) == [25] * 12


def test_load_profile_home_away_supports_arera_sides() -> None:
    """
    ARERA can appear as either side of a home_away profile.

    The blueprint wraps each side with ``VariableLoadProfile`` (default
    ±10%/±5% variation), so the test must look at the base profile inside
    the wrapper rather than the side directly.
    """
    from sim_stochastic_pv.simulation.load_profiles import (
        AreraLoadProfile,
        HomeAwayLoadProfile,
        VariableLoadProfile,
    )

    scenario = {
        "load_profile": {
            "kind": "home_away",
            "home": {"monthly_24h_w": [[200] * 24] * 12},
            "away": {"type": "arera"},
        },
        "min_days_home": [22] * 12,
        "max_days_home": [22] * 12,
    }
    profile = build_default_load_profile(scenario)
    assert isinstance(profile, HomeAwayLoadProfile)

    def unwrap(p):
        return p.base_profile if isinstance(p, VariableLoadProfile) else p

    assert isinstance(unwrap(profile.away_profile), AreraLoadProfile)


def test_load_profile_legacy_shape_still_works() -> None:
    """Legacy inline scenario form must not be broken by the new schema."""
    from sim_stochastic_pv.simulation.load_profiles import HomeAwayLoadProfile

    scenario = {
        "load_profile": {
            "home_profile_type": "custom",
            "home_profiles_w": [250] * 12,
            "away_profile": "arera",
            "min_days_home": [18] * 12,
            "max_days_home": [22] * 12,
        }
    }
    profile = build_default_load_profile(scenario)
    assert isinstance(profile, HomeAwayLoadProfile)
    assert list(profile.min_days_home) == [18] * 12


# ---------------------------------------------------------------------------
# Phase 8 single-sided DB profiles — bug fix 2026-05-28
# ---------------------------------------------------------------------------


def test_load_profile_db_saved_custom_monthly_w_does_not_crash() -> None:
    """
    Bug regression — a saved DB load profile of UI type "custom" stores
    only ``monthly_w`` at root (no ``kind`` field). Before the fix this
    fell into the legacy branch and crashed with
    ``Missing 'home_profiles_w' for custom profile``. After the fix the
    Phase-8 single-side detection routes it through the standard
    sub-profile factory with ARERA as the away side.
    """
    from sim_stochastic_pv.simulation.load_profiles import HomeAwayLoadProfile

    scenario = {
        "load_profile": {"monthly_w": [250] * 12},
        "min_days_home": [18] * 12,
        "max_days_home": [22] * 12,
    }
    profile = build_default_load_profile(scenario)
    assert isinstance(profile, HomeAwayLoadProfile)


def test_load_profile_db_saved_custom_24h_monthly_matrix() -> None:
    """``monthly_24h_w`` saved as a single-sided DB profile must build."""
    from sim_stochastic_pv.simulation.load_profiles import HomeAwayLoadProfile

    scenario = {
        "load_profile": {"monthly_24h_w": [[300] * 24] * 12},
        "min_days_home": [15] * 12,
        "max_days_home": [20] * 12,
    }
    profile = build_default_load_profile(scenario)
    assert isinstance(profile, HomeAwayLoadProfile)


def test_load_profile_db_saved_arera_only_uses_arera_both_sides() -> None:
    """Saved "ARERA" profile (type:arera at root) → ARERA home, ARERA away."""
    from sim_stochastic_pv.simulation.load_profiles import (
        AreraLoadProfile,
        HomeAwayLoadProfile,
        VariableLoadProfile,
    )

    scenario = {
        "load_profile": {"type": "arera"},
        "min_days_home": [10] * 12,
        "max_days_home": [15] * 12,
    }
    profile = build_default_load_profile(scenario)
    assert isinstance(profile, HomeAwayLoadProfile)

    def unwrap(p):
        return p.base_profile if isinstance(p, VariableLoadProfile) else p

    assert isinstance(unwrap(profile.home_profile), AreraLoadProfile)
    assert isinstance(unwrap(profile.away_profile), AreraLoadProfile)


def test_load_profile_home_away_rejects_missing_side() -> None:
    """Missing ``home`` or ``away`` sub-profile must raise a clear error."""
    import pytest as _pt

    bad = {"load_profile": {"kind": "home_away", "home": {"type": "arera"}}}
    with _pt.raises(ValueError, match="home_away"):
        build_default_load_profile(bad)


def test_load_profile_scenario_level_min_days_overrides_load_block() -> None:
    """
    When ``min_days_home`` exists both at scenario root and inside the
    ``load_profile`` block, the scenario-level value wins (new mental model).
    """
    scenario = {
        "load_profile": {
            "kind": "home_away",
            "home": {"monthly_w": [300] * 12},
            "away": {"monthly_w": [100] * 12},
            "min_days_home": [10] * 12,  # would be picked by legacy logic
            "max_days_home": [12] * 12,
        },
        "min_days_home": [25] * 12,  # scenario root — should win
        "max_days_home": [27] * 12,
    }
    profile = build_default_load_profile(scenario)
    assert list(profile.min_days_home) == [25] * 12
    assert list(profile.max_days_home) == [27] * 12


def test_load_profile_hydration_end_to_end_with_home_away(persistence) -> None:
    """
    End-to-end: save a kind:'home_away' profile in the DB, reference it from
    a scenario via load_profile_id, hydrate, build → must produce a
    HomeAwayLoadProfile with both sides correctly wired.
    """
    from sim_stochastic_pv.simulation.load_profiles import (
        AreraLoadProfile,
        HomeAwayLoadProfile,
        MonthlyAverageLoadProfile,
        VariableLoadProfile,
    )

    record = persistence.upsert_load_profile(
        name="Phase8 Home+Away",
        profile_type="home_away",
        data={
            "kind": "home_away",
            "home": {"monthly_w": [350] * 12},
            "away": {"type": "arera"},
        },
    )

    scenario_with_id = {
        "load_profile_id": record.id,
        "min_days_home": [21] * 12,
        "max_days_home": [24] * 12,
    }
    hydrated = persistence.hydrate_scenario_from_ids(scenario_with_id)

    # The hydration step injects load_profile.data into scenario["load_profile"]
    assert hydrated["load_profile"]["kind"] == "home_away"
    # min_days_home survives at scenario root
    assert hydrated["min_days_home"] == [21] * 12

    profile = build_default_load_profile(hydrated)
    assert isinstance(profile, HomeAwayLoadProfile)

    def unwrap(p):
        return p.base_profile if isinstance(p, VariableLoadProfile) else p

    assert isinstance(unwrap(profile.home_profile), MonthlyAverageLoadProfile)
    assert isinstance(unwrap(profile.away_profile), AreraLoadProfile)
    assert list(profile.min_days_home) == [21] * 12
    assert list(profile.max_days_home) == [24] * 12


def test_application_run_analysis_with_load_profile_id(persistence) -> None:
    """
    The high-level ``SimulationApplication.run_analysis`` must transparently
    hydrate ``load_profile_id`` references coming from a saved DB profile.
    This guards the regression that surfaced during the first end-to-end
    smoke test of Phase 8.
    """
    from sim_stochastic_pv.application import SimulationApplication

    record = persistence.upsert_load_profile(
        name="Phase8 App-level",
        profile_type="home_away",
        data={
            "kind": "home_away",
            "home": {"monthly_w": [300] * 12},
            "away": {"type": "arera"},
        },
    )

    scenario = {
        "scenario_name": "phase8_app_smoke",
        "load_profile_id": record.id,
        "min_days_home": [25] * 12,
        "max_days_home": [28] * 12,
        "solar": {
            "pv_kwp": 2.0,
            "month_params": [
                {
                    "avg_daily_kwh_per_kwp": 3.0,
                    "p_sunny": 0.5,
                    "sunny_factor": 1.1,
                    "cloudy_factor": 0.4,
                }
            ]
            * 12,
        },
        "energy": {
            "n_years": 1,
            "pv_kwp": 2.0,
            "battery_specs": {"capacity_kwh": 1.0, "cycles_life": 1000},
            "n_batteries": 0,
            "inverter_p_ac_max_kw": 2.0,
        },
        "price": {
            "model_type": "gbm",
            "base_price_eur_per_kwh": 0.25,
            "drift_annual": 0.02,
            "volatility_annual": 0.05,
        },
        "economic": {"investment_eur": 1000.0, "n_mc": 5},
    }

    app = SimulationApplication(
        save_outputs=False, persistence=persistence, result_builder=None
    )
    summary = app.run_analysis(scenario_data=scenario, n_mc=5)

    # The summary structure exposes the Phase 3 price block:
    assert "plots_data" in summary
    assert "price" in summary["plots_data"]
    # Final gain is finite (we don't assert sign — too few paths for stability)
    assert summary["final_gain_mean_eur"] is not None


# ---------------------------------------------------------------------------
# Phase 10 — price preview helper
# ---------------------------------------------------------------------------

from sim_stochastic_pv.simulation.prices import (
    PricePreviewResult,
    simulate_price_preview,
)


def test_price_preview_shape_and_band_growth() -> None:
    """
    GBM preview with non-zero volatility produces:
    - the expected dataclass shape with 12*n_years months,
    - a band that strictly widens over time (Var[log P] ~ σ² t).
    """
    model = GBMPriceModel(
        base_price_eur_per_kwh=0.25,
        drift_annual=0.025,
        volatility_annual=0.15,
    )
    preview = simulate_price_preview(model, n_years=5, n_paths=300, seed=1)
    assert isinstance(preview, PricePreviewResult)
    assert len(preview.months) == 60
    assert len(preview.mean_eur_per_kwh) == 60
    assert 1 <= len(preview.sample_paths) <= 20

    width_first = preview.p95_eur_per_kwh[0] - preview.p05_eur_per_kwh[0]
    width_last = preview.p95_eur_per_kwh[-1] - preview.p05_eur_per_kwh[-1]
    assert width_last > 2.0 * width_first


def test_price_preview_deterministic_when_volatility_zero() -> None:
    """When σ=0 every path collapses on the deterministic exp(μ t) curve."""
    model = GBMPriceModel(
        base_price_eur_per_kwh=0.20,
        drift_annual=0.02,
        volatility_annual=0.0,
    )
    preview = simulate_price_preview(model, n_years=3, n_paths=50, seed=0)
    # Bands coincide with the mean within float tolerance
    for m, p05, p95 in zip(
        preview.mean_eur_per_kwh,
        preview.p05_eur_per_kwh,
        preview.p95_eur_per_kwh,
    ):
        assert p05 == pytest.approx(m, rel=1e-12)
        assert p95 == pytest.approx(m, rel=1e-12)
    # And every sample path equals the mean curve
    for path in preview.sample_paths:
        assert path == pytest.approx(preview.mean_eur_per_kwh, rel=1e-12)


def test_price_preview_seed_determinism() -> None:
    """Identical seeds → identical previews (full byte-level reproducibility)."""
    model_a = GBMPriceModel(volatility_annual=0.10)
    model_b = GBMPriceModel(volatility_annual=0.10)
    a = simulate_price_preview(model_a, n_years=2, n_paths=30, seed=12345)
    b = simulate_price_preview(model_b, n_years=2, n_paths=30, seed=12345)
    assert a.mean_eur_per_kwh == b.mean_eur_per_kwh
    assert a.p05_eur_per_kwh == b.p05_eur_per_kwh
    assert a.sample_paths == b.sample_paths


# ---------------------------------------------------------------------------
# Phase 9 — simplified sizing
# ---------------------------------------------------------------------------

from sim_stochastic_pv.scenario_builder import (
    simplified_panel_count,
    build_default_optimization_request,
)


def test_simplified_panel_count_basic_arithmetic() -> None:
    """5 kW inverter @ +20 % needs 6 kW DC; with 400 W panels → 15 panels."""
    assert simplified_panel_count(5.0, 400.0, 0.20) == 15
    # With 540 W panels we expect ceil(6000 / 540) = 12 (slightly above 6 kW)
    assert simplified_panel_count(5.0, 540.0, 0.20) == 12


def test_simplified_panel_count_rounds_up() -> None:
    """
    The count must always be the *smallest* count that satisfies the
    constraint — so even a tiny shortfall triggers one more panel.
    """
    # 3 kW × 1.20 = 3600 W. With 500 W panels: ceil(3600/500) = 8.
    assert simplified_panel_count(3.0, 500.0, 0.20) == 8


def test_simplified_panel_count_minimum_one_panel() -> None:
    """Even with absurdly small targets the helper returns at least 1."""
    assert simplified_panel_count(0.1, 600.0, 0.0) == 1


def test_simplified_panel_count_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        simplified_panel_count(0.0, 400.0)
    with pytest.raises(ValueError):
        simplified_panel_count(5.0, 0.0)
    with pytest.raises(ValueError):
        simplified_panel_count(5.0, 400.0, target_dc_overcapacity_pct=-0.1)


def test_optimization_simplified_mode_overrides_panel_counts() -> None:
    """
    Phase 9: when ``sizing_mode == 'simplified'`` the request's
    ``panel_count_options`` is derived from the inverter/panel pairs and
    the user-supplied count list is ignored.
    """
    cfg = {
        "scenario_name": "phase9_simplified",
        "optimization": {
            "sizing_mode": "simplified",
            "target_dc_overcapacity_pct": 0.20,
            # Ignored on purpose — proves the override works.
            "panel_count_options": [99],
            "inverter_options": [
                {"name": "Inv5", "p_ac_max_kw": 5.0, "price_eur": 1500.0},
                {"name": "Inv6", "p_ac_max_kw": 6.0, "price_eur": 1700.0},
            ],
            "panel_options": [
                {"name": "P400", "power_w": 400.0, "price_eur": 120.0},
            ],
            "battery_options": [],
            "battery_count_options": [0],
            "include_no_battery": True,
        },
    }
    req = build_default_optimization_request(cfg)
    # Expected: Inv5/P400 → 15, Inv6/P400 → 18 → sorted union [15, 18]
    assert req.panel_count_options == [15, 18]
    assert 99 not in req.panel_count_options


def test_optimization_advanced_mode_passes_panel_counts_through() -> None:
    """In advanced mode the user-supplied list is honoured verbatim."""
    cfg = {
        "scenario_name": "phase9_advanced",
        "optimization": {
            "sizing_mode": "advanced",
            "panel_count_options": [10, 12, 14],
            "inverter_options": [
                {"name": "Inv5", "p_ac_max_kw": 5.0, "price_eur": 1500.0},
            ],
            "panel_options": [
                {"name": "P400", "power_w": 400.0, "price_eur": 120.0},
            ],
            "battery_options": [],
            "battery_count_options": [0],
            "include_no_battery": True,
        },
    }
    req = build_default_optimization_request(cfg)
    assert req.panel_count_options == [10, 12, 14]


# =============================================================================
# Phase 5 — WeeklyPatternLoadProfile
# =============================================================================

class TestPhase5WeeklyLoadProfile:
    """Tests for WeeklyPatternLoadProfile and scenario-builder integration."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flat_baseline_w(value_w: float = 200.0) -> np.ndarray:
        """Return a (12, 24) baseline array with constant value_w Watts."""
        return np.full((12, 24), value_w, dtype=float)

    @staticmethod
    def _asymmetric_pattern() -> np.ndarray:
        """
        Return a (7, 24) pattern where:
        - Mon (0): all values = 1.0  (average days)
        - Tue–Sun (1-6): all values = 1.0  (uniform → trivial normalization)
        For a non-trivial weekday test we use a custom per-test pattern.
        """
        return np.ones((7, 24), dtype=float)

    # ------------------------------------------------------------------
    # 1. Shape validation
    # ------------------------------------------------------------------

    def test_bad_baseline_shape_raises(self) -> None:
        """Constructor raises ValueError for baseline not shaped (12, 24)."""
        from sim_stochastic_pv.simulation.load_profiles import WeeklyPatternLoadProfile

        bad_baseline = np.ones((10, 24))
        pattern = np.ones((7, 24))
        with pytest.raises(ValueError, match=r"monthly_profiles_w.*shape"):
            WeeklyPatternLoadProfile(bad_baseline, pattern)

    def test_bad_pattern_shape_raises(self) -> None:
        """Constructor raises ValueError for pattern not shaped (7, 24)."""
        from sim_stochastic_pv.simulation.load_profiles import WeeklyPatternLoadProfile

        baseline = self._flat_baseline_w(200.0)
        bad_pattern = np.ones((5, 24))
        with pytest.raises(ValueError, match=r"weekly_pattern_w.*shape"):
            WeeklyPatternLoadProfile(baseline, bad_pattern)

    # ------------------------------------------------------------------
    # 2. Monthly average preservation (key invariant)
    # ------------------------------------------------------------------

    def test_weekly_mean_equals_baseline(self) -> None:
        """
        For every (month, hour) the mean load across all 7 weekdays equals the
        monthly baseline — i.e. the normalization preserves the energy budget.

        This is the core mathematical invariant of Phase 5.
        """
        from sim_stochastic_pv.simulation.load_profiles import (
            WeeklyPatternLoadProfile,
            WEEKLY_PRESETS,
        )

        baseline_w = self._flat_baseline_w(250.0)
        for preset_name, pattern in WEEKLY_PRESETS.items():
            load = WeeklyPatternLoadProfile(baseline_w, pattern)
            for m in range(12):
                for h in range(24):
                    weekly_mean = np.mean(
                        [load.get_hourly_load_kw(0, m, 0, h, d) for d in range(7)]
                    )
                    expected_kw = 250.0 / 1000.0
                    assert weekly_mean == pytest.approx(expected_kw, rel=1e-9), (
                        f"Preset '{preset_name}', month {m}, hour {h}: "
                        f"weekly mean {weekly_mean:.6f} ≠ baseline {expected_kw:.6f}"
                    )

    # ------------------------------------------------------------------
    # 3. Weekday/weekend distinction
    # ------------------------------------------------------------------

    def test_residential_typical_weekday_lt_weekend_daytime(self) -> None:
        """
        For 'residential_typical', weekday daytime load at hour 12 should be
        strictly less than Saturday load — the pattern captures that the family
        is away Mon–Fri during working hours.
        """
        from sim_stochastic_pv.simulation.load_profiles import (
            WeeklyPatternLoadProfile,
            WEEKLY_PRESETS,
        )

        baseline_w = self._flat_baseline_w(300.0)
        load = WeeklyPatternLoadProfile(baseline_w, WEEKLY_PRESETS["residential_typical"])

        # Monday noon (weekday, should be low)
        mon_noon = load.get_hourly_load_kw(0, 5, 0, 12, 0)   # weekday=0 Monday
        # Saturday noon (weekend, should be high)
        sat_noon = load.get_hourly_load_kw(0, 5, 5, 12, 5)   # weekday=5 Saturday

        assert mon_noon < sat_noon, (
            f"Expected weekday noon ({mon_noon:.3f} kW) < weekend noon ({sat_noon:.3f} kW)"
        )

    def test_commuter_weekday_daytime_lt_evening(self) -> None:
        """
        For 'commuter', weekday daytime (hour 12) load should be strictly less
        than weekday evening (hour 20) load — commuter returns late.
        """
        from sim_stochastic_pv.simulation.load_profiles import (
            WeeklyPatternLoadProfile,
            WEEKLY_PRESETS,
        )

        baseline_w = self._flat_baseline_w(300.0)
        load = WeeklyPatternLoadProfile(baseline_w, WEEKLY_PRESETS["commuter"])

        wed_noon    = load.get_hourly_load_kw(0, 5, 0, 12, 2)   # Wed, 12h
        wed_evening = load.get_hourly_load_kw(0, 5, 0, 20, 2)   # Wed, 20h

        assert wed_noon < wed_evening, (
            f"Commuter: noon ({wed_noon:.3f} kW) should be < evening ({wed_evening:.3f} kW)"
        )

    # ------------------------------------------------------------------
    # 4. Preset catalogue
    # ------------------------------------------------------------------

    def test_presets_have_correct_shape(self) -> None:
        """All entries in WEEKLY_PRESETS are (7, 24) float arrays."""
        from sim_stochastic_pv.simulation.load_profiles import WEEKLY_PRESETS

        expected_keys = {"residential_typical", "smart_worker", "commuter"}
        assert set(WEEKLY_PRESETS.keys()) == expected_keys

        for name, arr in WEEKLY_PRESETS.items():
            assert isinstance(arr, np.ndarray), f"Preset '{name}' must be ndarray"
            assert arr.shape == (7, 24), f"Preset '{name}' must be (7, 24), got {arr.shape}"
            assert arr.dtype == float, f"Preset '{name}' must be float dtype"
            assert (arr >= 0).all(), f"Preset '{name}' has negative values"

    # ------------------------------------------------------------------
    # 5. scenario_builder round-trip — sub-profile inside home_away
    # ------------------------------------------------------------------

    def test_builder_weekly_subprofile_in_home_away(self) -> None:
        """
        '_build_single_load_profile_factory' correctly instantiates a
        WeeklyPatternLoadProfile when home or away sub-profile has
        type='weekly' + monthly_w + weekly_pattern_w.
        """
        from sim_stochastic_pv.simulation.load_profiles import WeeklyPatternLoadProfile
        from sim_stochastic_pv.scenario_builder import build_default_load_profile

        pattern = [[float(d + h) for h in range(24)] for d in range(7)]
        # Add small offset to avoid zero columns
        pattern_shifted = [[v + 1.0 for v in row] for row in pattern]

        scenario = {
            "load_profile": {
                "kind": "home_away",
                "home": {
                    "type": "weekly",
                    "monthly_w": [200.0] * 12,
                    "weekly_pattern_w": pattern_shifted,
                },
                "away": {"type": "arera"},
            },
            "min_days_home": [20] * 12,
            "max_days_home": [25] * 12,
        }

        profile = build_default_load_profile(scenario)
        # Should not raise; result must be a LoadProfile
        assert profile is not None
        # Spot-check: calling get_hourly_load_kw with weekday works
        rng = np.random.default_rng(42)
        profile.reset_for_run(rng=rng, n_years=1)
        kw = profile.get_hourly_load_kw(0, 0, 0, 12, 3)
        assert kw >= 0.0

    # ------------------------------------------------------------------
    # 6. scenario_builder round-trip — standalone kind="weekly"
    # ------------------------------------------------------------------

    def test_builder_standalone_weekly_kind(self) -> None:
        """
        'build_default_load_profile' returns a WeeklyPatternLoadProfile directly
        when load_profile.kind == 'weekly'.
        """
        from sim_stochastic_pv.simulation.load_profiles import WeeklyPatternLoadProfile
        from sim_stochastic_pv.scenario_builder import build_default_load_profile
        from sim_stochastic_pv.simulation.load_profiles import WEEKLY_PRESETS

        scenario = {
            "load_profile": {
                "kind": "weekly",
                "type": "weekly",
                "monthly_w": [250.0] * 12,
                "weekly_pattern_w": WEEKLY_PRESETS["smart_worker"].tolist(),
            },
        }

        profile = build_default_load_profile(scenario)
        assert isinstance(profile, WeeklyPatternLoadProfile), (
            f"Expected WeeklyPatternLoadProfile, got {type(profile).__name__}"
        )
        # Verify baseline was stored
        assert profile.monthly_profiles_kw.shape == (12, 24)
        assert profile.weekly_pattern_w.shape == (7, 24)

    # ------------------------------------------------------------------
    # 7. Degenerate pattern (all zeros → passthrough)
    # ------------------------------------------------------------------

    def test_zero_column_produces_no_nan(self) -> None:
        """
        A column in weekly_pattern_w where all 7 values are 0 should produce
        exactly 0.0 kW (no load at that hour) rather than NaN.

        Explanation: when all pattern values for an hour are 0, the column mean
        is also 0.  The safe-mean fallback substitutes 1.0 to avoid division by
        zero, giving weight = 0/1 = 0.  The load is therefore baseline * 0 = 0.
        This is semantically correct: the user said "zero load at hour 5 for all
        days", so the result is indeed 0 — the baseline is not consulted because
        the user's intent is explicit.
        """
        from sim_stochastic_pv.simulation.load_profiles import WeeklyPatternLoadProfile

        baseline_w = self._flat_baseline_w(150.0)
        pattern = np.ones((7, 24), dtype=float)
        pattern[:, 5] = 0.0  # hour 5: all zeros

        load = WeeklyPatternLoadProfile(baseline_w, pattern)

        for d in range(7):
            kw = load.get_hourly_load_kw(0, 0, 0, 5, d)
            assert not np.isnan(kw), f"Weekday {d}, hour 5: expected finite value, got NaN"
            assert kw == pytest.approx(0.0, abs=1e-12), (
                f"Weekday {d}, hour 5 with zero column: expected 0.0 kW, got {kw}"
            )

        # Other hours remain unaffected (weight = 1.0 for all-ones column)
        kw_h12 = load.get_hourly_load_kw(0, 0, 0, 12, 3)
        assert kw_h12 == pytest.approx(150.0 / 1000.0, rel=1e-9)
