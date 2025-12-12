from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sim_stochastic_pv.energy_simulator import EnergySystemConfig
from sim_stochastic_pv.monte_carlo import EconomicConfig, MonteCarloResults
from sim_stochastic_pv.optimizer import InverterOption, PanelOption, ScenarioDefinition, ScenarioEvaluation
from sim_stochastic_pv.prices import EscalatingPriceModel
from sim_stochastic_pv.result_builder import ResultBuilder


def _make_stub_evaluation() -> ScenarioEvaluation:
    """Create a minimal ScenarioEvaluation for ResultBuilder tests."""
    df_profit = pd.DataFrame(
        {
            "month_index": [0],
            "year": [0],
            "month_in_year": [0],
            "prob_gain": [0.5],
            "mean_gain_eur": [10.0],
            "p05_gain_eur": [-5.0],
            "p95_gain_eur": [20.0],
            "mean_gain_real_eur": [9.0],
            "p05_gain_real_eur": [-6.0],
            "p95_gain_real_eur": [19.0],
        }
    )
    df_energy = pd.DataFrame(
        {
            "month_index": [0],
            "year": [0],
            "month_in_year": [0],
            "pv_prod_mean_kwh": [1.0],
            "pv_prod_p05_kwh": [0.8],
            "pv_prod_p95_kwh": [1.2],
            "solar_used_mean_kwh": [0.5],
            "solar_used_p05_kwh": [0.4],
            "solar_used_p95_kwh": [0.6],
            "grid_import_mean_kwh": [0.5],
            "grid_import_p05_kwh": [0.4],
            "grid_import_p95_kwh": [0.6],
            "savings_mean_kwh": [0.5],
            "savings_p05_kwh": [0.4],
            "savings_p95_kwh": [0.6],
        }
    )
    df_soc = pd.DataFrame(
        {"month_in_year": [0], "hour": [0], "soc_mean": [0.5], "soc_p05": [0.4], "soc_p95": [0.6]}
    )
    df_soh = pd.DataFrame(
        {
            "month_index": [0],
            "year": [0],
            "month_in_year": [0],
            "soh_mean": [0.99],
            "soh_p05": [0.97],
            "soh_p95": [1.0],
        }
    )
    results = MonteCarloResults(
        df_profit=df_profit,
        df_energy=df_energy,
        df_soc=df_soc,
        df_soh=df_soh,
        monthly_savings_eur_paths=np.zeros((1, 1)),
        monthly_savings_real_eur_paths=np.zeros((1, 1)),
        monthly_load_kwh_paths=np.zeros((1, 1)),
        irr_annual_paths=np.array([0.05]),
    )
    econ_cfg = EconomicConfig(investment_eur=500.0, n_mc=1)
    energy_cfg = EnergySystemConfig(n_years=1)
    definition = ScenarioDefinition(
        inverter=InverterOption(name="Inverter", p_ac_max_kw=1.0, price_eur=1000.0),
        panel=PanelOption(name="Panel", power_w=400.0, price_eur=200.0),
        panel_count=2,
        battery_option=None,
        battery_count=0,
    )
    return ScenarioEvaluation(
        definition=definition,
        economic_config=econ_cfg,
        energy_config=energy_cfg,
        results=results,
        break_even_month=1,
        final_gain_eur=10.0,
    )


def test_result_builder_build_analysis(tmp_path, monkeypatch):
    """Ensure build_analysis delegates to generate_report and returns its path."""
    evaluation = _make_stub_evaluation()
    builder = ResultBuilder(output_root=tmp_path)
    price_model = EscalatingPriceModel()

    fake_dir = tmp_path / "report_dir"
    fake_dir.mkdir()

    def fake_generate_report(**kwargs):
        return fake_dir

    monkeypatch.setattr("sim_stochastic_pv.result_builder.generate_report", fake_generate_report)
    output = builder.build_analysis(
        "scenario-test",
        results=evaluation.results,
        energy_config=evaluation.energy_config,
        economic_config=evaluation.economic_config,
        price_model=price_model,
    )
    assert output == fake_dir


def test_result_builder_build_optimization_bundle(tmp_path, monkeypatch):
    """Ensure optimization bundle persists plots, summaries, and reports."""
    evaluation = _make_stub_evaluation()
    price_model = EscalatingPriceModel()
    builder = ResultBuilder(output_root=tmp_path)

    monkeypatch.setattr("sim_stochastic_pv.result_builder._plot_profit_curves", lambda *args, **kwargs: None)
    monkeypatch.setattr("sim_stochastic_pv.result_builder._plot_final_profit_distribution", lambda *a, **k: None)
    monkeypatch.setattr("sim_stochastic_pv.result_builder._plot_break_even_vs_gain", lambda *a, **k: None)

    def fake_generate_report(**kwargs):
        output_root = Path(kwargs["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        return output_root

    monkeypatch.setattr("sim_stochastic_pv.result_builder.generate_report", fake_generate_report)

    run_dir = builder.build_optimization_bundle("batch_test", [evaluation], price_model)

    assert run_dir.exists()
    summary_path = run_dir / "comparison" / "summary.csv"
    assert summary_path.exists()
