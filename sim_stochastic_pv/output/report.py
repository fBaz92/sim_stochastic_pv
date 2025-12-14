"""
Main report generation function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..simulation.energy_simulator import EnergySystemConfig
from ..simulation.monte_carlo import EconomicConfig, MonteCarloResults, MonteCarloSimulator
from ..simulation.prices import PriceModel
from .utils import create_results_directory
from .plots import _plot_monthly_savings_distribution, _plot_energy_consumption_distributions
from .formatters import _write_text_report

def generate_report(
    scenario_name: str,
    results: MonteCarloResults,
    energy_config: EnergySystemConfig,
    economic_config: EconomicConfig,
    price_model: PriceModel,
    output_root: Path | str = "results",
) -> Path:
    """
    Generate full report: plots and textual summary saved to disk.
    """
    output_dir = _create_results_directory(scenario_name, Path(output_root))

    MonteCarloSimulator.plot_profit_bands(
        results.df_profit,
        save_path=output_dir / "profit.png",
        show=False,
    )
    MonteCarloSimulator.plot_monthly_energy_bands(
        results.df_energy,
        var_prefix="pv_prod",
        save_path=output_dir / "pv_production.png",
        show=False,
    )
    MonteCarloSimulator.plot_monthly_energy_bands(
        results.df_energy,
        var_prefix="solar_used",
        aggregate_by_year=True,
        save_path=output_dir / "solar_used_per_year.png",
        show=False,
    )
    MonteCarloSimulator.plot_soh_evolution(
        results.df_soh,
        save_path=output_dir / "soh.png",
        show=False,
    )
    MonteCarloSimulator.plot_monthly_soc_bands(
        results.df_soc,
        save_dir=output_dir / "soc_profiles",
        show=False,
    )

    _plot_monthly_savings_distribution(
        results.monthly_savings_eur_paths,
        save_path=output_dir / "savings_distribution.png",
    )
    month_in_year = results.df_energy.sort_values("month_index")["month_in_year"].values
    _plot_energy_consumption_distributions(
        results.monthly_load_kwh_paths,
        month_in_year=month_in_year,
        save_path_total=output_dir / "energy_consumption_distribution.png",
        save_path_grouped=output_dir / "energy_consumption_by_season.png",
    )

    break_even_text = _format_break_even(results.df_profit)
    yearly_summary = _build_yearly_summary(
        df_profit=results.df_profit,
        df_energy=results.df_energy,
        df_soh=results.df_soh,
        price_model=price_model,
        economic_config=economic_config,
    )
    _write_text_report(
        output_path=output_dir / "report.txt",
        scenario_name=scenario_name,
        energy_config=energy_config,
        economic_config=economic_config,
        price_model=price_model,
        break_even_text=break_even_text,
        yearly_summary=yearly_summary,
        df_profit=results.df_profit,
        df_energy=results.df_energy,
    )

    return output_dir
