"""
Result builder for optimization outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..simulation.optimizer import ScenarioEvaluation
from .utils import create_run_directory
from .plots import _short_label, _final_profit_paths, _plot_profit_curves, _plot_final_profit_distribution, _plot_break_even_vs_gain, _save_comparison_summary
from .report import generate_report

def _write_best_summary_txt(path: Path, evaluation: ScenarioEvaluation) -> None:
    """
    Write the human-readable summary for a single scenario evaluation.
    """
    defn = evaluation.definition
    df_profit = evaluation.results.df_profit
    final_row = df_profit.iloc[-1]
    lines = []
    lines.append(f"Scenario: {defn.describe()}")
    lines.append("")
    lines.append("== Componenti ==")
    lines.append(f"Inverter: {defn.inverter.name} (P_AC max {defn.inverter.p_ac_max_kw:.2f} kW)")
    lines.append(f"Campo FV: {defn.panel_count}x {defn.panel.name} ({defn.pv_kwp:.2f} kWp)")
    if defn.integrated_battery_specs and defn.integrated_battery_count > 0:
        batt_desc = (
            f"Batteria integrata: {defn.integrated_battery_count}x {defn.integrated_battery_specs.capacity_kwh:.2f} kWh"
        )
    elif defn.battery_option and defn.battery_count > 0:
        batt_desc = (
            f"Batterie: {defn.battery_count}x {defn.battery_option.name} "
            f"({defn.battery_option.specs.capacity_kwh:.2f} kWh)"
        )
    else:
        batt_desc = "Batterie: nessuna"
    lines.append(batt_desc)
    lines.append(f"Investimento totale: {defn.investment_eur:.2f} €")
    lines.append("")
    lines.append("== Metriche economiche ==")
    lines.append(f"Guadagno finale medio: {evaluation.final_gain_eur:.2f} €")
    lines.append(
        f"Intervallo 5°-95° percentile: {final_row['p05_gain_eur']:.2f} € / {final_row['p95_gain_eur']:.2f} €"
    )
    lines.append(
        f"Guadagno reale medio (inflazione): {final_row['mean_gain_real_eur']:.2f} € "
        f"(5°-95°: {final_row['p05_gain_real_eur']:.2f} € / {final_row['p95_gain_real_eur']:.2f} €)"
    )
    lines.append(f"Probabilità profitto positivo a fine orizzonte: {final_row['prob_gain']:.2%}")
    if evaluation.break_even_month is None:
        lines.append("Break-even non raggiunto nell'orizzonte simulato")
    else:
        year = evaluation.break_even_month // 12
        lines.append(
            f"Break-even al mese {evaluation.break_even_month} (anno {year + 1})"
        )
    lines.append(f"Percorsi Monte Carlo: {evaluation.economic_config.n_mc}")
    path.write_text("\n".join(lines), encoding="utf-8")


class ResultBuilder:
    """
    Handle persistence of analysis and optimization deliverables.
    """

    def __init__(self, output_root: str | Path = "results") -> None:
        """
        Args:
            output_root: Base directory for generated assets.
        """
        self.output_root = Path(output_root)

    def build_analysis(
        self,
        scenario_name: str,
        *,
        results,
        energy_config,
        economic_config,
        price_model,
    ) -> Path:
        """
        Save the analysis report for a single scenario.

        Args:
            scenario_name: Name used for the output directory.
            results: MonteCarloResults object.
            energy_config: Energy configuration dataclass.
            economic_config: Economic configuration dataclass.
            price_model: Price model instance.
        """
        return generate_report(
            scenario_name=scenario_name,
            results=results,
            energy_config=energy_config,
            economic_config=economic_config,
            price_model=price_model,
            output_root=self.output_root,
        )

    def build_optimization_bundle(
        self,
        request_name: str,
        evaluations: List[ScenarioEvaluation],
        price_model,
    ) -> Path:
        """
        Persist plots and reports for optimization results.

        Args:
            request_name: Base name for the run directory.
            evaluations: Scenario evaluations to summarize.
            price_model: Price model used for reports.
        """
        if not evaluations:
            raise ValueError("No evaluations available to build results.")

        run_dir = _create_run_directory(request_name, self.output_root)
        comparison_dir = run_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        _plot_profit_curves(evaluations, comparison_dir / "profit_curves.png")
        _plot_final_profit_distribution(evaluations, comparison_dir / "final_profit_distribution.png")
        _plot_break_even_vs_gain(evaluations, comparison_dir / "break_even_vs_gain.png")
        _save_comparison_summary(evaluations, comparison_dir / "summary.csv")

        best_break_even = min(
            evaluations,
            key=lambda ev: (
                ev.break_even_month if ev.break_even_month is not None else float("inf"),
                -ev.final_gain_eur,
            ),
        )
        best_gain = max(evaluations, key=lambda ev: ev.final_gain_eur)

        self._persist_best_scenario(run_dir / "best_break_even", best_break_even, price_model)
        self._persist_best_scenario(run_dir / "best_gain", best_gain, price_model)

        return run_dir

    def _persist_best_scenario(
        self,
        target_dir: Path,
        evaluation: ScenarioEvaluation,
        price_model,
    ) -> None:
        """
        Persist CSV, summary, and detailed report for one scenario.

        Args:
            target_dir: Destination directory.
            evaluation: ScenarioEvaluation to export.
            price_model: Price model forwarded to `generate_report`.
        """
        target_dir.mkdir(parents=True, exist_ok=True)

        df_profit = evaluation.results.df_profit
        df_profit.to_csv(target_dir / "profit_loss.csv", index=False)
        final_profit_paths = _final_profit_paths(evaluation)
        if final_profit_paths.size > 0:
            pd.DataFrame({"final_profit_eur": final_profit_paths}).to_csv(
                target_dir / "final_profit_distribution.csv",
                index=False,
            )

        _write_best_summary_txt(target_dir / "summary.txt", evaluation)
        generate_report(
            scenario_name=evaluation.definition.describe(),
            results=evaluation.results,
            energy_config=evaluation.energy_config,
            economic_config=evaluation.economic_config,
            price_model=price_model,
            output_root=target_dir / "detailed_report",
        )
