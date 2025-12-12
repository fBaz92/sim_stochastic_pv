from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .simulation.optimizer import ScenarioEvaluation
from .reporting import generate_report


def _slugify(value: str) -> str:
    """
    Convert a free-form string into a filesystem-safe slug.

    Args:
        value: Input string.

    Returns:
        Slug containing only alphanumeric characters, dash, or underscore.
    """
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value.strip()).strip("_")


def _create_run_directory(scenario_name: str, output_root: Path) -> Path:
    """
    Create the timestamped comparison directory for optimization results.
    """
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    slug = _slugify(scenario_name) or "scenario"
    run_dir = output_root / f"{timestamp}_{slug}_batch"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _short_label(value: str, max_len: int = 50) -> str:
    """
    Truncate long labels for plots.
    """
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


def _final_profit_paths(evaluation: ScenarioEvaluation) -> np.ndarray:
    """
    Return the final profit for each Monte Carlo path of an evaluation.
    """
    monthly = evaluation.results.monthly_savings_eur_paths
    if monthly.size == 0:
        return np.array([])
    profit_paths = monthly.cumsum(axis=1) - evaluation.definition.investment_eur
    return profit_paths[:, -1]


def _plot_profit_curves(evaluations: List[ScenarioEvaluation], save_path: Path) -> None:
    """
    Plot mean cumulative profit curves for the provided evaluations.
    """
    if not evaluations:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    for ev in evaluations:
        df = ev.results.df_profit
        ax.plot(
            df["month_index"],
            df["mean_gain_eur"],
            label=_short_label(ev.definition.describe(), 40),
        )

    ax.set_xlabel("Mese simulato")
    ax.set_ylabel("Profitto cumulato medio [€]")
    ax.set_title("Confronto profitti medi cumulati")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_final_profit_distribution(evaluations: List[ScenarioEvaluation], save_path: Path) -> None:
    """
    Plot violin charts summarizing final profit distributions.
    """
    datasets = []
    labels = []
    for ev in evaluations:
        profits = _final_profit_paths(ev)
        if profits.size == 0:
            continue
        datasets.append(profits)
        labels.append(_short_label(ev.definition.describe(), 35))
    if not datasets:
        return
    fig_width = max(8, len(labels) * 0.9)
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    parts = ax.violinplot(datasets, showmeans=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_alpha(0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Distribuzione profitto finale [€]")
    ax.set_title("Distribuzione Monte Carlo profitto finale")
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_break_even_vs_gain(evaluations: List[ScenarioEvaluation], save_path: Path) -> None:
    """
    Plot scatter relating break-even month and final gain.
    """
    if not evaluations:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    max_months = max(int(ev.results.df_profit["month_index"].max()) for ev in evaluations)
    horizon = max_months + 1
    for ev in evaluations:
        x_val = ev.break_even_month if ev.break_even_month is not None else horizon + 6
        marker = "o" if ev.break_even_month is not None else "x"
        ax.scatter(
            x_val,
            ev.final_gain_eur,
            label=_short_label(ev.definition.describe(), 35),
            marker=marker,
        )
    ax.axvline(horizon, color="gray", linestyle="--", linewidth=1, label="Fine orizzonte")
    ax.set_xlabel("Mese di break-even (o >orizzonte)")
    ax.set_ylabel("Profitto finale medio [€]")
    ax.set_title("Break-even vs profitto finale")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _save_comparison_summary(evaluations: List[ScenarioEvaluation], save_path: Path) -> None:
    """
    Save CSV summary of the evaluations used for comparison.
    """
    if not evaluations:
        return
    rows = []
    for ev in evaluations:
        df_profit = ev.results.df_profit
        final_row = df_profit.iloc[-1]
        rows.append(
            {
                "scenario": ev.definition.describe(),
                "inverter": ev.definition.inverter.name,
                "pv_kwp": round(ev.definition.pv_kwp, 3),
                "investment_eur": round(ev.definition.investment_eur, 2),
                "final_gain_eur": round(ev.final_gain_eur, 2),
                "break_even_month": ev.break_even_month if ev.break_even_month is not None else "-",
                "prob_gain_final": round(float(final_row["prob_gain"]), 3),
                "p05_final_eur": round(float(final_row["p05_gain_eur"]), 2),
                "p95_final_eur": round(float(final_row["p95_gain_eur"]), 2),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)


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
