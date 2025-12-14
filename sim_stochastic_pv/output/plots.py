"""
Plotting functions for optimization results.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..simulation.optimizer import ScenarioEvaluation

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




# Additional plotting functions from reporting module

def _plot_monthly_savings_distribution(
    savings_eur_paths: np.ndarray,
    save_path: Path,
) -> None:
    data = savings_eur_paths.flatten()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    weights = np.ones_like(data, dtype=float) / max(len(data), 1) * 100.0
    ax.hist(data, bins=40, color="#1f77b4", alpha=0.8, weights=weights)
    ax.set_xlabel("Risparmio mensile [€]")
    ax.set_ylabel("Percentuale [%]")
    ax.set_title("Distribuzione percentuale dei risparmi mensili")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def _plot_energy_consumption_distributions(
    load_kwh_paths: np.ndarray,
    month_in_year: np.ndarray,
    save_path_total: Path,
    save_path_grouped: Path,
) -> None:
    flattened = load_kwh_paths.flatten()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(flattened, bins=40, color="#ff7f0e", alpha=0.8, density=True)
    ax.set_xlabel("Consumo mensile [kWh]")
    ax.set_ylabel("Probabilità")
    ax.set_title("Distribuzione del consumo energetico (tutti i mesi)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path_total, dpi=300)
    plt.close(fig)

    groups: Dict[str, Iterable[int]] = {
        "Freddi (Ott-Feb)": COLD_MONTHS,
        "Caldi (Giu-Set)": HOT_MONTHS,
        "Temerati (Mar-Mag)": TEMPERATE_MONTHS,
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, months in groups.items():
        mask = np.isin(month_in_year, list(months))
        if not mask.any():
            continue
        data = load_kwh_paths[:, mask].flatten()
        if data.size == 0:
            continue
        ax.hist(
            data,
            bins=40,
            density=True,
            histtype="step",
            linewidth=2,
            label=label,
        )

    ax.set_xlabel("Consumo mensile [kWh]")
    ax.set_ylabel("Probabilità")
    ax.set_title("Distribuzione consumo per stagionalità")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path_grouped, dpi=300)
    plt.close(fig)


