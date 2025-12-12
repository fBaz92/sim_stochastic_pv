from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sim_stochastic_pv import ScenarioEvaluation, ScenarioOptimizer, generate_report

from scenario_setup import (
    build_default_economic_config,
    build_default_energy_config,
    build_default_load_profile,
    build_default_optimization_request,
    build_default_price_model,
    build_default_solar_model,
)


def _slugify(value: str) -> str:
    """
    Convert an arbitrary string to a filesystem-safe slug.

    Args:
        value: Raw string that may contain spaces or special characters.

    Returns:
        A lowercase string composed only of alphanumeric characters, dash, or underscore.
    """
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value.strip()).strip("_")


def _create_run_directory(scenario_name: str) -> Path:
    """
    Create a timestamped directory that collects all artifacts for a run.

    Args:
        scenario_name: Human-readable scenario batch name.

    Returns:
        Path to the created directory inside ``results/``.
    """
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    slug = _slugify(scenario_name) or "scenario"
    run_dir = Path("results") / f"{timestamp}_{slug}_batch"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _battery_label(definition) -> str:
    """
    Return a compact label describing the battery setup for a scenario definition.

    Args:
        definition: ScenarioDefinition with inverter, panel, and battery choices.

    Returns:
        Short string used in tables/plots to identify the battery configuration.
    """
    if definition.integrated_battery_specs and definition.integrated_battery_count > 0:
        return f"{definition.inverter.name}-integrated"
    if definition.battery_option and definition.battery_count > 0:
        return definition.battery_option.name
    return "no battery"


def _select_comparison_scenarios(
    evaluations: List[ScenarioEvaluation],
    top_n: int = 10,
) -> List[ScenarioEvaluation]:
    """
    Build a representative subset of scenarios for the comparison dashboard.

    Args:
        evaluations: All evaluated scenarios.
        top_n: Number of top performers to include before covering each inverter/battery.

    Returns:
        Sorted list of unique scenarios emphasizing high gain and component diversity.
    """
    if not evaluations:
        return []

    sorted_by_gain = sorted(evaluations, key=lambda ev: ev.final_gain_eur, reverse=True)
    selected: List[ScenarioEvaluation] = []
    seen_ids: set[int] = set()

    def _add(ev: ScenarioEvaluation) -> None:
        """Append scenario if it is not already part of the selection."""
        if id(ev) not in seen_ids:
            seen_ids.add(id(ev))
            selected.append(ev)

    for ev in sorted_by_gain[:top_n]:
        _add(ev)

    best_by_inverter: dict[str, ScenarioEvaluation] = {}
    for ev in sorted_by_gain:
        name = ev.definition.inverter.name
        best = best_by_inverter.get(name)
        if best is None or ev.final_gain_eur > best.final_gain_eur:
            best_by_inverter[name] = ev
    for ev in best_by_inverter.values():
        _add(ev)

    best_by_battery: dict[str, ScenarioEvaluation] = {}
    for ev in sorted_by_gain:
        label = _battery_label(ev.definition)
        best = best_by_battery.get(label)
        if best is None or ev.final_gain_eur > best.final_gain_eur:
            best_by_battery[label] = ev
    for ev in best_by_battery.values():
        _add(ev)

    return sorted(selected, key=lambda ev: ev.final_gain_eur, reverse=True)


def _short_label(value: str, max_len: int = 50) -> str:
    """
    Produce a compact label that fits legends or tables.

    Args:
        value: Original string.
        max_len: Desired maximum length.

    Returns:
        Possibly truncated string ending with ``...`` when over the limit.
    """
    return value if len(value) <= max_len else value[: max_len - 3] + "..."


def _final_profit_paths(evaluation: ScenarioEvaluation) -> np.ndarray:
    """
    Compute the distribution of final profit for one scenario.

    Args:
        evaluation: ScenarioEvaluation containing monthly savings paths.

    Returns:
        1D array with the final cumulative profit per Monte Carlo path.
    """
    monthly = evaluation.results.monthly_savings_eur_paths
    if monthly.size == 0:
        return np.array([])
    profit_paths = monthly.cumsum(axis=1) - evaluation.definition.investment_eur
    return profit_paths[:, -1]


def _compute_irr_stats(evaluation: ScenarioEvaluation) -> tuple[float, float, float] | None:
    """
    Summarize the IRR distribution for a scenario.

    Args:
        evaluation: ScenarioEvaluation exposing ``irr_annual_paths``.

    Returns:
        Tuple ``(mean, percentile_5, percentile_95)`` or ``None`` if IRR is undefined.
    """
    irr_paths = evaluation.results.irr_annual_paths
    if irr_paths.size == 0:
        return None
    valid = irr_paths[~np.isnan(irr_paths)]
    if valid.size == 0:
        return None
    return (
        float(np.mean(valid)),
        float(np.percentile(valid, 5)),
        float(np.percentile(valid, 95)),
    )


def _plot_profit_curves(evaluations: List[ScenarioEvaluation], save_path: Path) -> None:
    """
    Plot mean cumulative profit trajectories for all selected scenarios.

    Args:
        evaluations: Scenarios to plot.
        save_path: Output image path.

    Returns:
        None. Figure is saved to disk and closed.
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
    Visualize the final profit distribution for each scenario using violin plots.

    Args:
        evaluations: Scenario evaluations to compare.
        save_path: Output image path.

    Returns:
        None.
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
    Plot break-even month against final mean gain for several scenarios.

    Args:
        evaluations: Scenario evaluations to include.
        save_path: Output image path.

    Returns:
        None.
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
    Save a CSV file describing the main economic metrics for each scenario.

    Args:
        evaluations: Scenario evaluations to serialize.
        save_path: Destination CSV file.

    Returns:
        None. The CSV file is written to ``save_path``.
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
                "battery": _battery_label(ev.definition),
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
    Write a detailed text summary for a specific scenario.

    Args:
        path: Output text file path.
        evaluation: Scenario evaluation to summarize.

    Returns:
        None. The summary is saved to disk.
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
    irr_stats = _compute_irr_stats(evaluation)
    if irr_stats is None:
        lines.append("TIR: non disponibile (flussi non sufficienti)")
    else:
        irr_mean, irr_p05, irr_p95 = irr_stats
        lines.append(
            f"TIR medio: {irr_mean:.2%} (5° percentile {irr_p05:.2%} / 95° percentile {irr_p95:.2%})"
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


def _persist_best_scenario(
    label: str,
    evaluation: ScenarioEvaluation,
    run_dir: Path,
    price_model,
) -> Path:
    """
    Persist all assets for a winning scenario (CSV, plots, reports).

    Args:
        label: Subdirectory name, e.g., ``best_gain``.
        evaluation: Scenario evaluation to export.
        run_dir: Root directory for the optimization run.
        price_model: Price model used in the simulations (passed to reporting).

    Returns:
        Path pointing to the created scenario directory.
    """
    scenario_dir = run_dir / label
    scenario_dir.mkdir(parents=True, exist_ok=True)

    df_profit = evaluation.results.df_profit
    df_profit.to_csv(scenario_dir / "profit_loss.csv", index=False)
    final_profit_paths = _final_profit_paths(evaluation)
    if final_profit_paths.size > 0:
        pd.DataFrame({"final_profit_eur": final_profit_paths}).to_csv(
            scenario_dir / "final_profit_distribution.csv",
            index=False,
        )

    _write_best_summary_txt(scenario_dir / "summary.txt", evaluation)
    generate_report(
        scenario_name=evaluation.definition.describe(),
        results=evaluation.results,
        energy_config=evaluation.energy_config,
        economic_config=evaluation.economic_config,
        price_model=price_model,
        output_root=scenario_dir / "detailed_report",
    )
    return scenario_dir


def _load_profile_factory():
    """
    Factory wrapper to create a fresh load profile for each scenario evaluation.

    Returns:
        LoadProfile built by ``build_default_load_profile``.
    """
    return build_default_load_profile()


def main() -> None:
    """
    Run the optimization workflow, select best scenarios, and generate reports.

    Returns:
        None. All artifacts are written under ``results/`` and key metrics are printed.
    """
    request = build_default_optimization_request()
    base_energy_cfg = build_default_energy_config()
    econ_cfg = build_default_economic_config(n_mc=200)
    price_model = build_default_price_model()
    solar_model = build_default_solar_model()

    optimizer = ScenarioOptimizer(
        request=request,
        base_energy_config=base_energy_cfg,
        economic_config_template=econ_cfg,
        price_model=price_model,
        load_profile_factory=_load_profile_factory,
        solar_model=solar_model,
    )

    evaluations = optimizer.run(seed=321)
    if not evaluations:
        print("Nessuno scenario generato.")
        return

    run_dir = _create_run_directory(request.scenario_name)

    best_break_even = min(
        evaluations,
        key=lambda ev: (
            ev.break_even_month if ev.break_even_month is not None else float("inf"),
            -ev.final_gain_eur,
        ),
    )
    best_gain = max(evaluations, key=lambda ev: ev.final_gain_eur)

    comparison_dir = run_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_set = _select_comparison_scenarios(evaluations)
    _plot_profit_curves(comparison_set, comparison_dir / "profit_curves.png")
    _plot_final_profit_distribution(comparison_set, comparison_dir / "final_profit_distribution.png")
    _plot_break_even_vs_gain(comparison_set, comparison_dir / "break_even_vs_gain.png")
    _save_comparison_summary(comparison_set, comparison_dir / "summary.csv")

    gain_dir = _persist_best_scenario("best_gain", best_gain, run_dir, price_model)
    break_even_dir = _persist_best_scenario("best_break_even", best_break_even, run_dir, price_model)

    print("\n--- Risultati ottimizzazione ---")
    print("Scenario migliore per break-even:")
    print(f"  {best_break_even.definition.describe()}")
    if best_break_even.break_even_month is None:
        print("  Break-even non raggiunto")
    else:
        print(f"  Break-even al mese {best_break_even.break_even_month}")
    print(f"  Investimento: {best_break_even.definition.investment_eur:.2f} €")
    print(f"  Guadagno medio 20 anni: {best_break_even.final_gain_eur:.2f} €")
    irr_stats_be = _compute_irr_stats(best_break_even)
    if irr_stats_be is None:
        print("  TIR: non disponibile")
    else:
        irr_mean, irr_p05, irr_p95 = irr_stats_be
        print(f"  TIR medio: {irr_mean:.2%} (5° {irr_p05:.2%} / 95° {irr_p95:.2%})")
    print(f"  Report dettagliato: {break_even_dir}")

    print("\nScenario migliore per guadagno finale:")
    print(f"  {best_gain.definition.describe()}")
    if best_gain.break_even_month is None:
        print("  Break-even non raggiunto")
    else:
        print(f"  Break-even al mese {best_gain.break_even_month}")
    print(f"  Investimento: {best_gain.definition.investment_eur:.2f} €")
    print(f"  Guadagno medio 20 anni: {best_gain.final_gain_eur:.2f} €")
    irr_stats_gain = _compute_irr_stats(best_gain)
    if irr_stats_gain is None:
        print("  TIR: non disponibile")
    else:
        irr_mean, irr_p05, irr_p95 = irr_stats_gain
        print(f"  TIR medio: {irr_mean:.2%} (5° {irr_p05:.2%} / 95° {irr_p95:.2%})")
    print(f"  Report dettagliato: {gain_dir}")

    print("\nConfronto globale salvato in:", comparison_dir)
    print("Cartella run:", run_dir)


if __name__ == "__main__":
    main()
