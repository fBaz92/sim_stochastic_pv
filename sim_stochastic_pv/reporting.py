from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .energy_simulator import EnergySystemConfig
from .monte_carlo import EconomicConfig, MonteCarloResults, MonteCarloSimulator
from .prices import PriceModel

MONTH_NAMES = [
    "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
    "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre",
]

HOT_MONTHS = {5, 6, 7, 8}
COLD_MONTHS = {9, 10, 11, 0, 1}
TEMPERATE_MONTHS = set(range(12)) - HOT_MONTHS - COLD_MONTHS


def _slugify(value: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in value.strip()).strip("_")


def _create_results_directory(scenario_name: str, base_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    slug = _slugify(scenario_name) or "scenario"
    output_dir = base_dir / f"{timestamp}_{slug}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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


def _format_break_even(df_profit: pd.DataFrame) -> str:
    mask = df_profit["mean_gain_eur"] >= 0.0
    if not mask.any():
        return "Il break-even non è raggiunto entro l'orizzonte simulato."

    row = df_profit[mask].iloc[0]
    year = int(row["year"])
    month_name = MONTH_NAMES[int(row["month_in_year"])]
    return f"Break-even atteso nel mese {month_name} dell'anno {year + 1} (mese {int(row['month_index'])})."


def _build_yearly_summary(
    df_profit: pd.DataFrame,
    df_energy: pd.DataFrame,
    df_soh: pd.DataFrame,
    price_model: PriceModel,
    economic_config: EconomicConfig,
) -> pd.DataFrame:
    df_energy = df_energy.copy()
    prices = [
        price_model.get_price(int(row.year), int(row.month_in_year))
        for row in df_energy.itertuples()
    ]
    df_energy["savings_mean_eur"] = df_energy["savings_mean_kwh"] * np.array(prices)
    inflation_factors = np.power(
        1.0 + economic_config.inflation_rate,
        df_energy["year"].values,
    )
    df_energy["savings_mean_real_eur"] = df_energy["savings_mean_eur"] / inflation_factors

    annual = df_energy.groupby("year").agg(
        savings_mean_eur=("savings_mean_eur", "sum"),
        savings_mean_real_eur=("savings_mean_real_eur", "sum"),
        pv_prod_mean_kwh=("pv_prod_mean_kwh", "sum"),
    )
    annual["gain_loss_eur"] = df_profit.groupby("year")["mean_gain_eur"].last()
    annual["gain_loss_real_eur"] = df_profit.groupby("year")["mean_gain_real_eur"].last()
    annual["soh_mean"] = df_soh.groupby("year")["soh_mean"].mean()
    return annual.reset_index()


def _write_text_report(
    output_path: Path,
    scenario_name: str,
    energy_config: EnergySystemConfig,
    economic_config: EconomicConfig,
    price_model: PriceModel,
    break_even_text: str,
    yearly_summary: pd.DataFrame,
    df_profit: pd.DataFrame,
    df_energy: pd.DataFrame,
) -> None:
    lines = []
    lines.append(f"Scenario: {scenario_name}")
    lines.append("== Configurazione sistema energetico ==")
    lines.append(f"- Orizzonte: {energy_config.n_years} anni")
    lines.append(f"- Impianto FV: {energy_config.pv_kwp:.2f} kWp")
    lines.append(f"- Batterie: {energy_config.n_batteries}x {energy_config.battery_specs.capacity_kwh:.2f} kWh "
                 f"(cicli vita {energy_config.battery_specs.cycles_life})")
    lines.append(f"- Inverter AC max: {energy_config.inverter_p_ac_max_kw:.2f} kW")
    lines.append(f"- Efficienze carica/scarica: {energy_config.eta_charge:.2f}/{energy_config.eta_discharge:.2f}")
    lines.append(f"- Investimento iniziale: {economic_config.investment_eur:.2f} €")
    lines.append(f"- Percorsi Monte Carlo: {economic_config.n_mc}")
    lines.append(f"- Inflazione attesa: {economic_config.inflation_rate:.2%}")

    price_info = []
    if hasattr(price_model, "base_price"):
        price_info.append(f"prezzo base {getattr(price_model, 'base_price'):.3f} €/kWh")
    if hasattr(price_model, "annual_escalation"):
        price_info.append(f"crescita annua {getattr(price_model, 'annual_escalation'):.2%}")
    if price_info:
        lines.append("Modello prezzi: " + ", ".join(price_info))
    lines.append("")
    lines.append("== Break-even ==")
    lines.append(break_even_text)
    lines.append("")
    lines.append("== ROI nominale vs reale ==")
    final_nominal = df_profit["mean_gain_eur"].iloc[-1]
    final_real = df_profit["mean_gain_real_eur"].iloc[-1]
    p05_real = df_profit["p05_gain_real_eur"].iloc[-1]
    p95_real = df_profit["p95_gain_real_eur"].iloc[-1]
    lines.append(f"ROI medio cumulato nominale: {final_nominal:.2f} €")
    lines.append(
        f"ROI medio cumulato reale: {final_real:.2f} € "
        f"(5°-95° percentile: {p05_real:.2f} € / {p95_real:.2f} €)"
    )
    lines.append("")
    lines.append("== Sintesi annuale ==")
    lines.append(
        f"{'Anno':>4} | {'Risparmio [€]':>16} | {'Risparmio reale [€]':>20} | "
        f"{'Gain/Loss [€]':>15} | {'Gain/Loss reale [€]':>20} | {'SoH medio':>9} | {'PV prod [kWh]':>13}"
    )
    lines.append("-" * 118)
    for _, row in yearly_summary.iterrows():
        lines.append(
            f"{int(row['year']) + 1:4d} | "
            f"{row['savings_mean_eur']:16.2f} | "
            f"{row['savings_mean_real_eur']:20.2f} | "
            f"{row['gain_loss_eur']:15.2f} | "
            f"{row['gain_loss_real_eur']:20.2f} | "
            f"{row['soh_mean']:9.3f} | "
            f"{row['pv_prod_mean_kwh']:13.1f}"
        )

    total_pv = yearly_summary["pv_prod_mean_kwh"].sum()
    best_month_idx = df_energy["pv_prod_mean_kwh"].idxmax()
    best_month = df_energy.loc[best_month_idx]
    best_month_name = MONTH_NAMES[int(best_month["month_in_year"])]
    best_year = int(best_month["year"]) + 1
    lines.append("")
    lines.append("== Produzione FV ==")
    lines.append(f"Produzione media complessiva simulata: {total_pv:.1f} kWh.")
    lines.append(
        f"Il mese più produttivo risulta {best_month_name} anno {best_year} "
        f"con ~{best_month['pv_prod_mean_kwh']:.1f} kWh."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


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
