"""
Text formatting and report generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

MONTH_NAMES = [
    "Gennaio", "Febbraio", "Marzo", "Aprile", "Maggio", "Giugno",
    "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre",
]

HOT_MONTHS = {5, 6, 7, 8}
COLD_MONTHS = {9, 10, 11, 0, 1}
TEMPERATE_MONTHS = set(range(12)) - HOT_MONTHS - COLD_MONTHS

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


