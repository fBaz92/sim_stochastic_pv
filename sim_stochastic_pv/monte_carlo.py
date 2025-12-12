from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .energy_simulator import EnergySystemSimulator
from .prices import PriceModel


def _npv(rate: float, cashflows: np.ndarray) -> float:
    periods = np.arange(cashflows.size, dtype=float)
    return np.sum(cashflows / np.power(1.0 + rate, periods))


def _compute_irr_monthly(
    cashflows: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    if cashflows.size < 2:
        return np.nan
    if not (np.any(cashflows > 0) and np.any(cashflows < 0)):
        return np.nan

    low = -0.9999
    high = 5.0
    npv_low = _npv(low, cashflows)
    npv_high = _npv(high, cashflows)

    expand = 0
    while npv_low * npv_high > 0 and expand < 12:
        high *= 2.0
        npv_high = _npv(high, cashflows)
        expand += 1
        if high > 1e6:
            return np.nan
    if npv_low * npv_high > 0:
        return np.nan

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        npv_mid = _npv(mid, cashflows)
        if abs(npv_mid) < tol:
            return mid
        if npv_low * npv_mid < 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid
    return mid


def _compute_irr_annual(cashflows: np.ndarray) -> float:
    irr_monthly = _compute_irr_monthly(cashflows)
    if np.isnan(irr_monthly):
        return np.nan
    return (1.0 + irr_monthly) ** 12 - 1.0


@dataclass
class EconomicConfig:
    """
    Economic configuration for Monte Carlo simulation.
    
    Attributes:
        investment_eur: Initial investment cost in EUR.
        n_mc: Number of Monte Carlo simulation paths to run.
    """
    investment_eur: float = 2500.0
    n_mc: int = 200
    inflation_rate: float = 0.025


@dataclass
class MonteCarloResults:
    """
    Container for Monte Carlo outputs.
    """
    df_profit: pd.DataFrame
    df_energy: pd.DataFrame
    df_soc: pd.DataFrame
    df_soh: pd.DataFrame
    monthly_savings_eur_paths: np.ndarray
    monthly_savings_real_eur_paths: np.ndarray
    monthly_load_kwh_paths: np.ndarray
    irr_annual_paths: np.ndarray


class MonteCarloSimulator:
    """
    Monte Carlo simulator for energy system economic analysis.
    
    Runs multiple simulation paths to compute statistical distributions
    of profit, energy flows, battery state of health, and state of charge.
    """
    
    def __init__(
        self,
        energy_simulator: EnergySystemSimulator,
        price_model: PriceModel,
        economic_config: EconomicConfig,
    ) -> None:
        """
        Initialize the Monte Carlo simulator.
        
        Args:
            energy_simulator: Energy system simulator for single-path simulation.
            price_model: Price model for calculating energy costs.
            economic_config: Economic configuration (investment, number of paths).
        """
        self.energy_simulator = energy_simulator
        self.price_model = price_model
        self.economic_config = economic_config

    def run(
        self,
        seed: int = 123,
        progress_callback: Callable[[int, int, float, float], None] | None = None,
        show_progress: bool = True,
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation and compute statistical results.
        
        Args:
            seed: Random seed for reproducibility.
        
        Returns:
            Tuple containing (in order):
            - df_profit: DataFrame with profit statistics (mean, percentiles).
            - df_energy: DataFrame with energy flow statistics (PV, solar used, grid import, savings).
            - df_soc: DataFrame with state of charge profiles (first year, by month/hour).
            - df_soh: DataFrame with state of health evolution over time.
        """
        cfg = self.economic_config
        n_years = self.energy_simulator.config.n_years
        n_months = n_years * 12
        n_mc = cfg.n_mc

        rng_global = np.random.default_rng(seed)

        profit_cum_paths = np.zeros((n_mc, n_months))
        pv_prod_paths = np.zeros((n_mc, n_months))
        solar_used_paths = np.zeros((n_mc, n_months))
        grid_import_paths = np.zeros((n_mc, n_months))
        savings_kwh_paths = np.zeros((n_mc, n_months))
        savings_eur_paths = np.zeros((n_mc, n_months))
        savings_real_eur_paths = np.zeros((n_mc, n_months))
        load_kwh_paths = np.zeros((n_mc, n_months))

        profit_cum_real_paths = np.zeros((n_mc, n_months))
        soh_paths = np.zeros((n_mc, n_months))
        soc_profiles_paths = np.zeros((n_mc, 12, 24))
        irr_annual_paths = np.full(n_mc, np.nan)

        months = np.arange(n_months)
        years = months // 12
        month_in_year = months % 12
        inflation_factors = np.power(1.0 + cfg.inflation_rate, years)

        bar_len = 30
        start_time = time.time()
        update_every = max(1, n_mc // 100)

        def print_progress(iteration: int) -> None:
            """Print progress bar for Monte Carlo simulation."""
            done = iteration + 1
            frac = done / n_mc
            elapsed = time.time() - start_time
            eta = (elapsed / frac - elapsed) if frac > 0 else 0.0

            filled = int(bar_len * frac)
            bar = "#" * filled + "-" * (bar_len - filled)

            msg = (
                f"\rMC {done:5d}/{n_mc:<5d} "
                f"[{bar}] {frac*100:6.2f}%  "
                f"elapsed: {elapsed:6.1f}s  ETA: {eta:6.1f}s"
            )
            sys.stdout.write(msg)
            sys.stdout.flush()

        for i in range(n_mc):
            rng = np.random.default_rng(rng_global.integers(0, 1_000_000_000))
            self.price_model.reset_for_run(rng=rng, n_years=n_years)

            (
                monthly_pv_prod_kwh,
                monthly_pv_direct_kwh,
                monthly_batt_to_load_kwh,
                monthly_grid_import_kwh,
                monthly_load_kwh,
                soh_end_of_month,
                soc_profile_first_year,
            ) = self.energy_simulator.run_one_path(rng)

            monthly_solar_used_kwh = monthly_pv_direct_kwh + monthly_batt_to_load_kwh
            monthly_savings_kwh = monthly_load_kwh - monthly_grid_import_kwh

            monthly_savings_eur = np.zeros(n_months)
            for m in range(n_months):
                year = m // 12
                month_in_year_idx = m % 12
                price = self.price_model.get_price(year, month_in_year_idx)
                monthly_savings_eur[m] = monthly_savings_kwh[m] * price

            profit_cum = -cfg.investment_eur + np.cumsum(monthly_savings_eur)
            monthly_savings_real = monthly_savings_eur / inflation_factors
            profit_cum_real = -cfg.investment_eur + np.cumsum(monthly_savings_real)
            cashflows = np.concatenate(([-cfg.investment_eur], monthly_savings_eur))
            irr_annual_paths[i] = _compute_irr_annual(cashflows)

            profit_cum_paths[i, :] = profit_cum
            profit_cum_real_paths[i, :] = profit_cum_real
            pv_prod_paths[i, :] = monthly_pv_prod_kwh
            solar_used_paths[i, :] = monthly_solar_used_kwh
            grid_import_paths[i, :] = monthly_grid_import_kwh
            savings_kwh_paths[i, :] = monthly_savings_kwh
            savings_eur_paths[i, :] = monthly_savings_eur
            savings_real_eur_paths[i, :] = monthly_savings_real
            load_kwh_paths[i, :] = monthly_load_kwh

            soh_paths[i, :] = soh_end_of_month
            soc_profiles_paths[i, :, :] = soc_profile_first_year

            iteration_done = i + 1
            elapsed = time.time() - start_time
            frac = iteration_done / n_mc
            eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
            if progress_callback is not None:
                progress_callback(iteration_done, n_mc, elapsed, eta)
            elif show_progress:
                if (i + 1) % update_every == 0 or (i + 1) == n_mc:
                    print_progress(i)

        if progress_callback is None and show_progress:
            sys.stdout.write("\n")

        prob_gain = (profit_cum_paths > 0.0).mean(axis=0)
        mean_gain = profit_cum_paths.mean(axis=0)
        p05_gain = np.percentile(profit_cum_paths, 5, axis=0)
        p95_gain = np.percentile(profit_cum_paths, 95, axis=0)
        mean_gain_real = profit_cum_real_paths.mean(axis=0)
        p05_gain_real = np.percentile(profit_cum_real_paths, 5, axis=0)
        p95_gain_real = np.percentile(profit_cum_real_paths, 95, axis=0)

        df_profit = pd.DataFrame(
            {
                "month_index": months,
                "year": years,
                "month_in_year": month_in_year,
                "prob_gain": prob_gain,
                "mean_gain_eur": mean_gain,
                "p05_gain_eur": p05_gain,
                "p95_gain_eur": p95_gain,
                "mean_gain_real_eur": mean_gain_real,
                "p05_gain_real_eur": p05_gain_real,
                "p95_gain_real_eur": p95_gain_real,
            }
        )

        def stats(arr_paths: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Compute statistical summary (mean, 5th percentile, 95th percentile).
            
            Args:
                arr_paths: Array of shape (n_paths, n_months) with simulation paths.
            
            Returns:
                Tuple of (mean, p05, p95) arrays, each of length n_months.
            """
            return (
                arr_paths.mean(axis=0),
                np.percentile(arr_paths, 5, axis=0),
                np.percentile(arr_paths, 95, axis=0),
            )

        mean_pv, p05_pv, p95_pv = stats(pv_prod_paths)
        mean_used, p05_used, p95_used = stats(solar_used_paths)
        mean_grid, p05_grid, p95_grid = stats(grid_import_paths)
        mean_sav_kwh, p05_sav_kwh, p95_sav_kwh = stats(savings_kwh_paths)

        df_energy = pd.DataFrame(
            {
                "month_index": months,
                "year": years,
                "month_in_year": month_in_year,
                "pv_prod_mean_kwh": mean_pv,
                "pv_prod_p05_kwh": p05_pv,
                "pv_prod_p95_kwh": p95_pv,
                "solar_used_mean_kwh": mean_used,
                "solar_used_p05_kwh": p05_used,
                "solar_used_p95_kwh": p95_used,
                "grid_import_mean_kwh": mean_grid,
                "grid_import_p05_kwh": p05_grid,
                "grid_import_p95_kwh": p95_grid,
                "savings_mean_kwh": mean_sav_kwh,
                "savings_p05_kwh": p05_sav_kwh,
                "savings_p95_kwh": p95_sav_kwh,
            }
        )

        soc_mean = soc_profiles_paths.mean(axis=0)
        soc_p05 = np.percentile(soc_profiles_paths, 5, axis=0)
        soc_p95 = np.percentile(soc_profiles_paths, 95, axis=0)

        rows = []
        for m in range(12):
            for h in range(24):
                rows.append(
                    {
                        "month_in_year": m,
                        "hour": h,
                        "soc_mean": soc_mean[m, h],
                        "soc_p05": soc_p05[m, h],
                        "soc_p95": soc_p95[m, h],
                    }
                )

        df_soc = pd.DataFrame(rows)

        soh_mean, soh_p05, soh_p95 = stats(soh_paths)
        df_soh = pd.DataFrame(
            {
                "month_index": months,
                "year": years,
                "month_in_year": month_in_year,
                "soh_mean": soh_mean,
                "soh_p05": soh_p05,
                "soh_p95": soh_p95,
            }
        )

        return MonteCarloResults(
            df_profit=df_profit,
            df_energy=df_energy,
            df_soc=df_soc,
            df_soh=df_soh,
            monthly_savings_eur_paths=savings_eur_paths,
            monthly_savings_real_eur_paths=savings_real_eur_paths,
            monthly_load_kwh_paths=load_kwh_paths,
            irr_annual_paths=irr_annual_paths,
        )

    # ---------- plotting utilities ----------

    @staticmethod
    def plot_profit_bands(
        df_profit: pd.DataFrame,
        save_path: Path | str | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot cumulative profit bands with mean and percentile ranges.
        
        Args:
            df_profit: DataFrame with profit statistics from run().
        """
        x = df_profit["month_index"].values
        mean_gain = df_profit["mean_gain_eur"].values
        p05 = df_profit["p05_gain_eur"].values
        p95 = df_profit["p95_gain_eur"].values

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, mean_gain, label="Profitto medio cumulato")
        ax.fill_between(x, p05, p95, alpha=0.3, label="Banda 5°-95° percentile")
        ax.axhline(0.0, linestyle="--", label="Break-even")
        ax.set_xlabel("Mese dall'investimento")
        ax.set_ylabel("Profitto cumulato [€]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def plot_monthly_energy_bands(
        df_energy: pd.DataFrame,
        var_prefix: str = "pv_prod",
        aggregate_by_year: bool = False,
        save_path: Path | str | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot energy flow bands with mean and percentile ranges.
        
        Args:
            df_energy: DataFrame with energy statistics from run().
            var_prefix: Variable prefix to plot (e.g., "pv_prod", "solar_used", "grid_import").
            aggregate_by_year: If True, aggregate monthly data by year.
        """
        mean_col = f"{var_prefix}_mean_kwh"
        p05_col = f"{var_prefix}_p05_kwh"
        p95_col = f"{var_prefix}_p95_kwh"

        if aggregate_by_year:
            grouped = df_energy.groupby("year")[[mean_col, p05_col, p95_col]].sum()
            x = grouped.index.values
            mean = grouped[mean_col].values
            p05 = grouped[p05_col].values
            p95 = grouped[p95_col].values
            xlabel = "Anno"
        else:
            x = df_energy["month_index"].values
            mean = df_energy[mean_col].values
            p05 = df_energy[p05_col].values
            p95 = df_energy[p95_col].values
            xlabel = "Mese dall'inizio"

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, mean, label=f"{var_prefix} medio")
        ax.fill_between(x, p05, p95, alpha=0.3, label="Banda 5°-95° percentile")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Energia [kWh]")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def plot_soh_evolution(
        df_soh: pd.DataFrame,
        save_path: Path | str | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot battery state of health evolution over time.
        
        Args:
            df_soh: DataFrame with SoH statistics from run().
        """
        x = df_soh["month_index"].values
        soh_mean = df_soh["soh_mean"].values
        soh_p05 = df_soh["soh_p05"].values
        soh_p95 = df_soh["soh_p95"].values

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(x, soh_mean, label="SoH medio")
        ax.fill_between(x, soh_p05, soh_p95, alpha=0.3, label="Banda 5°-95° percentile")
        ax.set_xlabel("Mese dall'inizio")
        ax.set_ylabel("SoH [p.u.]")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        unique_years = np.unique(df_soh["year"].values)
        year_ticks = unique_years * 12
        ax_top.set_xticks(year_ticks)
        ax_top.set_xticklabels([str(y) for y in unique_years])
        ax_top.set_xlabel("Anno")

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300)
        if show:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def plot_monthly_soc_bands(
        df_soc: pd.DataFrame,
        save_dir: Path | str | None = None,
        show: bool = True,
    ) -> None:
        """
        Plot state of charge profiles for each month (first year average).
        
        Creates separate plots for each month showing hourly SoC patterns
        with mean and percentile bands.
        
        Args:
            df_soc: DataFrame with SoC statistics from run().
        """
        months_labels = [
            "Gen", "Feb", "Mar", "Apr", "Mag", "Giu",
            "Lug", "Ago", "Set", "Ott", "Nov", "Dic"
        ]

        base_dir = Path(save_dir) if save_dir is not None else None
        if base_dir is not None:
            base_dir.mkdir(parents=True, exist_ok=True)

        for m in range(12):
            subset = df_soc[df_soc["month_in_year"] == m].sort_values("hour")
            h = subset["hour"].values
            soc_mean = subset["soc_mean"].values
            soc_p05 = subset["soc_p05"].values
            soc_p95 = subset["soc_p95"].values

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(h, soc_mean, label="SoC medio")
            ax.fill_between(h, soc_p05, soc_p95, alpha=0.3, label="5°–95° percentile")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Ora del giorno")
            ax.set_ylabel("SoC [p.u.]")
            ax.set_title(f"Profilo SoC – {months_labels[m]}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            if base_dir is not None:
                fig.savefig(base_dir / f"soc_month_{m:02d}.png", dpi=300)
            if show:
                plt.show()
            else:
                plt.close(fig)
