"""
Core Monte Carlo simulation classes.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..energy_simulator import EnergySystemSimulator
from ..prices import PriceModel

from .finance import _npv, _compute_irr_monthly, _compute_irr_annual

@dataclass
class EconomicConfig:
    """
    Economic parameters for Monte Carlo financial analysis.

    Encapsulates the key financial inputs needed for evaluating PV system
    economics across multiple stochastic scenarios. These parameters determine
    initial costs, simulation granularity, and inflation adjustments for
    real return calculations.

    Attributes:
        investment_eur: Total initial investment cost (EUR).
            Includes all upfront costs: panels, inverter, battery, installation,
            permits, electrical work, etc. This is the negative cashflow at t=0.
            Typical residential PV+battery: 6,000-15,000 EUR.
            Typical residential PV-only: 4,000-8,000 EUR.
            Default: 2500.0 EUR (minimal system).
        n_mc: Number of Monte Carlo simulation paths (integer).
            More paths = better statistical accuracy but longer computation.
            Each path represents one possible future scenario with random
            weather, load, and price variations.
            Typical values:
            - 100-200: Quick analysis (minutes)
            - 500-1000: Production analysis (5-30 minutes)
            - 2000+: High-precision research (hours)
            Default: 200 (good balance for most cases).
        inflation_rate: Annual inflation rate for real return calculations (decimal).
            Used to discount nominal savings to present-day value, accounting
            for purchasing power erosion over time.
            Example: 0.025 = 2.5% annual inflation.
            Typical European values: 0.015-0.035 (1.5%-3.5%).
            Real return = nominal return adjusted by this inflation.
            Default: 0.025 (2.5% per year, conservative estimate).

    Example:
        ```python
        from sim_stochastic_pv.simulation.monte_carlo import EconomicConfig

        # Budget residential system
        config_budget = EconomicConfig(
            investment_eur=6000.0,    # Small 3kW system
            n_mc=500,                  # Moderate precision
            inflation_rate=0.02        # 2% inflation
        )

        # Premium system with high precision
        config_premium = EconomicConfig(
            investment_eur=12000.0,   # Large 6kW + battery
            n_mc=1000,                 # High precision
            inflation_rate=0.025       # 2.5% inflation
        )

        # Quick test run
        config_test = EconomicConfig(
            investment_eur=8000.0,
            n_mc=100,                  # Fast execution
            inflation_rate=0.03
        )
        ```

    Notes:
        - investment_eur should include ALL upfront costs (no recurring)
        - n_mc affects computation time linearly (2x paths = 2x time)
        - inflation_rate impacts "real" metrics but not nominal ones
        - Real metrics are more economically meaningful than nominal
        - For comparison: use same n_mc across scenarios
        - Seed control in run() ensures reproducibility despite randomness
    """
    investment_eur: float = 2500.0
    n_mc: int = 200
    inflation_rate: float = 0.025


@dataclass
class MonteCarloResults:
    """
    Complete results bundle from Monte Carlo simulation.

    Contains all aggregated statistics and raw path data from a Monte Carlo
    analysis run. Provides both summary DataFrames (for easy plotting and
    analysis) and raw arrays (for custom post-processing).

    The results are organized into four main categories:
    1. Financial: Cumulative profit evolution and statistics
    2. Energy: Production, consumption, and grid interaction flows
    3. Battery state: State of charge (SoC) and state of health (SoH)
    4. Raw paths: Individual simulation trajectories for detailed analysis

    Attributes:
        df_profit: Financial performance statistics over time (DataFrame).
            Columns:
            - month_index: 0-based month counter (0 = month 1)
            - year: Year index (0 = first year)
            - month_in_year: Month within year (0-11 for Jan-Dec)
            - prob_gain: Probability of positive cumulative profit (0-1)
            - mean_gain_eur: Mean cumulative profit (nominal EUR)
            - p05_gain_eur: 5th percentile profit (EUR, pessimistic case)
            - p95_gain_eur: 95th percentile profit (EUR, optimistic case)
            - mean_gain_real_eur: Mean real profit (inflation-adjusted EUR)
            - p05_gain_real_eur: 5th percentile real profit (EUR)
            - p95_gain_real_eur: 95th percentile real profit (EUR)
            Shape: (n_months, 10 columns) where n_months = n_years × 12

        df_energy: Energy flow statistics over time (DataFrame).
            Columns:
            - month_index, year, month_in_year: Time indices
            - pv_prod_mean_kwh: Mean monthly PV production (kWh)
            - pv_prod_p05_kwh, pv_prod_p95_kwh: Production percentiles
            - solar_used_mean_kwh: Mean solar energy consumed (direct + battery)
            - solar_used_p05_kwh, solar_used_p95_kwh: Usage percentiles
            - grid_import_mean_kwh: Mean grid import (kWh)
            - grid_import_p05_kwh, grid_import_p95_kwh: Import percentiles
            - savings_mean_kwh: Mean energy savings vs no-PV case
            - savings_p05_kwh, savings_p95_kwh: Savings percentiles
            Shape: (n_months, 15 columns)

        df_soc: Battery state of charge profile for first year (DataFrame).
            Average hourly SoC pattern for each month (first year only).
            Columns:
            - month_in_year: Month index (0-11)
            - hour: Hour of day (0-23)
            - soc_mean: Mean SoC fraction (0-1)
            - soc_p05: 5th percentile SoC
            - soc_p95: 95th percentile SoC
            Shape: (288 rows = 12 months × 24 hours, 5 columns)

        df_soh: Battery state of health degradation over time (DataFrame).
            Tracks battery capacity fade across system lifetime.
            Columns:
            - month_index, year, month_in_year: Time indices
            - soh_mean: Mean SoH fraction (1.0 = new, 0.0 = dead)
            - soh_p05: 5th percentile SoH (worst case)
            - soh_p95: 95th percentile SoH (best case)
            Shape: (n_months, 6 columns)

        monthly_savings_eur_paths: Raw monthly savings for each path (ndarray).
            Nominal EUR savings per month per Monte Carlo path.
            Shape: (n_mc, n_months). Useful for custom statistical analysis.

        monthly_savings_real_eur_paths: Inflation-adjusted savings (ndarray).
            Real EUR savings (discounted by inflation) per month per path.
            Shape: (n_mc, n_months). More economically meaningful than nominal.

        monthly_load_kwh_paths: Monthly electricity consumption (ndarray).
            Total load per month per path. Shape: (n_mc, n_months).
            Varies stochastically due to occupancy patterns.

        irr_annual_paths: Annual IRR for each simulation path (ndarray).
            Internal rate of return (annualized) for each path.
            Shape: (n_mc,). Contains np.nan for paths with no valid IRR.

    Example:
        ```python
        from sim_stochastic_pv.simulation.monte_carlo import MonteCarloSimulator

        # Run simulation (see MonteCarloSimulator.run() for setup)
        results = simulator.run(seed=42)

        # Analyze final profit distribution
        final_month = results.df_profit.iloc[-1]
        print(f"Final mean profit: {final_month['mean_gain_real_eur']:.0f} EUR")
        print(f"Probability profitable: {final_month['prob_gain']:.1%}")

        # Energy statistics
        year_1_energy = results.df_energy[results.df_energy['year'] == 0]
        total_pv_yr1 = year_1_energy['pv_prod_mean_kwh'].sum()
        print(f"Year 1 PV production: {total_pv_yr1:.0f} kWh")

        # Battery degradation
        final_soh = results.df_soh.iloc[-1]
        print(f"Final battery SoH: {final_soh['soh_mean']:.1%}")

        # IRR distribution
        import numpy as np
        valid_irr = results.irr_annual_paths[~np.isnan(results.irr_annual_paths)]
        print(f"Mean IRR: {valid_irr.mean():.2%}")
        print(f"IRR 5th percentile: {np.percentile(valid_irr, 5):.2%}")
        ```

    Notes:
        - All DataFrames indexed by month/hour for easy time-series analysis
        - Percentiles (p05, p95) capture uncertainty range (90% confidence)
        - Real values account for inflation, nominal values do not
        - SoC profile only for first year (representative pattern)
        - SoH tracks full lifetime degradation trajectory
        - Raw path arrays enable custom analysis beyond summary statistics
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
    Monte Carlo simulator for stochastic PV system economic analysis.

    Orchestrates multiple simulation runs to quantify financial risk and
    expected returns for photovoltaic energy systems. Each simulation path
    represents one possible future with random weather, load patterns, and
    price escalation.

    The simulator combines:
    - Energy system simulation (PV production, battery operation, loads)
    - Price modeling (electricity costs with stochastic escalation)
    - Financial analysis (NPV, IRR, cumulative profit)

    Results include statistical distributions (mean, percentiles) of:
    - Cumulative profit over time (nominal and inflation-adjusted)
    - Energy production and consumption
    - Battery state of charge and degradation
    - Internal rate of return (IRR)

    Typical workflow:
    1. Create energy simulator (solar, battery, load models)
    2. Create price model (base price, escalation, volatility)
    3. Configure economics (investment, n_mc paths, inflation)
    4. Run Monte Carlo simulation
    5. Analyze results (profit probability, IRR, energy stats)

    Attributes:
        energy_simulator: EnergySystemSimulator instance for single-path runs.
            Configured with PV system size, battery specs, load profile, etc.
        price_model: PriceModel instance for electricity pricing.
            Handles price escalation and stochastic variation over time.
        economic_config: EconomicConfig with investment cost and simulation params.

    Example:
        ```python
        from sim_stochastic_pv.simulation import (
            EnergySystemSimulator, PriceModel, MonteCarloSimulator,
            EconomicConfig
        )

        # Setup components
        energy_sim = EnergySystemSimulator(...)  # Configure system
        price_model = PriceModel(...)  # Configure pricing
        econ_config = EconomicConfig(
            investment_eur=8000.0,
            n_mc=500,
            inflation_rate=0.025
        )

        # Create and run Monte Carlo simulator
        mc_sim = MonteCarloSimulator(energy_sim, price_model, econ_config)
        results = mc_sim.run(seed=42)

        # Analyze results
        final_profit = results.df_profit.iloc[-1]
        print(f"Mean final profit: {final_profit['mean_gain_real_eur']:.0f} EUR")
        print(f"Prob profitable: {final_profit['prob_gain']:.1%}")

        # Calculate payback time
        profitable = results.df_profit[results.df_profit['mean_gain_real_eur'] > 0]
        if not profitable.empty:
            payback_months = profitable.iloc[0]['month_index']
            print(f"Payback time: {payback_months} months ({payback_months/12:.1f} years)")

        # Plot results
        MonteCarloSimulator.plot_profit_bands(results.df_profit, show=True)
        MonteCarloSimulator.plot_soh_evolution(results.df_soh, show=True)
        ```

    Notes:
        - Each Monte Carlo path is independent (no cross-path correlation)
        - Computation time scales linearly with n_mc
        - Results are reproducible when using same seed
        - Provides both summary statistics and raw path data
        - Static plotting methods available for visualization
    """

    def __init__(
        self,
        energy_simulator: EnergySystemSimulator,
        price_model: PriceModel,
        economic_config: EconomicConfig,
    ) -> None:
        """
        Initialize Monte Carlo simulator with system models and economic parameters.

        Sets up the simulation framework by linking the energy system model,
        price model, and economic configuration. Does not run any simulations -
        call run() to execute the Monte Carlo analysis.

        Args:
            energy_simulator: Configured EnergySystemSimulator instance.
                Must be fully initialized with solar model, battery specs,
                load profile, and simulation duration (n_years).
                This simulator will be called n_mc times with different seeds.
            price_model: Configured PriceModel instance.
                Defines base electricity price and escalation parameters.
                Will be reset for each Monte Carlo path to generate independent
                price trajectories.
            economic_config: Economic parameters for analysis.
                Specifies initial investment, number of Monte Carlo paths,
                and inflation rate for real return calculations.

        Example:
            ```python
            from sim_stochastic_pv.simulation import (
                EnergySystemSimulator, EnergySystemConfig,
                PriceModel, MonteCarloSimulator, EconomicConfig
            )
            import numpy as np

            # Configure energy system
            energy_config = EnergySystemConfig(n_years=20, ...)
            energy_sim = EnergySystemSimulator(energy_config)

            # Configure pricing
            price_model = PriceModel(
                base_price_eur_per_kwh=0.22,
                annual_escalation=0.025,
                use_stochastic=True
            )

            # Configure economics
            econ_config = EconomicConfig(
                investment_eur=9000.0,
                n_mc=500,
                inflation_rate=0.025
            )

            # Create simulator (no simulation yet)
            simulator = MonteCarloSimulator(energy_sim, price_model, econ_config)

            # Now run simulation
            results = simulator.run(seed=123)
            ```

        Notes:
            - Does not validate compatibility between models (user responsibility)
            - energy_simulator.config.n_years determines simulation horizon
            - price_model must support n_years worth of pricing
            - Models are not copied - modifications affect simulation
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
        Execute full Monte Carlo simulation and return comprehensive results.

        Runs n_mc independent simulation paths, each representing a possible
        future scenario with stochastic weather, load, and prices. Aggregates
        results into statistical summaries (mean, percentiles) for financial
        and energy metrics.

        Each simulation path:
        1. Generates random seed for this path
        2. Resets price model with stochastic escalation
        3. Runs energy system simulation for n_years
        4. Calculates monthly savings using current prices
        5. Computes cumulative profit and IRR
        6. Tracks battery degradation (SoH) and state (SoC)

        After all paths complete, aggregates into:
        - Mean values across all paths
        - 5th and 95th percentiles (uncertainty bounds)
        - Probability distributions (e.g., prob of positive profit)

        Args:
            seed: Master random seed for reproducibility (integer).
                All Monte Carlo paths derive their seeds from this value.
                Same seed = identical results (deterministic).
                Different seeds = different statistical outcomes.
                Default: 123.
            progress_callback: Optional callback function for progress updates.
                Signature: callback(iteration: int, total: int, elapsed: float, eta: float)
                Called periodically during simulation for custom progress tracking.
                If None, uses built-in progress bar (if show_progress=True).
            show_progress: Whether to display progress bar in console (bool).
                Only used if progress_callback is None.
                True: Shows text progress bar with ETA.
                False: Silent execution (useful for batch runs).
                Default: True.

        Returns:
            MonteCarloResults: Complete results bundle containing:
                - df_profit: Financial statistics over time
                - df_energy: Energy flow statistics
                - df_soc: Battery SoC profiles (first year)
                - df_soh: Battery degradation over lifetime
                - Raw path arrays for custom analysis
                See MonteCarloResults docstring for full details.

        Example:
            ```python
            # Basic usage
            results = simulator.run(seed=42)

            # Silent execution (no progress bar)
            results = simulator.run(seed=42, show_progress=False)

            # Custom progress tracking
            def my_callback(done, total, elapsed, eta):
                print(f"Progress: {done}/{total} ({100*done/total:.1f}%)")

            results = simulator.run(seed=42, progress_callback=my_callback)

            # Analyze results
            final = results.df_profit.iloc[-1]
            print(f"Final profit: {final['mean_gain_real_eur']:.0f} EUR")
            print(f"Prob profit: {final['prob_gain']:.1%}")

            # Check IRR distribution
            import numpy as np
            valid_irr = results.irr_annual_paths[~np.isnan(results.irr_annual_paths)]
            print(f"Mean IRR: {valid_irr.mean():.2%}")
            print(f"IRR range (5-95): {np.percentile(valid_irr, [5, 95])}")
            ```

        Performance:
            - Computation time ≈ (n_mc × n_years × complexity_factor)
            - Typical timing (single-threaded):
              - 100 paths × 20 years: 1-2 minutes
              - 500 paths × 20 years: 5-10 minutes
              - 1000 paths × 20 years: 10-20 minutes
            - Progress bar updates every 1% completion or 100 paths

        Notes:
            - Each path is independent (embarrassingly parallel problem)
            - Seed ensures reproducibility despite randomness
            - IRR may be nan for some paths (if no valid solution)
            - Real metrics (inflation-adjusted) more meaningful than nominal
            - First year SoC profile representative of typical operation
            - Battery SoH degrades over time (tracked for all years)
            - Results ready for immediate plotting or custom analysis
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

        start_time = time.time()

        # Use tqdm for progress tracking when appropriate
        iterator = range(n_mc)
        if progress_callback is None and show_progress:
            iterator = tqdm(
                iterator,
                desc="Monte Carlo simulation",
                unit="path",
                disable=None  # Auto-detect TTY
            )

        for i in iterator:
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

            # Call progress callback if provided
            if progress_callback is not None:
                iteration_done = i + 1
                elapsed = time.time() - start_time
                frac = iteration_done / n_mc
                eta = (elapsed / frac - elapsed) if frac > 0 else 0.0
                progress_callback(iteration_done, n_mc, elapsed, eta)

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
