"""
Core Monte Carlo simulation classes.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..electrical import aggregate_kpis as _aggregate_electrical_kpis
from ..energy_simulator import EnergySystemSimulator
from ..load_profiles import aggregate_appliances_kpis as _aggregate_appliances_kpis
from ..prices import PriceModel
from ..thermal_load import aggregate_thermal_kpis as _aggregate_thermal_kpis

from .finance import _npv, _compute_irr_monthly, _compute_irr_annual


@dataclass
class TaxBonusConfig:
    """
    Tax bonus configuration for PV system economic analysis (Phase 11).

    Models the Italian "Detrazione fiscale" (tax credit) for PV
    investments: a fraction of the upfront CAPEX is returned to the user
    as an annual income for a fixed number of years. By convention each
    yearly instalment is paid out at the end of the corresponding year
    (December of year 1, December of year 2, ...). For the simulator,
    this means a non-zero cash inflow at months 11, 23, 35, ...

    When ``enabled=False`` (the default) the simulation behaves exactly
    as if no bonus existed: all downstream maths is unchanged. This makes
    the feature strictly additive and back-compatible.

    Attributes:
        enabled: Whether the tax bonus applies (bool).
            If False, ``fraction_of_investment`` and ``duration_years``
            are ignored entirely. Default: False.
        fraction_of_investment: Fraction of CAPEX returned overall (float).
            Decimal value in [0, 1]. Example: 0.50 means 50% of the
            initial investment is returned across the bonus duration.
            Default: 0.50 (matches Italy's standard "Detrazione 50%").
        duration_years: Number of yearly instalments (int).
            Italy's standard scheme spreads the bonus across 10 yearly
            payments. Must be >= 1. If ``duration_years > n_years`` the
            simulator silently pays only the instalments that fit within
            the simulation horizon (no extension). Default: 10.

    Example:
        ```python
        # Italian "Detrazione 50% in 10 anni" — the typical residential case
        bonus = TaxBonusConfig(
            enabled=True,
            fraction_of_investment=0.5,
            duration_years=10,
        )

        # Hypothetical "65% in 5 anni" scheme
        bonus_premium = TaxBonusConfig(
            enabled=True,
            fraction_of_investment=0.65,
            duration_years=5,
        )
        ```

    Notes:
        - Yearly instalment = ``investment_eur * fraction_of_investment
          / duration_years``. Paid at month index ``12k - 1`` for k = 1..K.
        - The bonus is a **nominal** cash inflow: it does not adjust to
          inflation. The simulator divides it by the inflation factor of
          its month (same as for energy savings) to obtain its real
          purchasing power for the ``profit_cum_real`` series.
        - The bonus is included in the IRR cash-flow array, so IRR and
          break-even improve when enabled.
        - Validation lives at the boundary (Pydantic schemas, CLI
          validator). This dataclass trusts its inputs (CLAUDE.md §2.4).
    """
    enabled: bool = False
    fraction_of_investment: float = 0.50
    duration_years: int = 10


@dataclass
class InflationConfig:
    """
    Inflation model parameters for real return calculations (Phase 11).

    Replaces the legacy scalar ``EconomicConfig.inflation_rate`` with a
    richer object that supports both the historical deterministic regime
    and a new path-dependent stochastic regime (Truncated Normal per year).
    The deterministic case is preserved exactly so that pre-Phase-11 runs
    remain byte-identical when no stochastic component is requested.

    The semantics of ``mean`` are identical to the legacy ``inflation_rate``:
    it is the **annual compounded rate** used to discount nominal cash flows
    to present-day purchasing power. The new ``std``, ``min_clip`` and
    ``max_clip`` are only consulted when ``mode='stochastic'``.

    Attributes:
        mode: Sampling regime for the inflation rate (string).
            - ``'deterministic'`` (default): the same constant ``mean`` is
              applied every year, identical to the legacy behaviour. No
              random calls are consumed — this guarantees byte-identical
              results with respect to runs that predate Phase 11.
            - ``'stochastic'``: each Monte Carlo path samples one annual
              rate per year from a Normal(``mean``, ``std``) clipped to
              ``[min_clip, max_clip]``. The path-dependent factors are
              applied to compute the real (inflation-adjusted) cash flow.
        mean: Mean annual inflation rate (decimal, e.g. 0.025 = 2.5%).
            Typical European long-run values: 0.015–0.035.
            Default: 0.025.
        std: Annual standard deviation of the inflation rate (decimal).
            Only used when ``mode='stochastic'``. Set to 0.0 for a
            deterministic-like Stochastic mode (Truncated Normal collapses
            to a delta). Default: 0.0.
        min_clip: Lower bound applied to each sampled annual rate (decimal).
            Prevents extreme deflation tails. Default: -0.02 (i.e. -2%).
        max_clip: Upper bound applied to each sampled annual rate (decimal).
            Prevents extreme hyperinflation tails. Default: 0.10 (i.e. 10%).

    Example:
        ```python
        # Legacy-equivalent deterministic case
        deterministic = InflationConfig(mean=0.025)

        # Realistic Euro-area volatility (≈ historical 2000–2024)
        stochastic = InflationConfig(
            mode='stochastic',
            mean=0.025,
            std=0.015,
            min_clip=-0.01,
            max_clip=0.08,
        )
        ```

    Notes:
        - ``mode='deterministic'`` is the default for backward compatibility.
        - With ``mode='stochastic'`` and ``std=0`` you recover the
          deterministic behaviour, but you DO consume RNG calls — prefer
          ``mode='deterministic'`` if you don't need stochasticity at all.
        - Annual rates are sampled once per (path, year). The same annual
          rate is reused for all 12 months of that year — there is no
          intra-year inflation jitter.
        - The compounded factors are obtained by ``cumprod(1 + r_annual)``
          and then broadcast to monthly granularity via ``np.repeat(..., 12)``.
    """
    mode: Literal["deterministic", "stochastic"] = "deterministic"
    mean: float = 0.025
    std: float = 0.0
    min_clip: float = -0.02
    max_clip: float = 0.10


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
            **Deprecated in Phase 11**: prefer the ``inflation`` field
            (InflationConfig object). When both are provided ``inflation``
            wins; when ``inflation`` is None this scalar is wrapped into
            an ``InflationConfig(mode='deterministic', mean=inflation_rate)``
            at simulation time. Kept here for backward compatibility with
            JSON scenarios and code paths that predate Phase 11.
        inflation: Rich inflation configuration (InflationConfig, optional).
            When provided, takes precedence over ``inflation_rate``. Enables
            the new stochastic regime (Truncated Normal annual rates) used
            by Monte Carlo to widen the real-return p05–p95 band. When
            None (default), the simulator falls back to ``inflation_rate``
            and behaves exactly like pre-Phase-11 code.

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
    inflation: Optional[InflationConfig] = None
    tax_bonus: Optional[TaxBonusConfig] = None


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

        df_price: Electricity price statistics over the Monte Carlo run
            (DataFrame). One row per simulation month with mean and tail
            percentiles of the price drawn from the underlying price model.
            Columns:
            - month_index, year, month_in_year: Time indices
            - price_mean_eur_per_kwh: Mean price across all paths (EUR/kWh)
            - price_p05_eur_per_kwh: 5th percentile (pessimistic tail)
            - price_p95_eur_per_kwh: 95th percentile (optimistic tail)
            Shape: (n_months, 6 columns). For deterministic price models
            (legacy ``EscalatingPriceModel`` with stochastic flag off, or
            GBM with volatility=0) the three series coincide.

        monthly_savings_eur_paths: Raw monthly savings for each path (ndarray).
            Nominal EUR savings per month per Monte Carlo path.
            Shape: (n_mc, n_months). Useful for custom statistical analysis.

        monthly_savings_real_eur_paths: Inflation-adjusted savings (ndarray).
            Real EUR savings (discounted by inflation) per month per path.
            Shape: (n_mc, n_months). More economically meaningful than nominal.

        monthly_load_kwh_paths: Monthly electricity consumption (ndarray).
            Total load per month per path. Shape: (n_mc, n_months).
            Varies stochastically due to occupancy patterns.

        price_paths_eur_per_kwh: Raw electricity price paths (ndarray).
            One row per Monte Carlo path, one column per simulation month.
            Shape: (n_mc, n_months). Captures the full stochastic price
            trajectory of every path, enabling downstream consumers to
            draw fan charts, compute custom percentiles, or sample a
            handful of representative trajectories for visualisation.

        irr_annual_paths: Annual IRR for each simulation path (ndarray).
            Internal rate of return (annualized) for each path.
            Shape: (n_mc,). Contains np.nan for paths with no valid IRR.

        break_even_month_per_path: First month index where cumulative profit
            turns non-negative, per path (ndarray). Shape: (n_mc,).
            Value -1 signals the path never breaks even within the horizon.
            Month index 0 = first simulation month.
            Added in Phase 4 of the roadmap; defaults to None for
            backward-compatible manual construction.

        prob_break_even_within_horizon: Fraction of paths that reach
            break-even at any point within the simulation horizon (float, 0–1).
            Added in Phase 4; defaults to None.

        break_even_month_median: Median break-even month across paths that
            actually break even (float, or None if no path breaks even).
            Added in Phase 4; defaults to None.

        break_even_month_p05: 5th-percentile break-even month (optimistic
            tail, i.e. the earliest break-even scenario). None if fewer
            than one path breaks even. Added in Phase 4; defaults to None.

        break_even_month_p95: 95th-percentile break-even month (pessimistic
            tail, i.e. the slowest-to-break-even scenario). None if no
            path breaks even. Added in Phase 4; defaults to None.

        npv_median_eur: Median final cumulative profit (nominal EUR) across
            all paths at the end of the simulation horizon. Equals
            np.median(profit_cum_paths[:, -1]) and is distinct from
            mean_gain_eur (which is the mean, not the median). Added in
            Phase 4; defaults to None.

        irr_mean: Mean annual IRR across paths that yield a valid finite
            IRR (excluding np.nan paths). None if no valid IRR exists.
            Added in Phase 4; defaults to None.

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
    df_price: Optional[pd.DataFrame] = None
    price_paths_eur_per_kwh: Optional[np.ndarray] = None
    # Phase 4 — break-even and aggregate KPI fields.  All optional so that
    # legacy code and tests that construct MonteCarloResults by hand continue
    # to work without providing these values.
    break_even_month_per_path: Optional[np.ndarray] = None
    prob_break_even_within_horizon: Optional[float] = None
    break_even_month_median: Optional[float] = None
    break_even_month_p05: Optional[float] = None
    break_even_month_p95: Optional[float] = None
    npv_median_eur: Optional[float] = None
    irr_mean: Optional[float] = None
    # Phase 11 — inflation fields. Populated only when
    # ``InflationConfig.mode='stochastic'``; remain None in deterministic
    # runs so legacy clients that construct results by hand are unaffected.
    inflation_annual_rates_paths: Optional[np.ndarray] = None
    df_inflation: Optional[pd.DataFrame] = None
    # Phase 11 — tax bonus diagnostics. ``bonus_per_month_eur`` is the
    # sparse vector applied to every Monte Carlo path; ``tax_bonus_total_eur``
    # is the total bonus disbursed during the simulation horizon (sum of
    # the sparse vector). Both default to harmless values so that legacy
    # clients which build MonteCarloResults by hand keep working.
    bonus_per_month_eur: Optional[np.ndarray] = None
    tax_bonus_total_eur: float = 0.0
    # Phase 16 — opt-in electrical KPIs. When the scenario activates the
    # detailed electrical model (``electrical.mode='mppt_window'``) the
    # simulator collects per-path :class:`ElectricalKPIs` instances and
    # exposes both the raw list and an aggregated dict here. Both fields
    # are ``None`` whenever the model is disabled, so legacy consumers
    # stay unaffected.
    electrical_kpis_per_path: Optional[list] = None
    electrical_kpis_summary: Optional[dict] = None
    # Phase 17 — opt-in thermal (HVAC) KPIs. Populated when the scenario
    # enables ``thermal_load.enabled=true``. Both ``None`` otherwise.
    thermal_kpis_per_path: Optional[list] = None
    thermal_kpis_summary: Optional[dict] = None
    # Phase 17-bis — opt-in appliance event-based KPIs. Populated when
    # the scenario activates ``load_profile.appliances.enabled=true``.
    appliances_kpis_per_path: Optional[list] = None
    appliances_kpis_summary: Optional[dict] = None


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

    @staticmethod
    def _build_tax_bonus_per_month(
        cfg: EconomicConfig, n_months: int
    ) -> np.ndarray:
        """
        Build the sparse monthly cash-inflow vector for the tax bonus.

        When the bonus is disabled (or no TaxBonusConfig is attached to
        ``cfg``), returns a zero vector — so the rest of the cash-flow
        pipeline is unaffected. When enabled, places one yearly instalment
        at the last month of each of the first ``min(duration_years,
        n_years)`` years (month indices 11, 23, 35, ...).

        Args:
            cfg: EconomicConfig. ``investment_eur`` and ``tax_bonus`` are read.
            n_months: Simulation horizon in months.

        Returns:
            np.ndarray of shape (n_months,), dtype float64. Zeros everywhere
            except at month indices ``12k - 1`` for ``k = 1..K``, where
            ``K = min(tax_bonus.duration_years, n_months // 12)``.

        Notes:
            - Total disbursed = ``investment_eur * fraction * K /
              duration_years``. When ``duration_years <= n_years`` this
              equals exactly ``investment_eur * fraction``.
            - When ``duration_years > n_years`` the user loses the
              instalments that fall outside the horizon — documented and
              tested.
        """
        bonus = np.zeros(n_months)
        tb = cfg.tax_bonus
        if tb is None or not tb.enabled:
            return bonus
        n_years_in_horizon = n_months // 12
        n_payments = min(tb.duration_years, n_years_in_horizon)
        if n_payments <= 0:
            return bonus
        yearly_amount = (
            cfg.investment_eur * tb.fraction_of_investment / tb.duration_years
        )
        for k in range(1, n_payments + 1):
            bonus[k * 12 - 1] = yearly_amount
        return bonus

    @staticmethod
    def _resolve_inflation_config(cfg: EconomicConfig) -> InflationConfig:
        """
        Pick the effective InflationConfig for a simulation run.

        When ``cfg.inflation`` is provided it wins. Otherwise the legacy
        scalar ``cfg.inflation_rate`` is wrapped into a deterministic
        InflationConfig so that pre-Phase-11 scenarios keep working
        unchanged.

        Args:
            cfg: EconomicConfig to inspect.

        Returns:
            InflationConfig: the configuration to use downstream. Never None.

        Notes:
            - In ``mode='deterministic'`` the result is byte-identical to
              the legacy ``np.power(1 + inflation_rate, years)`` factor
              computation (same ``mean``, no random draws).
            - This helper does NOT consume RNG state — safe to call before
              any path-specific RNG setup.
        """
        if cfg.inflation is not None:
            return cfg.inflation
        return InflationConfig(mode="deterministic", mean=cfg.inflation_rate)

    @staticmethod
    def _build_inflation_factors_stochastic(
        inflation_cfg: InflationConfig,
        rng: np.random.Generator,
        n_mc: int,
        n_years: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pre-sample path-dependent inflation factors for all Monte Carlo paths.

        Draws ``n_mc × n_years`` annual rates from a Truncated Normal
        distribution N(mean, std) clipped to ``[min_clip, max_clip]``,
        composes them into cumulative factors (year 0 → factor 1.0; year k
        → product of ``(1 + r_j)`` for ``j ∈ [1..k]``), and expands the
        result to monthly granularity by repeating each annual factor 12
        times (no intra-year jitter).

        Sampling happens **once, upfront**, before the path loop. This
        avoids interleaving the inflation RNG with the per-path RNG used by
        the energy/price simulators, keeping behaviour predictable and
        easy to reason about.

        Args:
            inflation_cfg: InflationConfig with ``mode='stochastic'``. The
                ``mean``, ``std``, ``min_clip`` and ``max_clip`` fields are
                consulted.
            rng: NumPy ``Generator`` used to draw the rates. Mutated.
            n_mc: Number of Monte Carlo paths.
            n_years: Simulation horizon in whole years.

        Returns:
            Tuple ``(annual_rates_paths, monthly_factors_paths)``:
                - ``annual_rates_paths``: float64 array of shape
                  ``(n_mc, n_years)`` with the clipped annual rates
                  (decimal). Preserved verbatim for diagnostic plotting
                  (fan chart of expected inflation).
                - ``monthly_factors_paths``: float64 array of shape
                  ``(n_mc, n_months)`` with cumulative factors at monthly
                  granularity. Factor for year 0 is 1.0 for all 12 months.

        Notes:
            - "Truncated Normal" here is implemented as
              ``np.clip(N(mean, std), min_clip, max_clip)``. This is biased
              for very wide std vs. tight clips (mass piles up on the
              boundaries), but acceptable for the inflation range we care
              about. Alternative: rejection sampling via ``scipy.stats.
              truncnorm`` — overkill for the current use case.
            - Calling this helper consumes ``n_mc * n_years`` standard
              normal draws from ``rng``.
        """
        n_months = n_years * 12
        annual_rates = rng.normal(
            loc=inflation_cfg.mean,
            scale=inflation_cfg.std,
            size=(n_mc, n_years),
        )
        annual_rates = np.clip(
            annual_rates, inflation_cfg.min_clip, inflation_cfg.max_clip
        )
        # Cumulative factor up to and INCLUDING year k. Year 0 keeps
        # factor 1.0 by construction (we prepend a column of ones).
        # Note that the convention matches the deterministic helper:
        # months in year 0 get factor 1.0, months in year 1 get (1 + r_1),
        # months in year 2 get (1 + r_1)(1 + r_2), and so on.
        ones_col = np.ones((n_mc, 1))
        cumulative_year_end = np.cumprod(1.0 + annual_rates, axis=1)
        # We want the factor that APPLIES TO year k (i.e. cumulative product
        # up to year k-1 for k >= 1, and 1.0 for k=0). So shift by one
        # column to the right and drop the last cumulative column.
        cumulative_per_year = np.concatenate(
            [ones_col, cumulative_year_end[:, :-1]], axis=1
        )
        monthly_factors = np.repeat(cumulative_per_year, 12, axis=1)
        return annual_rates, monthly_factors

    @staticmethod
    def _build_inflation_factors_deterministic(
        mean_rate: float, n_months: int
    ) -> np.ndarray:
        """
        Compute deterministic cumulative inflation factors at monthly granularity.

        Returns the same vector as the legacy ``np.power(1 + r, years)``
        computation: factor 1.0 for all 12 months of year 0, ``(1+r)`` for
        year 1, ``(1+r)^2`` for year 2, etc. This guarantees byte-identical
        real cash flows with respect to runs that predate Phase 11 when
        ``mean_rate`` equals the legacy ``EconomicConfig.inflation_rate``.

        Args:
            mean_rate: Constant annual inflation rate (decimal).
            n_months: Length of the output vector. Must be a multiple of 12
                for the year-step semantics to hold cleanly, but partial
                years are tolerated (last bucket truncated).

        Returns:
            np.ndarray of shape (n_months,), dtype float64, with the
            cumulative inflation factor at each month.

        Example:
            ```python
            f = MonteCarloSimulator._build_inflation_factors_deterministic(
                mean_rate=0.025, n_months=24
            )
            # f[0:12] == 1.0, f[12:24] == 1.025
            ```
        """
        years = np.arange(n_months) // 12
        return np.power(1.0 + mean_rate, years)

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
        # Capture the price drawn from the price model for every (path, month):
        # this is what makes the fan chart of simulated prices in the UI
        # possible (Phase 3 of the roadmap).
        price_paths = np.zeros((n_mc, n_months))

        profit_cum_real_paths = np.zeros((n_mc, n_months))
        soh_paths = np.zeros((n_mc, n_months))
        soc_profiles_paths = np.zeros((n_mc, 12, 24))
        irr_annual_paths = np.full(n_mc, np.nan)

        months = np.arange(n_months)
        years = months // 12
        month_in_year = months % 12
        # Phase 11 — inflation factors are resolved via InflationConfig.
        # In ``mode='deterministic'`` the resulting vector is byte-identical
        # to the legacy ``np.power(1 + inflation_rate, years)`` formula
        # (broadcast to all paths), so runs that predate Phase 11 are
        # unaffected. In ``mode='stochastic'`` we pre-sample one annual rate
        # per (path, year) BEFORE the per-path loop so the path-specific
        # RNG used by the energy/price simulators is not contaminated by
        # inflation draws.
        # Phase 11 — tax bonus sparse vector (zeros if disabled).
        # Same vector for every path: it depends only on the configured
        # investment, fraction and duration, not on the stochastic state.
        bonus_per_month = self._build_tax_bonus_per_month(cfg, n_months)
        tax_bonus_total_eur = float(bonus_per_month.sum())

        inflation_cfg = self._resolve_inflation_config(cfg)
        if inflation_cfg.mode == "deterministic":
            inflation_factors_path0 = self._build_inflation_factors_deterministic(
                inflation_cfg.mean, n_months
            )
            inflation_factors_paths = np.broadcast_to(
                inflation_factors_path0, (n_mc, n_months)
            )
            inflation_annual_rates_paths = None  # not used downstream
        else:
            inflation_annual_rates_paths, inflation_factors_paths = (
                self._build_inflation_factors_stochastic(
                    inflation_cfg, rng_global, n_mc, n_years
                )
            )

        start_time = time.time()

        # Phase 16 — collect per-path electrical KPIs when the detailed
        # electrical model is wired. Remains an empty list (then None)
        # in legacy runs.
        electrical_kpis_per_path: list = []
        # Phase 17 — same idea for HVAC KPIs.
        thermal_kpis_per_path: list = []
        # Phase 17-bis — and the appliance event-based KPIs.
        appliances_kpis_per_path: list = []

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
                price_paths[i, m] = price
                monthly_savings_eur[m] = monthly_savings_kwh[m] * price

            # Phase 11 — fold the tax bonus into the savings stream BEFORE
            # computing cumulative profit and IRR. The bonus is a nominal
            # cash inflow and naturally propagates into both the nominal
            # and the inflation-adjusted curves (the latter via division
            # by inflation_factors_paths[i] below).
            monthly_savings_eur = monthly_savings_eur + bonus_per_month

            profit_cum = -cfg.investment_eur + np.cumsum(monthly_savings_eur)
            monthly_savings_real = monthly_savings_eur / inflation_factors_paths[i]
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

            # Phase 16 — capture the electrical KPI snapshot from the
            # per-path run (None when the model is off).
            kpis_path = getattr(self.energy_simulator, "last_electrical_kpis", None)
            if kpis_path is not None:
                electrical_kpis_per_path.append(kpis_path)
            # Phase 17 — capture the HVAC KPI snapshot (None when off).
            thermal_path_kpis = getattr(self.energy_simulator, "last_thermal_kpis", None)
            if thermal_path_kpis is not None:
                thermal_kpis_per_path.append(thermal_path_kpis)
            # Phase 17-bis — and the appliance KPI snapshot.
            appliances_path_kpis = getattr(
                self.energy_simulator, "last_appliances_kpis", None
            )
            if appliances_path_kpis is not None:
                appliances_kpis_per_path.append(appliances_path_kpis)

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

        # Phase 3 — price path statistics (mean ± p05/p95) per simulation
        # month. For deterministic price models the three columns coincide.
        price_mean, price_p05, price_p95 = stats(price_paths)
        df_price = pd.DataFrame(
            {
                "month_index": months,
                "year": years,
                "month_in_year": month_in_year,
                "price_mean_eur_per_kwh": price_mean,
                "price_p05_eur_per_kwh": price_p05,
                "price_p95_eur_per_kwh": price_p95,
            }
        )

        # Phase 4 — break-even analysis per path.
        #
        # For each Monte Carlo path find the first month_index at which the
        # cumulative nominal profit turns non-negative (≥ 0).  A path that
        # never recovers the investment within the simulation horizon receives
        # value -1 so that callers can distinguish it from "month 0" (which
        # would mean savings covered the investment in the very first month,
        # theoretically possible only when investment_eur = 0).
        #
        # Vectorised strategy:
        #   be_mask          – boolean (n_mc × n_months), True = profitable
        #   ever_break_even  – (n_mc,) True = the path breaks even at some point
        #   np.argmax along axis=1 returns the first True index for rows that
        #   have at least one True; for rows with no True it returns 0 (wrong),
        #   but those rows are masked away by np.where.
        be_mask = profit_cum_paths >= 0.0  # shape (n_mc, n_months)
        ever_break_even: np.ndarray = be_mask.any(axis=1)  # shape (n_mc,)
        first_be_month_raw: np.ndarray = np.argmax(be_mask, axis=1)
        break_even_month_per_path: np.ndarray = np.where(
            ever_break_even, first_be_month_raw, -1
        ).astype(np.int64)

        prob_break_even_within_horizon: float = float(ever_break_even.mean())

        valid_be_months = break_even_month_per_path[ever_break_even]
        if len(valid_be_months) > 0:
            break_even_month_median: Optional[float] = float(np.median(valid_be_months))
            break_even_month_p05: Optional[float] = float(np.percentile(valid_be_months, 5))
            break_even_month_p95: Optional[float] = float(np.percentile(valid_be_months, 95))
        else:
            break_even_month_median = None
            break_even_month_p05 = None
            break_even_month_p95 = None

        # Median final cumulative profit across all paths (nominal EUR).
        npv_median_eur: float = float(np.median(profit_cum_paths[:, -1]))

        # Mean annual IRR, excluding paths where no valid IRR was found.
        valid_irr = irr_annual_paths[~np.isnan(irr_annual_paths)]
        irr_mean: Optional[float] = float(valid_irr.mean()) if len(valid_irr) > 0 else None

        # Phase 11 — inflation diagnostic DataFrame. Built only when the
        # simulator ran in stochastic mode; for deterministic mode there is
        # nothing to summarise beyond the constant rate already stored in
        # the InflationConfig itself.
        df_inflation: Optional[pd.DataFrame] = None
        if inflation_annual_rates_paths is not None:
            year_idx = np.arange(n_years)
            mean_rate = inflation_annual_rates_paths.mean(axis=0)
            p05_rate = np.percentile(inflation_annual_rates_paths, 5, axis=0)
            p95_rate = np.percentile(inflation_annual_rates_paths, 95, axis=0)
            # Compose the per-year cumulative factor that APPLIES to each
            # year (year 0 -> 1.0; year k -> prod(1+r_1..r_k)).
            ones_col = np.ones((n_mc, 1))
            cumulative_year_end = np.cumprod(
                1.0 + inflation_annual_rates_paths, axis=1
            )
            cumulative_per_year = np.concatenate(
                [ones_col, cumulative_year_end[:, :-1]], axis=1
            )
            mean_factor = cumulative_per_year.mean(axis=0)
            p05_factor = np.percentile(cumulative_per_year, 5, axis=0)
            p95_factor = np.percentile(cumulative_per_year, 95, axis=0)
            df_inflation = pd.DataFrame(
                {
                    "year": year_idx,
                    "mean_rate": mean_rate,
                    "p05_rate": p05_rate,
                    "p95_rate": p95_rate,
                    "mean_factor": mean_factor,
                    "p05_factor": p05_factor,
                    "p95_factor": p95_factor,
                }
            )

        return MonteCarloResults(
            df_profit=df_profit,
            df_energy=df_energy,
            df_soc=df_soc,
            df_soh=df_soh,
            df_price=df_price,
            monthly_savings_eur_paths=savings_eur_paths,
            monthly_savings_real_eur_paths=savings_real_eur_paths,
            monthly_load_kwh_paths=load_kwh_paths,
            price_paths_eur_per_kwh=price_paths,
            irr_annual_paths=irr_annual_paths,
            # Phase 4 — break-even and aggregate KPIs
            break_even_month_per_path=break_even_month_per_path,
            prob_break_even_within_horizon=prob_break_even_within_horizon,
            break_even_month_median=break_even_month_median,
            break_even_month_p05=break_even_month_p05,
            break_even_month_p95=break_even_month_p95,
            npv_median_eur=npv_median_eur,
            irr_mean=irr_mean,
            # Phase 11 — inflation diagnostics (None in deterministic mode).
            inflation_annual_rates_paths=inflation_annual_rates_paths,
            df_inflation=df_inflation,
            # Phase 11 — tax bonus diagnostics. Both are zero/None-safe
            # when the bonus is disabled, so downstream code can read them
            # without conditionals.
            bonus_per_month_eur=bonus_per_month,
            tax_bonus_total_eur=tax_bonus_total_eur,
            # Phase 16 — opt-in electrical KPIs. When the model is off
            # the lists are empty and we leave both fields as None for
            # backward-compat with legacy consumers.
            electrical_kpis_per_path=(
                electrical_kpis_per_path if electrical_kpis_per_path else None
            ),
            electrical_kpis_summary=(
                _aggregate_electrical_kpis(electrical_kpis_per_path)
                if electrical_kpis_per_path
                else None
            ),
            # Phase 17 — opt-in thermal (HVAC) KPIs. Same legacy contract.
            thermal_kpis_per_path=(
                thermal_kpis_per_path if thermal_kpis_per_path else None
            ),
            thermal_kpis_summary=(
                _aggregate_thermal_kpis(thermal_kpis_per_path)
                if thermal_kpis_per_path
                else None
            ),
            # Phase 17-bis — opt-in appliance event-based KPIs.
            appliances_kpis_per_path=(
                appliances_kpis_per_path if appliances_kpis_per_path else None
            ),
            appliances_kpis_summary=(
                _aggregate_appliances_kpis(appliances_kpis_per_path)
                if appliances_kpis_per_path
                else None
            ),
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
