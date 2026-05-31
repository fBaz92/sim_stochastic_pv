"""
Electricity price escalation and stochastic modeling utilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


class PriceModel(ABC):
    """
    Abstract base class for electricity price modeling.

    Defines the interface for electricity price models used in economic
    simulations. Implementations provide price-per-kWh as a function of
    year and month, enabling modeling of price escalation, seasonal
    variation, and stochastic uncertainty.

    The interface supports both deterministic and stochastic pricing:
    - Deterministic: Fixed escalation rate (e.g., 2% annual increase)
    - Stochastic: Random variations around mean escalation (Monte Carlo)

    Subclasses must implement:
    - get_price(): Return price for specific (year, month)
    - reset_for_run(): Optional initialization before each Monte Carlo path

    Example subclasses:
    - EscalatingPriceModel: Compound annual escalation + seasonal factors
    - FixedPriceModel: Constant price (no escalation)
    - HistoricalPriceModel: Replay actual historical prices

    Example:
        ```python
        # Using a concrete implementation
        from sim_stochastic_pv.simulation.prices import EscalatingPriceModel

        price_model = EscalatingPriceModel(
            base_price_eur_per_kwh=0.22,
            annual_escalation=0.025
        )

        # Get price for first month of third year
        price = price_model.get_price(year_index=2, month_in_year=0)
        print(f"Price: {price:.4f} EUR/kWh")

        # Reset for Monte Carlo path
        import numpy as np
        rng = np.random.default_rng(seed=42)
        price_model.reset_for_run(rng=rng, n_years=20)
        ```

    Notes:
        - year_index is 0-based (0 = first year, 1 = second year, etc.)
        - month_in_year is 0-based (0 = Jan, 11 = Dec)
        - Prices in EUR/kWh (or other consistent currency unit)
        - reset_for_run() enables stochastic price paths in Monte Carlo
    """

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Prepare price model for a new simulation run (optional hook).

        Called before each Monte Carlo simulation path to initialize or
        reset any stochastic state. The default implementation is a no-op,
        suitable for deterministic price models.

        Stochastic models should override this to:
        1. Store the provided random number generator
        2. Generate price trajectory for n_years
        3. Reset any internal state from previous runs

        Args:
            rng: Random number generator for stochastic price variation.
                If None, model should use internal/default generator.
                Provided by Monte Carlo simulator to ensure reproducibility.
            n_years: Number of simulation years for which to prepare prices.
                If None, model may skip precomputation and generate on-demand.

        Example:
            ```python
            # Stochastic model implementation
            class StochasticPriceModel(PriceModel):
                def reset_for_run(self, rng=None, n_years=None):
                    if rng is not None:
                        self._rng = rng
                    if n_years is not None:
                        # Precompute stochastic price path
                        self._prices = self._generate_path(n_years)

            # Usage in Monte Carlo
            for mc_path in range(n_mc):
                rng = np.random.default_rng(seed=path_seed)
                price_model.reset_for_run(rng=rng, n_years=20)
                # Run simulation with this price path...
            ```

        Notes:
            - Default implementation does nothing (for deterministic models)
            - Stochastic models must store rng for get_price() calls
            - Called once per Monte Carlo path
            - Should be fast (precomputation recommended)
        """
        return

    @abstractmethod
    def get_price(self, year_index: int, month_in_year: int) -> float:
        """
        Get electricity price for a specific year and month.

        Returns the price per kWh at the specified time point. This is the
        core method called repeatedly during simulation to determine energy
        costs for each month.

        Args:
            year_index: Simulation year (0-based integer).
                0 = first year, 1 = second year, etc.
                Typically ranges from 0 to n_years-1 (e.g., 0-19 for 20 years).
            month_in_year: Month within the year (0-based integer, 0-11).
                0 = January, 1 = February, ..., 11 = December.

        Returns:
            float: Electricity price in EUR per kWh (or currency unit).
                Must be positive. Typical residential values: 0.15-0.35 EUR/kWh.

        Raises:
            NotImplementedError: If subclass doesn't implement this method.

        Example:
            ```python
            # Implementation in a subclass
            class SimpleEscalatingPrice(PriceModel):
                def __init__(self, base_price, escalation_rate):
                    self.base_price = base_price
                    self.escalation_rate = escalation_rate

                def get_price(self, year_index, month_in_year):
                    # Simple compound escalation
                    return self.base_price * (1 + self.escalation_rate) ** year_index

            # Usage
            model = SimpleEscalatingPrice(base_price=0.20, escalation_rate=0.02)

            # Year 0, January: 0.20 EUR/kWh
            price_y0 = model.get_price(0, 0)

            # Year 5, June: 0.20 × 1.02^5 ≈ 0.221 EUR/kWh
            price_y5 = model.get_price(5, 5)
            ```

        Notes:
            - Called many times during simulation (n_years × 12 × n_mc times)
            - Should be fast (avoid heavy computation; precompute in reset_for_run)
            - Return value used for monthly electricity cost calculation
            - Must be deterministic for given (year, month) within a run
            - Stochastic variation comes from reset_for_run(), not here
        """
        raise NotImplementedError

    def get_price_hourly(
        self, year_index: int, month_in_year: int, hour_of_day: int
    ) -> float:
        """
        Get electricity price for a specific year, month, and hour of day.

        Extends the monthly :meth:`get_price` with an hour-of-day axis so that
        callers which value energy flows at hourly resolution (e.g. self-
        consumption against an intraday retail tariff) have a single entry
        point. The base-class default ignores ``hour_of_day`` and returns the
        monthly price unchanged for all 24 hours: a model with no intraday
        structure is, by construction, flat across the day. This keeps any
        existing monthly-only consumer **byte-identical** — every hour of a
        month carries exactly ``get_price(year, month)``.

        Subclasses that genuinely carry an intraday price shape (for instance
        a retail tariff derived from an hourly wholesale surface) may override
        this method; they must keep :meth:`get_price` as the monthly aggregate
        so the two views stay mutually consistent.

        Args:
            year_index: Simulation year (0-based integer). Same semantics as
                :meth:`get_price`.
            month_in_year: Month within the year (0-based, 0 = January,
                11 = December). Same semantics as :meth:`get_price`.
            hour_of_day: Hour of day (0-based integer, 0 = 00:00–01:00,
                23 = 23:00–24:00). Ignored by the default implementation.

        Returns:
            float: Electricity price in EUR per kWh for that (year, month,
                hour). For the default implementation this equals
                ``get_price(year_index, month_in_year)`` for every hour.

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.prices import EscalatingPriceModel

            model = EscalatingPriceModel(use_stochastic_escalation=False)
            model.reset_for_run(rng=np.random.default_rng(0), n_years=5)

            monthly = model.get_price(2, 0)            # year 2, January
            hourly = [model.get_price_hourly(2, 0, h) for h in range(24)]
            assert all(abs(h - monthly) < 1e-12 for h in hourly)  # flat day
            ```

        Notes:
            - Default is intentionally flat across the day (no intraday shape).
            - Overriding subclasses must preserve the monthly-aggregate
              invariant described above.
            - Like :meth:`get_price`, this is deterministic within a single
              Monte Carlo path once ``reset_for_run`` has completed.
        """
        return self.get_price(year_index, month_in_year)


class EscalatingPriceModel(PriceModel):
    """
    Electricity price model with compound escalation and seasonal variation.

    Models realistic electricity price evolution over time incorporating:
    1. Base price: Starting price per kWh (year 0)
    2. Annual escalation: Compound growth year-over-year (e.g., inflation)
    3. Seasonal variation: Monthly multipliers (higher in winter, lower in summer)
    4. Stochastic uncertainty: Random variations around mean escalation (optional)

    Price Formula:
        price(year, month) = base_price × yearly_factor(year) × (1 + seasonal[month])

    Where:
        yearly_factor(year) = cumulative product of (1 + escalation + variation[year])

    The stochastic component adds realism by modeling escalation uncertainty,
    representing unpredictable market dynamics, policy changes, and energy
    crises. Each Monte Carlo path gets a different escalation trajectory.

    Attributes:
        base_price: Initial electricity price (EUR/kWh) at year 0.
        annual_escalation: Mean annual escalation rate (decimal, e.g., 0.02 = 2%).
        seasonal_factors: Array of 12 monthly factors (additive, typically ±5%).
        use_stochastic_escalation: Whether to add random variations to escalation.
        escalation_variation_percentiles: (min, max) bounds for random variations.

    Example:
        ```python
        from sim_stochastic_pv.simulation.prices import EscalatingPriceModel

        # Conservative deterministic model
        model_det = EscalatingPriceModel(
            base_price_eur_per_kwh=0.22,
            annual_escalation=0.02,  # 2% per year
            use_stochastic_escalation=False
        )

        # Stochastic model with uncertainty
        model_stoch = EscalatingPriceModel(
            base_price_eur_per_kwh=0.22,
            annual_escalation=0.025,  # 2.5% mean escalation
            use_stochastic_escalation=True,
            escalation_variation_percentiles=(-0.05, 0.05)  # ±5% variation
        )

        # Year 10 price (deterministic)
        # Base: 0.22, After 10 years @ 2%: 0.22 × 1.02^10 ≈ 0.268 EUR/kWh
        price_y10 = model_det.get_price(10, 0)

        # Custom seasonal pattern (higher in winter)
        model_seasonal = EscalatingPriceModel(
            base_price_eur_per_kwh=0.20,
            annual_escalation=0.02,
            seasonal_factors=[
                0.10, 0.08, 0.05, 0.02, 0.00, -0.02,  # Jan-Jun
                -0.03, -0.02, 0.00, 0.03, 0.06, 0.10   # Jul-Dec
            ]
        )
        # January: +10%, July: -3% vs annual average
        ```

    Default Seasonal Pattern:
        Month | Factor | Effect
        ------|--------|--------
        Jan   | +5%    | Winter peak
        Feb   | +4%    | High winter demand
        Mar   | +2%    | Shoulder season
        Apr   |  0%    | Mild weather
        May   | -2%    | Spring
        Jun   | -3%    | Summer low
        Jul   | -3%    | Summer low
        Aug   | -2%    | Summer
        Sep   |  0%    | Shoulder season
        Oct   | +2%    | Fall
        Nov   | +4%    | Early winter
        Dec   | +5%    | Winter peak

    Notes:
        - Escalation compounds annually: price_year_N = price_year_0 × (1+rate)^N
        - Stochastic variation sampled from normal distribution (clipped to bounds)
        - Seasonal factors are ADDITIVE: price × (1 + factor), not multiplicative
        - Default seasonality mimics European residential patterns
        - For Monte Carlo, call reset_for_run() before each simulation path
    """

    def __init__(
        self,
        base_price_eur_per_kwh: float = 0.25,
        annual_escalation: float = 0.02,
        seasonal_factors: List[float] | None = None,
        use_stochastic_escalation: bool = True,
        escalation_variation_percentiles: Tuple[float, float] = (-0.05, 0.05),
    ) -> None:
        """
        Initialize escalating price model with escalation and seasonality.

        Sets up the pricing model with base price, escalation rate, seasonal
        patterns, and optional stochastic variation parameters. Does not
        generate price paths yet - call reset_for_run() to prepare for simulation.

        Args:
            base_price_eur_per_kwh: Base electricity price (EUR/kWh).
                Starting price at year 0, month 0 (before seasonal adjustment).
                Typical European residential: 0.18-0.30 EUR/kWh.
                Default: 0.25 EUR/kWh.
            annual_escalation: Mean annual price escalation rate (decimal).
                Compound growth rate applied each year.
                Typical values: 0.015-0.035 (1.5%-3.5% annual).
                Should approximate long-term inflation + energy market trends.
                Default: 0.02 (2% per year).
            seasonal_factors: Monthly price factors (12 values, Jan-Dec).
                Each factor is additive: price × (1 + factor).
                Examples: 0.05 = +5% above annual average, -0.03 = -3% below.
                If None, uses default European residential pattern (higher winter).
                Must have exactly 12 values if provided.
            use_stochastic_escalation: Enable stochastic escalation variation (bool).
                True: Add random year-to-year variation around mean escalation.
                False: Use fixed escalation rate (deterministic).
                Default: True (recommended for Monte Carlo).
            escalation_variation_percentiles: Bounds for stochastic variation (tuple).
                (min, max) range for random escalation perturbations.
                Values are additive to annual_escalation.
                Example: (-0.05, 0.05) means escalation can vary from
                (annual_escalation - 5%) to (annual_escalation + 5%) each year.
                First value must be negative, second positive.
                Default: (-0.05, 0.05) (±5% variation).

        Raises:
            ValueError: If seasonal_factors provided but not length 12.
            ValueError: If escalation_variation_percentiles invalid (see _build_yearly_factors).

        Example:
            ```python
            # Minimal setup (all defaults)
            model = EscalatingPriceModel()

            # Conservative residential model
            model_conservative = EscalatingPriceModel(
                base_price_eur_per_kwh=0.20,
                annual_escalation=0.015,  # 1.5% escalation
                use_stochastic_escalation=False  # No uncertainty
            )

            # High-uncertainty commercial model
            model_commercial = EscalatingPriceModel(
                base_price_eur_per_kwh=0.18,
                annual_escalation=0.03,  # 3% mean escalation
                use_stochastic_escalation=True,
                escalation_variation_percentiles=(-0.10, 0.10)  # ±10% variation
            )

            # Flat seasonal pattern (no seasonality)
            model_flat = EscalatingPriceModel(
                base_price_eur_per_kwh=0.22,
                annual_escalation=0.02,
                seasonal_factors=[0.0] * 12  # No seasonal variation
            )
            ```

        Notes:
            - Initialization does NOT generate price trajectories
            - Call reset_for_run() before using in Monte Carlo simulation
            - Stochastic variation only applied if use_stochastic_escalation=True
            - Seasonal factors applied in get_price(), not precomputed
            - Default seasonal pattern mimics European residential consumption
        """
        self.base_price = base_price_eur_per_kwh
        self.annual_escalation = annual_escalation
        self.use_stochastic_escalation = use_stochastic_escalation
        self.escalation_variation_percentiles = escalation_variation_percentiles

        if seasonal_factors is None:
            self.seasonal_factors = np.array(
                [0.05, 0.04, 0.02, 0.0, -0.02, -0.03,
                 -0.03, -0.02, 0.0, 0.02, 0.04, 0.05]
            )
        else:
            if len(seasonal_factors) != 12:
                raise ValueError("seasonal_factors must have length 12")
            self.seasonal_factors = np.array(seasonal_factors)

        self._yearly_factors: np.ndarray | None = None
        self._rng: np.random.Generator | None = None
        self._fallback_rng = np.random.default_rng()

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Prepare price model for new Monte Carlo simulation path.

        Generates a new stochastic price trajectory for the upcoming simulation
        run. This method is called once before each Monte Carlo path to ensure
        independent price realizations across different scenarios.

        The method precomputes yearly price escalation factors (base_price ×
        cumulative_escalation) for all simulation years, incorporating random
        variations if stochastic escalation is enabled. This precomputation
        improves performance during simulation (get_price() becomes a simple
        array lookup instead of repeated calculations).

        Behavior:
        - If n_years is None: Clears cached factors (lazy evaluation mode)
        - If n_years provided: Generates full price trajectory for all years
        - If rng is None: Uses internal/fallback RNG (not recommended for MC)
        - If rng provided: Uses it for reproducible stochastic variations

        Args:
            rng: Random number generator for stochastic price variation.
                If provided, used to generate random escalation variations
                for this simulation path. If None, uses internal fallback RNG.
                For reproducible Monte Carlo, pass a seeded generator.
                Type: numpy.random.Generator (e.g., np.random.default_rng(seed))
            n_years: Number of simulation years to prepare prices for.
                Determines size of precomputed price trajectory array.
                Should match the simulation horizon in MonteCarloSimulator.
                If None, skips precomputation (factors computed on-demand).
                Typical values: 20-30 years for PV system lifetime analysis.

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.prices import EscalatingPriceModel

            model = EscalatingPriceModel(
                base_price_eur_per_kwh=0.22,
                annual_escalation=0.025,
                use_stochastic_escalation=True
            )

            # Monte Carlo loop
            for mc_path in range(1000):
                # Create independent RNG for this path
                rng = np.random.default_rng(seed=42 + mc_path)

                # Reset model with new stochastic trajectory
                model.reset_for_run(rng=rng, n_years=25)

                # Now get_price() will use this path's trajectory
                price_year_10 = model.get_price(10, 0)  # Year 10, January
                # Each mc_path gets different stochastic price evolution
            ```

        Notes:
            - Called automatically by MonteCarloSimulator before each path
            - Precomputation significantly improves performance (12 × n_years calls)
            - Stochastic variation only applied if use_stochastic_escalation=True
            - For deterministic models, all paths produce identical trajectories
            - Lazy mode (n_years=None) useful for single deterministic runs
        """
        if n_years is None:
            self._yearly_factors = None
            return

        if rng is not None:
            self._rng = rng
        elif self._rng is None:
            self._rng = self._fallback_rng

        self._yearly_factors = self._build_yearly_factors(n_years)

    def _build_yearly_factors(self, n_years: int) -> np.ndarray:
        """
        Generate cumulative price escalation factors for all simulation years.

        Computes the compound escalation multipliers applied to the base price
        for each year of the simulation. Incorporates both deterministic annual
        escalation and optional stochastic year-to-year variations.

        Algorithm:
        1. Generate random escalation variations from truncated normal distribution
        2. Iteratively build cumulative factors: factor[y] = ∏(1 + esc + var[i]) for i=0..y-1
        3. Clip growth rates to ensure positive prices (minimum 1e-6)

        The stochastic variations are sampled from a normal distribution with:
        - Mean: 0 (centered on annual_escalation)
        - Std dev: Derived from escalation_variation_percentiles (±90% range)
        - Truncation: Hard limits at percentile bounds to prevent extreme values

        Mathematical Formula:
            factor[year] = ∏_{i=0}^{year-1} max(1e-6, 1 + annual_escalation + variation[i])

            where variation[i] ~ TruncatedNormal(0, σ, p05, p95)
            and σ = max(|p05|, |p95|) / 1.6448536... (inverse of 90th percentile z-score)

        Args:
            n_years: Number of years to generate factors for.
                Determines array length. Should match simulation horizon.
                Typical values: 20-30 years for residential PV systems.

        Returns:
            np.ndarray: Yearly escalation factors (length n_years).
                factor[0] = 1.0 (year 0, no escalation yet)
                factor[1] = 1 + esc + var[0] (year 1, first escalation applied)
                factor[2] = (1 + esc + var[0]) × (1 + esc + var[1]) (year 2, compounded)
                ...
                Each factor is the cumulative multiplier from year 0 to that year.

                To get actual price: price[year] = base_price × factor[year] × seasonal

        Raises:
            ValueError: If escalation_variation_percentiles not (negative, positive).
                This check ensures stochastic bounds are sensible (symmetric around 0).

        Example:
            ```python
            # Deterministic escalation (2% per year)
            model = EscalatingPriceModel(
                base_price_eur_per_kwh=0.20,
                annual_escalation=0.02,
                use_stochastic_escalation=False
            )
            model._rng = np.random.default_rng(42)
            factors = model._build_yearly_factors(5)
            # factors ≈ [1.0, 1.02, 1.0404, 1.0612, 1.0824]
            # (each year compounds: 1.0, 1.02^1, 1.02^2, 1.02^3, 1.02^4)

            # Stochastic escalation (2% ± 5% variation)
            model_stoch = EscalatingPriceModel(
                base_price_eur_per_kwh=0.20,
                annual_escalation=0.02,
                use_stochastic_escalation=True,
                escalation_variation_percentiles=(-0.05, 0.05)
            )
            model_stoch._rng = np.random.default_rng(42)
            factors_stoch = model_stoch._build_yearly_factors(5)
            # factors_stoch ≈ [1.0, 1.017, 1.055, 1.071, 1.105]
            # (each year varies: could be -3% to +7% escalation)
            ```

        Notes:
            - Called by reset_for_run() to precompute trajectory
            - Variations sampled independently each year (no autocorrelation)
            - Minimum growth of 1e-6 prevents negative or zero prices
            - Truncated normal ensures realistic bounds (no 50% swings)
            - Constant 1.6448536... is inverse CDF of normal at 95th percentile
            - Deterministic mode (use_stochastic_escalation=False) produces
              perfect compound escalation: factor[y] = (1 + annual_escalation)^y
        """
        factors = np.ones(n_years, dtype=float)
        rng = self._rng or self._fallback_rng

        p05, p95 = self.escalation_variation_percentiles
        if p05 >= 0 or p95 <= 0:
            raise ValueError("escalation_variation_percentiles must be (neg, pos)")

        sigma = max(abs(p05), abs(p95)) / 1.6448536269514722
        variations = np.zeros(n_years, dtype=float)
        if self.use_stochastic_escalation:
            variations = rng.normal(loc=0.0, scale=sigma, size=n_years)
            variations = np.clip(variations, p05, p95)

        cumulative = 1.0
        for year in range(n_years):
            factors[year] = cumulative
            growth = max(1e-6, 1.0 + self.annual_escalation + variations[year])
            cumulative *= growth
        return factors

    def get_price(self, year_index: int, month_in_year: int) -> float:
        """
        Calculate electricity price for specific year and month.

        Returns the final electricity price (EUR/kWh) incorporating base price,
        cumulative annual escalation up to the specified year, and seasonal
        adjustment for the given month.

        This is the main query method called repeatedly during simulation to
        determine energy costs for each month. The method is optimized for
        performance through precomputed yearly factors (if reset_for_run was
        called with n_years).

        Price Formula:
            price = base_price × yearly_factor(year) × (1 + seasonal_factor[month])

        Where:
            yearly_factor(year) = precomputed cumulative escalation from _yearly_factors
                                  OR (1 + annual_escalation)^year if not precomputed
            seasonal_factor[month] = additive adjustment (e.g., +0.05 for +5% in Jan)

        Args:
            year_index: Simulation year (0-based integer).
                0 = first year, 1 = second year, etc.
                Should be < n_years if model was initialized with reset_for_run(n_years).
                Values beyond precomputed range fall back to deterministic escalation.
                Typical range: 0-19 for 20-year analysis, 0-29 for 30-year analysis.
            month_in_year: Month within the year (0-based integer, 0-11).
                0 = January, 1 = February, ..., 11 = December.
                Used to index into seasonal_factors array for monthly adjustment.
                Values outside 0-11 will raise IndexError.

        Returns:
            float: Electricity price in EUR per kWh for specified time point.
                Always positive. Typical residential range: 0.15-0.40 EUR/kWh.
                Includes all three components: base, escalation, seasonality.

        Example:
            ```python
            import numpy as np
            from sim_stochastic_pv.simulation.prices import EscalatingPriceModel

            # Setup model with 2% annual escalation, winter peak seasonality
            model = EscalatingPriceModel(
                base_price_eur_per_kwh=0.22,
                annual_escalation=0.02,
                use_stochastic_escalation=False
            )

            # Prepare for 20-year simulation (precompute trajectory)
            model.reset_for_run(rng=np.random.default_rng(42), n_years=20)

            # Get price for first year, January (0-indexed)
            # Base: 0.22, Year factor: 1.0, Seasonal: +5%
            # → 0.22 × 1.0 × 1.05 = 0.231 EUR/kWh
            price_y0_jan = model.get_price(year_index=0, month_in_year=0)

            # Get price for 10th year, July
            # Base: 0.22, Year factor: 1.02^10 ≈ 1.219, Seasonal: -3%
            # → 0.22 × 1.219 × 0.97 ≈ 0.260 EUR/kWh
            price_y10_jul = model.get_price(year_index=10, month_in_year=6)

            # Stochastic variation (different each Monte Carlo path)
            model_stoch = EscalatingPriceModel(
                base_price_eur_per_kwh=0.22,
                annual_escalation=0.025,
                use_stochastic_escalation=True
            )

            # Path 1
            model_stoch.reset_for_run(rng=np.random.default_rng(1), n_years=20)
            price_path1 = model_stoch.get_price(10, 0)  # e.g., 0.275 EUR/kWh

            # Path 2 (different random seed → different trajectory)
            model_stoch.reset_for_run(rng=np.random.default_rng(2), n_years=20)
            price_path2 = model_stoch.get_price(10, 0)  # e.g., 0.291 EUR/kWh
            ```

        Notes:
            - Called very frequently: n_mc × n_years × 12 times per full MC simulation
            - Optimized through precomputation: O(1) lookup vs O(1) calculation
            - Deterministic fallback if year_index exceeds precomputed range
            - Seasonal factors are ADDITIVE: (1 + factor), not multiplicative
            - Stochastic variation comes from reset_for_run(), not this method
            - Within a single Monte Carlo path, this method is deterministic
            - Thread-safe for read-only access (after reset_for_run() completes)
        """
        if self._yearly_factors is not None and year_index < self._yearly_factors.size:
            factor_year = self._yearly_factors[year_index]
        else:
            factor_year = (1.0 + self.annual_escalation) ** year_index
        factor_season = 1.0 + self.seasonal_factors[month_in_year]
        return self.base_price * factor_year * factor_season


# ---------------------------------------------------------------------------
# Stochastic price models (Phase 2 of the roadmap).
#
# The classes below replace the legacy iid jitter logic of EscalatingPriceModel
# with proper stochastic differential equation discretisations. They are
# parametrised in "human" units (annual drift, annual volatility, prices in
# EUR/kWh) but operate internally on the log-price for numerical robustness.
# ---------------------------------------------------------------------------


# Continuous-compounding minimum monthly price. Acts as a numerical guard
# against pathological GBM/OU paths that would underflow EUR/kWh.
_MIN_PRICE_EUR_PER_KWH = 1e-4


class GBMPriceModel(PriceModel):
    """
    Geometric Brownian Motion electricity price model (random walk in log-price).

    Implements the canonical GBM stochastic differential equation, discretised
    at the monthly time step Δt = 1/12 year:

        log P_{t+Δt} = log P_t + (μ − σ²/2) Δt + σ √Δt · ε_t,   ε_t ~ N(0,1)

    where:
        - μ = ``drift_annual`` is the continuous-time drift in log-price
          (≈ expected annual percentage growth for small values)
        - σ = ``volatility_annual`` is the annualised log-volatility
        - the Itō correction ``-σ²/2`` keeps ``E[P_t] = P_0 · exp(μ t)``
          (otherwise the median would lag behind the mean by σ²/2 per year).

    Unlike :class:`EscalatingPriceModel` — which uses iid clipped perturbations
    on the *escalation rate* and reverts to a deterministic trend — GBM lets
    shocks *persist*. The variance of ``log P_t`` grows linearly with time
    (``Var[log P_t] = σ² t``), which is exactly what is needed to expose the
    long-horizon risk hidden in a 20-year residential PV investment.

    Optionally, an additive monthly seasonal cycle can be overlaid as a
    multiplicative factor (defaults to 1.0 for every month → no seasonality).

    Attributes:
        base_price_eur_per_kwh: Initial price (EUR/kWh) at year 0, month 0,
            before the seasonal multiplier is applied.
        drift_annual: Continuous-time drift of log-price (1/year). For small
            values it is approximately the expected annual percentage growth.
            Typical residential EU range: 0.015-0.040 (1.5%–4%/year).
        volatility_annual: Annualised log-volatility (1/√year). Controls how
            wide the fan of simulated trajectories opens up over time.
            Typical residential EU range: 0.05-0.15 (pre-2021 historical was
            ~0.08; the energy crisis pushed it to 0.20+). Larger σ → wider
            uncertainty bands, more risk in the investment.
        seasonal_factors: 12-element array of multiplicative monthly factors
            (Jan-Dec). Each factor is applied as ``price × factor[month]``.
            Defaults to ``[1.0]*12`` (no seasonality). Use values centred on
            1.0 (e.g. winter 1.05, summer 0.95) so the annual average price
            is not biased upwards.

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.prices import GBMPriceModel

        # Pre-crisis EU residential calibration
        gbm = GBMPriceModel(
            base_price_eur_per_kwh=0.25,
            drift_annual=0.025,
            volatility_annual=0.08,
        )

        # Reset and read one full Monte Carlo path
        rng = np.random.default_rng(seed=42)
        gbm.reset_for_run(rng=rng, n_years=20)

        for year in range(20):
            price_jan = gbm.get_price(year, 0)
            print(f"Year {year:>2d} Jan: {price_jan:.4f} EUR/kWh")

        # Different paths via different seeds
        gbm.reset_for_run(rng=np.random.default_rng(7), n_years=20)
        path_b = [gbm.get_price(y, 0) for y in range(20)]
        ```

    Notes:
        - ``reset_for_run(rng, n_years)`` precomputes the entire ``(n_years*12)``
          monthly trajectory and ``get_price`` becomes an O(1) lookup.
        - The model is monthly by construction: yearly aggregates can be
          obtained downstream by averaging the 12 monthly values.
        - Prices are floored at ``_MIN_PRICE_EUR_PER_KWH`` (1e-4 EUR/kWh) to
          avoid log-of-zero issues in pathological negative-drift scenarios.
        - This model has **no mean reversion**: trajectories can drift away
          from the initial price indefinitely. If that feels unrealistic for
          your horizon, use :class:`MeanRevertingPriceModel` instead.
        - ``seasonal_factors`` are multiplicative (consistent with the
          log-space dynamics), whereas EscalatingPriceModel uses additive
          ``(1 + factor)``. They are NOT interchangeable across models.
    """

    def __init__(
        self,
        base_price_eur_per_kwh: float = 0.25,
        drift_annual: float = 0.025,
        volatility_annual: float = 0.08,
        seasonal_factors: List[float] | None = None,
    ) -> None:
        """
        Initialise a GBM price model.

        Args:
            base_price_eur_per_kwh: Initial price at year 0, month 0 (EUR/kWh).
                Must be positive. Typical residential EU: 0.18–0.35.
            drift_annual: Continuous-time annual drift μ (decimal).
                Default 0.025 (≈ +2.5%/year).
            volatility_annual: Annualised log-volatility σ ≥ 0.
                Default 0.08 (pre-2021 EU residential calibration).
            seasonal_factors: Optional 12-element list of multiplicative
                monthly factors (Jan..Dec). Defaults to ``[1.0]*12``.

        Raises:
            ValueError: If ``base_price_eur_per_kwh <= 0``,
                ``volatility_annual < 0``, or ``seasonal_factors`` is the
                wrong length.

        Example:
            ```python
            from sim_stochastic_pv.simulation.prices import GBMPriceModel

            # Default (pre-crisis EU calibration)
            gbm = GBMPriceModel()

            # High-uncertainty scenario for stress testing
            stress = GBMPriceModel(
                base_price_eur_per_kwh=0.30,
                drift_annual=0.035,
                volatility_annual=0.20,
            )

            # With mild winter premium
            seasonal = GBMPriceModel(
                base_price_eur_per_kwh=0.25,
                seasonal_factors=[1.05, 1.04, 1.02, 1.0, 0.98, 0.97,
                                  0.97, 0.98, 1.0, 1.02, 1.04, 1.05],
            )
            ```

        Notes:
            - ``volatility_annual = 0`` makes the model purely deterministic:
              ``P_t = P_0 · exp(μ t)`` exactly.
            - Initialisation does NOT generate a price path; call
              ``reset_for_run(rng, n_years)`` for that.
        """
        if base_price_eur_per_kwh <= 0:
            raise ValueError(
                f"base_price_eur_per_kwh must be positive, got {base_price_eur_per_kwh}"
            )
        if volatility_annual < 0:
            raise ValueError(
                f"volatility_annual must be non-negative, got {volatility_annual}"
            )

        self.base_price = float(base_price_eur_per_kwh)
        self.drift_annual = float(drift_annual)
        self.volatility_annual = float(volatility_annual)

        if seasonal_factors is None:
            self.seasonal_factors = np.ones(12, dtype=float)
        else:
            if len(seasonal_factors) != 12:
                raise ValueError(
                    f"seasonal_factors must have length 12, got {len(seasonal_factors)}"
                )
            arr = np.array(seasonal_factors, dtype=float)
            if np.any(arr <= 0):
                raise ValueError("seasonal_factors must all be strictly positive")
            self.seasonal_factors = arr

        # Time step in years (monthly granularity).
        self._dt_years: float = 1.0 / 12.0

        # Precomputed monthly base path (before seasonal multiplier).
        # Shape (n_years * 12,) once reset_for_run has been called.
        self._monthly_path_eur_per_kwh: np.ndarray | None = None
        self._rng: np.random.Generator | None = None
        self._fallback_rng = np.random.default_rng()

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Generate a full Monte Carlo price path for the upcoming simulation.

        Implements the GBM discretisation iteratively from the base price
        and stores the resulting ``(n_years * 12,)`` array. Subsequent
        calls to :meth:`get_price` are O(1) lookups into this array.

        Args:
            rng: Random number generator. If ``None`` the model's internal
                fallback RNG is used (NOT recommended for Monte Carlo).
            n_years: Simulation horizon in years. If ``None`` the path is
                cleared (lazy mode) and ``get_price`` falls back to the
                deterministic median trajectory.

        Example:
            ```python
            import numpy as np
            gbm = GBMPriceModel(volatility_annual=0.10)
            for path_id in range(500):
                rng = np.random.default_rng(42 + path_id)
                gbm.reset_for_run(rng=rng, n_years=20)
                # Each path has independent shocks
                final_price = gbm.get_price(19, 11)
            ```

        Notes:
            - Reproducibility: identical ``rng`` seeds produce identical paths.
            - The Itō correction ``-σ²/2 · Δt`` is included so the mean
              path matches the deterministic ``P_0 · exp(μ t)`` exactly.
            - Underflow guard: any monthly price below
              ``_MIN_PRICE_EUR_PER_KWH`` is clamped to that value.
        """
        if rng is not None:
            self._rng = rng
        elif self._rng is None:
            self._rng = self._fallback_rng

        if n_years is None:
            self._monthly_path_eur_per_kwh = None
            return

        self._monthly_path_eur_per_kwh = self._simulate_path(int(n_years))

    def _simulate_path(self, n_years: int) -> np.ndarray:
        """
        Simulate a full GBM trajectory of length ``n_years * 12`` months.

        Args:
            n_years: Number of years to simulate.

        Returns:
            np.ndarray of shape ``(n_years * 12,)``: monthly base prices
            (EUR/kWh) before the seasonal multiplier is applied.

        Notes:
            - Internal helper. Use ``reset_for_run`` from outside.
            - Uses vectorised cumulative sum of log-increments for speed.
        """
        n_steps = n_years * 12
        if n_steps <= 0:
            return np.array([self.base_price], dtype=float)

        rng = self._rng or self._fallback_rng
        dt = self._dt_years
        mu = self.drift_annual
        sigma = self.volatility_annual

        # Per-step log-increment: (μ - σ²/2) Δt + σ √Δt ε
        drift_term = (mu - 0.5 * sigma * sigma) * dt
        diffusion_term = sigma * np.sqrt(dt) * rng.standard_normal(n_steps)
        log_increments = drift_term + diffusion_term

        # Cumulative log-price, then exponentiate
        log_prices = np.log(self.base_price) + np.cumsum(log_increments)
        prices = np.exp(log_prices)

        # Numerical floor against pathological tails
        return np.maximum(prices, _MIN_PRICE_EUR_PER_KWH)

    def get_price(self, year_index: int, month_in_year: int) -> float:
        """
        Return the price for the requested ``(year, month)`` from the
        precomputed Monte Carlo trajectory.

        If ``reset_for_run`` was never called with an ``n_years`` argument,
        falls back to the deterministic median path
        ``P_0 · exp(μ · t)`` (i.e. volatility is ignored on the lazy path).

        Args:
            year_index: Simulation year (0-based).
            month_in_year: Month within the year (0=Jan ... 11=Dec).

        Returns:
            float: Electricity price in EUR/kWh, including the monthly
            seasonal multiplier.

        Notes:
            - O(1) lookup once the path has been precomputed.
            - The lazy fallback (no precomputed path) is intentionally
              deterministic so quick interactive prototyping does not
              require an RNG.
        """
        if self._monthly_path_eur_per_kwh is not None:
            month_index = year_index * 12 + month_in_year
            n = self._monthly_path_eur_per_kwh.size
            if 0 <= month_index < n:
                base = float(self._monthly_path_eur_per_kwh[month_index])
            else:
                # Out-of-range: extrapolate using the last point.
                base = float(self._monthly_path_eur_per_kwh[-1])
        else:
            # Deterministic median fallback: P_0 · exp(μ t)
            t_years = year_index + month_in_year * self._dt_years
            base = self.base_price * float(np.exp(self.drift_annual * t_years))

        return base * float(self.seasonal_factors[month_in_year])


class MeanRevertingPriceModel(PriceModel):
    """
    Ornstein-Uhlenbeck price model with mean reversion in log-price space.

    Discretised at the monthly time step Δt = 1/12 year:

        x_{t+Δt} = x_t + κ (θ − x_t) Δt + σ √Δt · ε_t,   x_t = log P_t

    where:
        - κ = ``mean_reversion_speed_annual`` controls how fast trajectories
          are pulled back towards ``θ`` (1/year, typical energy markets: 0.1–0.6).
        - θ = ``log(long_term_price_eur_per_kwh)`` is the equilibrium
          log-price the process oscillates around.
        - σ = ``volatility_annual`` is the annualised log-volatility.

    Unlike :class:`GBMPriceModel` (whose variance grows linearly with time
    and trajectories can drift arbitrarily far), the OU process has a
    **bounded stationary variance** ``σ²/(2κ)``: the price wanders around
    ``long_term_price_eur_per_kwh`` rather than diverging. This is the more
    realistic dynamics for commodity prices over multi-decade horizons.

    The trade-off is one more parameter (κ) to calibrate and a model that
    inherently *under*-represents the kind of regime shifts that GBM captures
    by accident. Pick GBM if you want exposure to scenarios where the price
    permanently triples; pick OU if you believe regulation or substitution
    pulls the price back towards some long-term equilibrium.

    Attributes:
        base_price_eur_per_kwh: Initial price (EUR/kWh) at year 0, month 0.
        long_term_price_eur_per_kwh: Equilibrium price the process reverts
            towards. Setting it equal to ``base_price`` makes the process
            symmetric (no expected drift); setting it higher introduces a
            gradual long-term increase.
        mean_reversion_speed_annual: κ (1/year). Larger values pull
            trajectories back faster. Half-life of a shock = ln(2)/κ years.
        volatility_annual: σ (1/√year). Determines stationary dispersion.
        seasonal_factors: 12-element multiplicative monthly factors.

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.simulation.prices import MeanRevertingPriceModel

        ou = MeanRevertingPriceModel(
            base_price_eur_per_kwh=0.25,
            long_term_price_eur_per_kwh=0.28,  # gentle long-term increase
            mean_reversion_speed_annual=0.35,  # ~2 years half-life
            volatility_annual=0.12,
        )
        ou.reset_for_run(rng=np.random.default_rng(0), n_years=20)
        path = [ou.get_price(y, 0) for y in range(20)]
        ```

    Notes:
        - Stationary mean of ``log P`` is exactly ``log(long_term_price)``.
        - Stationary std-dev of ``log P`` is ``σ / √(2κ)``; for σ=0.12,
          κ=0.35 it is ~0.143, so ~95% of stationary prices lie within
          ``long_term × exp(±0.29)``.
        - Setting ``κ = 0`` would degenerate to a pure random walk with no
          drift; we raise instead, suggesting :class:`GBMPriceModel`.
    """

    def __init__(
        self,
        base_price_eur_per_kwh: float = 0.25,
        long_term_price_eur_per_kwh: float = 0.25,
        mean_reversion_speed_annual: float = 0.30,
        volatility_annual: float = 0.12,
        seasonal_factors: List[float] | None = None,
    ) -> None:
        """
        Initialise a mean-reverting (OU) price model.

        Args:
            base_price_eur_per_kwh: Starting price (EUR/kWh), > 0.
            long_term_price_eur_per_kwh: Equilibrium price, > 0.
            mean_reversion_speed_annual: κ > 0 (1/year).
            volatility_annual: σ ≥ 0 (1/√year).
            seasonal_factors: Optional 12-element multiplicative monthly factors.

        Raises:
            ValueError: If any of the positivity constraints is violated, or
                if ``seasonal_factors`` has the wrong length.

        Notes:
            - ``mean_reversion_speed_annual = 0`` is rejected: it would
              collapse the model to a pure random walk without drift, which
              is more honestly represented by ``GBMPriceModel(drift=0)``.
        """
        if base_price_eur_per_kwh <= 0:
            raise ValueError("base_price_eur_per_kwh must be positive")
        if long_term_price_eur_per_kwh <= 0:
            raise ValueError("long_term_price_eur_per_kwh must be positive")
        if mean_reversion_speed_annual <= 0:
            raise ValueError(
                "mean_reversion_speed_annual must be strictly positive; "
                "use GBMPriceModel for κ=0 (no reversion)"
            )
        if volatility_annual < 0:
            raise ValueError("volatility_annual must be non-negative")

        self.base_price = float(base_price_eur_per_kwh)
        self.long_term_price = float(long_term_price_eur_per_kwh)
        self.mean_reversion_speed_annual = float(mean_reversion_speed_annual)
        self.volatility_annual = float(volatility_annual)

        if seasonal_factors is None:
            self.seasonal_factors = np.ones(12, dtype=float)
        else:
            if len(seasonal_factors) != 12:
                raise ValueError(
                    f"seasonal_factors must have length 12, got {len(seasonal_factors)}"
                )
            arr = np.array(seasonal_factors, dtype=float)
            if np.any(arr <= 0):
                raise ValueError("seasonal_factors must all be strictly positive")
            self.seasonal_factors = arr

        self._dt_years: float = 1.0 / 12.0
        self._monthly_path_eur_per_kwh: np.ndarray | None = None
        self._rng: np.random.Generator | None = None
        self._fallback_rng = np.random.default_rng()

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Generate a Monte Carlo OU trajectory for the upcoming run.

        Args:
            rng: Random number generator.
            n_years: Simulation horizon in years; if ``None`` the path is
                cleared and ``get_price`` falls back to the deterministic
                exponential decay towards ``long_term_price``.
        """
        if rng is not None:
            self._rng = rng
        elif self._rng is None:
            self._rng = self._fallback_rng

        if n_years is None:
            self._monthly_path_eur_per_kwh = None
            return

        self._monthly_path_eur_per_kwh = self._simulate_path(int(n_years))

    def _simulate_path(self, n_years: int) -> np.ndarray:
        """
        Simulate the OU log-price recursively (cannot vectorise the
        mean-reversion term efficiently because each step depends on the
        previous log-price).

        Returns:
            np.ndarray of shape ``(n_years * 12,)``: monthly base prices.
        """
        n_steps = n_years * 12
        if n_steps <= 0:
            return np.array([self.base_price], dtype=float)

        rng = self._rng or self._fallback_rng
        dt = self._dt_years
        kappa = self.mean_reversion_speed_annual
        sigma = self.volatility_annual
        theta = np.log(self.long_term_price)
        sqrt_dt = np.sqrt(dt)

        log_prices = np.empty(n_steps, dtype=float)
        x = np.log(self.base_price)
        epsilons = rng.standard_normal(n_steps)
        for i in range(n_steps):
            x = x + kappa * (theta - x) * dt + sigma * sqrt_dt * epsilons[i]
            log_prices[i] = x

        prices = np.exp(log_prices)
        return np.maximum(prices, _MIN_PRICE_EUR_PER_KWH)

    def get_price(self, year_index: int, month_in_year: int) -> float:
        """
        Return the price for the requested ``(year, month)`` from the
        precomputed OU trajectory.

        Falls back to the deterministic mean of the OU process when no
        path has been generated yet:
        ``E[log P_t] = θ + (log P_0 − θ) · exp(−κ t)``.
        """
        if self._monthly_path_eur_per_kwh is not None:
            month_index = year_index * 12 + month_in_year
            n = self._monthly_path_eur_per_kwh.size
            if 0 <= month_index < n:
                base = float(self._monthly_path_eur_per_kwh[month_index])
            else:
                base = float(self._monthly_path_eur_per_kwh[-1])
        else:
            t_years = year_index + month_in_year * self._dt_years
            theta = np.log(self.long_term_price)
            x0 = np.log(self.base_price)
            log_mean = theta + (x0 - theta) * float(
                np.exp(-self.mean_reversion_speed_annual * t_years)
            )
            base = float(np.exp(log_mean))

        return base * float(self.seasonal_factors[month_in_year])


# ---------------------------------------------------------------------------
# Phase 10 — Preview helper for the Database UI
# ---------------------------------------------------------------------------


# Maximum number of full trajectories returned by the preview helper. The
# fan-chart visualisation loses readability beyond ~30 lines anyway, and
# 20 keeps the JSON payload tight even for long horizons.
_PRICE_PREVIEW_SAMPLE_PATHS_LIMIT: int = 20


@dataclass
class PricePreviewResult:
    """
    Lightweight bundle of statistics returned by :func:`simulate_price_preview`.

    Designed to be JSON-friendly: a route handler can serialise the dataclass
    directly via Pydantic / dataclasses.asdict without any further reshaping.

    Attributes:
        months: 0-based month indices (length ``n_years * 12``).
        mean_eur_per_kwh: Cross-path mean price per month.
        p05_eur_per_kwh: 5th-percentile band lower edge.
        p95_eur_per_kwh: 95th-percentile band upper edge.
        sample_paths: Up to ``_PRICE_PREVIEW_SAMPLE_PATHS_LIMIT`` full
            trajectories, each one ``n_years * 12`` long.

    For deterministic configurations (e.g. GBM with ``volatility_annual=0``
    or legacy ``EscalatingPriceModel`` with stochastic flag off) the three
    bands coincide and every sample path is identical — that visual collapse
    is the correct sanity-check signal of "no uncertainty modelled".
    """

    months: List[int]
    mean_eur_per_kwh: List[float]
    p05_eur_per_kwh: List[float]
    p95_eur_per_kwh: List[float]
    sample_paths: List[List[float]]


def simulate_price_preview(
    price_model: PriceModel,
    n_years: int = 20,
    n_paths: int = 200,
    seed: int = 42,
    sample_paths_limit: int = _PRICE_PREVIEW_SAMPLE_PATHS_LIMIT,
) -> PricePreviewResult:
    """
    Run a stand-alone Monte Carlo of the price model — no energy simulation
    involved — and return the same statistics the Dashboard uses for its
    fan chart.

    This is the building block of the Phase 10 "preview" feature in the
    Database section of the UI: the user picks a set of price-model
    parameters and immediately sees what the simulated trajectories look
    like, without paying the cost of a full PV+battery simulation.

    Args:
        price_model: Any :class:`PriceModel` subclass already configured
            (e.g. ``GBMPriceModel(drift_annual=0.03, volatility_annual=0.10)``).
            The model's internal state is mutated by repeated
            ``reset_for_run`` calls.
        n_years: Simulation horizon in years. Determines the number of
            monthly time steps (``n_years * 12``). Default 20.
        n_paths: Number of Monte Carlo paths to draw. Higher → tighter
            empirical percentiles but slower. Default 200.
        seed: Master seed driving the per-path sub-seeds for
            reproducibility. Default 42.
        sample_paths_limit: Cap on the number of full trajectories included
            in the response (the rest contribute only to the statistics).
            Default :data:`_PRICE_PREVIEW_SAMPLE_PATHS_LIMIT`.

    Returns:
        PricePreviewResult: Bands + sample trajectories, ready to be
        serialised to JSON.

    Example:
        ```python
        from sim_stochastic_pv.simulation.prices import (
            GBMPriceModel,
            simulate_price_preview,
        )

        model = GBMPriceModel(
            base_price_eur_per_kwh=0.25,
            drift_annual=0.025,
            volatility_annual=0.12,
        )
        preview = simulate_price_preview(model, n_years=15, n_paths=300)
        assert len(preview.months) == 15 * 12
        assert preview.p05_eur_per_kwh[-1] < preview.p95_eur_per_kwh[-1]
        ```

    Notes:
        - Deterministic for a given ``(price_model, seed, n_paths, n_years)``.
        - The function does NOT mutate the global RNG state.
        - The "sample stride" used to pick which trajectories to keep is
          ``max(1, n_paths // sample_paths_limit)`` — i.e. evenly spaced
          across the index space, not a random sub-sample.
    """
    if n_years <= 0:
        raise ValueError(f"n_years must be positive, got {n_years}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")

    n_months = n_years * 12
    rng_master = np.random.default_rng(seed)
    paths = np.empty((n_paths, n_months), dtype=float)

    for i in range(n_paths):
        rng = np.random.default_rng(rng_master.integers(0, 1_000_000_000))
        price_model.reset_for_run(rng=rng, n_years=n_years)
        for m in range(n_months):
            year = m // 12
            month_in_year = m % 12
            paths[i, m] = price_model.get_price(year, month_in_year)

    mean = paths.mean(axis=0)
    p05 = np.percentile(paths, 5, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    if n_paths <= sample_paths_limit:
        idx = np.arange(n_paths)
    else:
        stride = max(1, n_paths // sample_paths_limit)
        idx = np.arange(0, n_paths, stride)[:sample_paths_limit]

    return PricePreviewResult(
        months=list(range(n_months)),
        mean_eur_per_kwh=[float(x) for x in mean],
        p05_eur_per_kwh=[float(x) for x in p05],
        p95_eur_per_kwh=[float(x) for x in p95],
        sample_paths=[[float(x) for x in paths[i, :]] for i in idx],
    )
