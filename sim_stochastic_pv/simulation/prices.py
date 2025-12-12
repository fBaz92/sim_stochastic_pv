"""
Electricity price escalation and stochastic modeling utilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
