from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class PriceModel(ABC):
    """Interface for energy price model: €/kWh as function of year/month."""

    def reset_for_run(
        self,
        rng: np.random.Generator | None = None,
        n_years: int | None = None,
    ) -> None:
        """
        Hook called before each Monte Carlo path (default no-op).
        """
        return

    @abstractmethod
    def get_price(self, year_index: int, month_in_year: int) -> float:
        """
        Get the energy price for a specific year and month.
        
        Args:
            year_index: Year index (0-based, 0 = first year).
            month_in_year: Month index (0-11, 0 = January).
        
        Returns:
            Energy price in EUR per kWh.
        """
        raise NotImplementedError


class EscalatingPriceModel(PriceModel):
    """
    price(year, month) = base_price * (1 + annual_escalation)^year * (1 + seasonal_factor[month])
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
        Initialize an escalating price model with annual and seasonal factors.
        
        Args:
            base_price_eur_per_kwh: Base energy price in EUR per kWh.
            annual_escalation: Annual price escalation rate (e.g., 0.02 = 2% per year).
            seasonal_factors: Monthly seasonal factors (12 values, None for default).
                Each factor is applied as: price *= (1 + seasonal_factor[month]).
            use_stochastic_escalation: Whether to sample annual escalation deltas.
            escalation_variation_percentiles: Tuple (p05, p95) for the stochastic
                variation added to the annual escalation (percent values, e.g. -0.05/+0.05).
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
        if n_years is None:
            self._yearly_factors = None
            return

        if rng is not None:
            self._rng = rng
        elif self._rng is None:
            self._rng = self._fallback_rng

        self._yearly_factors = self._build_yearly_factors(n_years)

    def _build_yearly_factors(self, n_years: int) -> np.ndarray:
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
        Calculate energy price with annual escalation and seasonal factors.
        
        Args:
            year_index: Year index (0-based).
            month_in_year: Month index (0-11).
        
        Returns:
            Energy price in EUR per kWh.
        """
        if self._yearly_factors is not None and year_index < self._yearly_factors.size:
            factor_year = self._yearly_factors[year_index]
        else:
            factor_year = (1.0 + self.annual_escalation) ** year_index
        factor_season = 1.0 + self.seasonal_factors[month_in_year]
        return self.base_price * factor_year * factor_season
