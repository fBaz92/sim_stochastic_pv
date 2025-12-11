from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class PriceModel(ABC):
    """Interface for energy price model: €/kWh as function of year/month."""

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
    ) -> None:
        """
        Initialize an escalating price model with annual and seasonal factors.
        
        Args:
            base_price_eur_per_kwh: Base energy price in EUR per kWh.
            annual_escalation: Annual price escalation rate (e.g., 0.02 = 2% per year).
            seasonal_factors: Monthly seasonal factors (12 values, None for default).
                Each factor is applied as: price *= (1 + seasonal_factor[month]).
        """
        self.base_price = base_price_eur_per_kwh
        self.annual_escalation = annual_escalation

        if seasonal_factors is None:
            self.seasonal_factors = np.array(
                [0.05, 0.04, 0.02, 0.0, -0.02, -0.03,
                 -0.03, -0.02, 0.0, 0.02, 0.04, 0.05]
            )
        else:
            if len(seasonal_factors) != 12:
                raise ValueError("seasonal_factors must have length 12")
            self.seasonal_factors = np.array(seasonal_factors)

    def get_price(self, year_index: int, month_in_year: int) -> float:
        """
        Calculate energy price with annual escalation and seasonal factors.
        
        Args:
            year_index: Year index (0-based).
            month_in_year: Month index (0-11).
        
        Returns:
            Energy price in EUR per kWh.
        """
        factor_year = (1.0 + self.annual_escalation) ** year_index
        factor_season = 1.0 + self.seasonal_factors[month_in_year]
        return self.base_price * factor_year * factor_season
