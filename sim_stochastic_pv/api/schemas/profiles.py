"""
Load and price profile schemas for API validation.

This module contains Pydantic models for electricity profiles:
- Load Profiles: Electricity consumption patterns for homes
- Price Profiles: Electricity pricing models and escalation

Profiles can be saved and reused across multiple scenarios, enabling
consistent modeling and easy comparison of different configurations.
"""

from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field


class LoadProfileResponse(BaseModel):
    """
    Electricity load profile response schema.

    Represents a stored electricity consumption profile pattern.
    Used when returning load profile data from API GET endpoints.

    Load profiles define how electricity consumption varies over time,
    including daily patterns, seasonal variations, and occupancy effects.

    Attributes:
        id: Unique database identifier.
        name: Human-readable profile name (must be unique).
        profile_type: Type of load profile.
            - "arera": Italian ARERA standard residential profile
            - "custom": Custom monthly average profile (12 values)
            - "custom_24h": Custom hourly profile (12 months × 24 hours)
        data: Profile configuration data as JSON.
            Structure depends on profile_type:
            - arera: Empty dict (uses built-in profile)
            - custom: {"home_profiles_w": [w1, w2, ..., w12]}
            - custom_24h: {"home_profiles_w": [[h1, h2, ..., h24], ...] × 12}

    Example:
        ```python
        # ARERA profile (Italian standard)
        {
            "id": 1,
            "name": "Standard Italian Home",
            "profile_type": "arera",
            "data": {}
        }

        # Custom monthly profile
        {
            "id": 2,
            "name": "Low Consumption Home",
            "profile_type": "custom",
            "data": {
                "home_profiles_w": [200, 180, 190, 210, 230, 250,
                                   270, 260, 240, 220, 200, 190]
            }
        }

        # Custom hourly profile
        {
            "id": 3,
            "name": "Night Worker Profile",
            "profile_type": "custom_24h",
            "data": {
                "home_profiles_w": [
                    [100, 90, 80, ..., 120],  # January (24 hours)
                    [110, 95, 85, ..., 130],  # February (24 hours)
                    ...  # 12 months total
                ],
                "min_days_home": [25, 25, 25, ..., 25]  # Days at home per month
            }
        }
        ```

    Notes:
        - home_profiles_w values are in watts (W)
        - custom profiles can be monthly averages or full hourly patterns
        - Profiles include both "home" and "away" consumption patterns
    """

    id: int
    name: str = Field(..., description="Unique profile name")
    profile_type: str = Field(..., description="Profile type: 'arera', 'custom', or 'custom_24h'")
    data: Dict[str, Any] = Field(..., description="Profile configuration (structure varies by type)")

    class Config:
        """Pydantic configuration for ORM mode support."""
        orm_mode = True


class LoadProfileCreate(BaseModel):
    """
    Schema for creating or updating a load profile.

    Input schema for POST /api/profiles/load endpoint. Creates a new profile
    or updates an existing one (upsert by name).

    Attributes:
        name: Unique profile identifier (required).
        profile_type: Type of profile (required).
            Valid values: "arera", "custom", "custom_24h"
        data: Profile configuration matching the profile_type (required).

    Example:
        ```python
        # POST /api/profiles/load
        {
            "name": "Summer Peak Reducer",
            "profile_type": "custom",
            "data": {
                "home_profiles_w": [
                    220, 200, 210, 230, 250, 300,  # Jan-Jun
                    280, 270, 260, 240, 220, 210   # Jul-Dec
                ],
                "away_profiles_w": [
                    100, 100, 100, 100, 120, 150,
                    140, 130, 120, 110, 100, 100
                ],
                "min_days_home": [26, 24, 26, 25, 26, 24,
                                 28, 28, 26, 26, 24, 26],
                "home_variation_percentiles": [-0.1, 0.1],
                "away_variation_percentiles": [-0.05, 0.05]
            }
        }
        ```

    Raises:
        ValidationError: If profile_type is invalid or data structure doesn't match type.
    """

    name: str = Field(..., min_length=1, description="Unique profile name")
    profile_type: str = Field(..., description="Profile type: 'arera', 'custom', or 'custom_24h'")
    data: Dict[str, Any] = Field(..., description="Profile configuration data")


class PriceProfileResponse(BaseModel):
    """
    Electricity price profile response schema.

    Represents a stored electricity pricing model including base price
    and escalation parameters. Used when returning price profile data.

    Price profiles model electricity costs over the system lifetime,
    including inflation and market dynamics.

    Attributes:
        id: Unique database identifier.
        name: Human-readable profile name (must be unique).
        data: Price model configuration as JSON.
            Expected fields:
            - base_price_eur_per_kwh: Initial electricity price (EUR/kWh)
            - annual_escalation: Average annual price increase rate (e.g., 0.02 = 2%/year)
            - use_stochastic_escalation: Whether to use stochastic escalation
            - escalation_variation_percentiles: Range of escalation uncertainty (e.g., [-0.05, 0.05])

    Example:
        ```python
        {
            "id": 1,
            "name": "Italian Residential 2025",
            "data": {
                "base_price_eur_per_kwh": 0.22,
                "annual_escalation": 0.025,
                "use_stochastic_escalation": true,
                "escalation_variation_percentiles": [-0.05, 0.05]
            }
        }
        ```

    Notes:
        - base_price_eur_per_kwh should match current market rates
        - annual_escalation typically ranges from 0% to 5% (0.00 to 0.05)
        - Stochastic escalation adds Monte Carlo uncertainty to price evolution
    """

    id: int
    name: str = Field(..., description="Unique profile name")
    data: Dict[str, Any] = Field(..., description="Price model configuration")

    class Config:
        """Pydantic configuration for ORM mode support."""
        orm_mode = True


class PriceProfileCreate(BaseModel):
    """
    Schema for creating or updating a price profile.

    Input schema for POST /api/profiles/price endpoint. Creates a new profile
    or updates an existing one (upsert by name).

    Attributes:
        name: Unique profile identifier (required).
        data: Price model configuration (required).
            Should include all price model parameters.

    Example:
        ```python
        # POST /api/profiles/price
        {
            "name": "Conservative Escalation Model",
            "data": {
                "base_price_eur_per_kwh": 0.20,
                "annual_escalation": 0.015,
                "use_stochastic_escalation": true,
                "escalation_variation_percentiles": [-0.03, 0.03]
            }
        }
        ```

    Raises:
        ValidationError: If name is empty or data is missing required fields.
    """

    name: str = Field(..., min_length=1, description="Unique profile name")
    data: Dict[str, Any] = Field(..., description="Price model configuration")
