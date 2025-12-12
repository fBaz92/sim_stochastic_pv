"""
Configuration and scenario storage schemas for API validation.

This module contains Pydantic models for storing and retrieving
simulation configurations:
- SavedConfiguration: Generic saved configuration (scenarios or campaigns)
- ScenarioResponse: Historical scenario execution records

These schemas enable the database-driven workflow where users can:
1. Save scenario configurations with hardware IDs
2. Execute saved scenarios directly from the database
3. Benefit from automatic hardware spec updates
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from pydantic import BaseModel, Field


class SavedConfigurationResponse(BaseModel):
    """
    Saved configuration response schema.

    Represents a stored scenario or campaign configuration that can be
    executed via the database-driven workflow endpoints.

    This enables the key architectural pattern where scenarios reference
    hardware by ID rather than embedding values, ensuring hardware updates
    propagate to all scenarios using that hardware.

    Attributes:
        id: Unique database identifier.
        name: Human-readable configuration name (must be unique).
        config_type: Type of configuration.
            - "scenario": Single-scenario configuration for analysis
            - "campaign": Multi-scenario configuration for optimization
        data: Configuration data as JSON.
            Structure depends on config_type:
            - scenario: Full scenario dict with optional hardware IDs
            - campaign: Optimization config with hardware_selections

    Example:
        ```python
        # Saved scenario with hardware IDs
        {
            "id": 1,
            "name": "Standard Home 3kW System",
            "config_type": "scenario",
            "data": {
                "inverter_id": 5,
                "battery_id": 3,
                "load_profile": {
                    "home_profile_type": "arera",
                    "away_profile": "arera",
                    "min_days_home": [25] * 12
                },
                "solar": {
                    "pv_kwp": 3.0,
                    "degradation_per_year": 0.007
                },
                "energy": {
                    "n_years": 20,
                    "pv_kwp": 3.0,
                    "n_batteries": 1
                },
                "price": {
                    "base_price_eur_per_kwh": 0.20,
                    "annual_escalation": 0.02
                },
                "economic": {
                    "n_mc": 500,
                    "investment_eur": 8000.0
                }
            }
        }

        # Saved campaign with hardware selections
        {
            "id": 2,
            "name": "Home Optimization Campaign",
            "config_type": "campaign",
            "data": {
                "hardware_selections": {
                    "inverter_ids": [1, 2, 3],
                    "panel_ids": [1],
                    "battery_ids": [1, 2]
                },
                "optimization": {
                    "panel_count_options": [6, 8, 10],
                    "battery_count_options": [0, 1, 2],
                    "include_no_battery": true
                },
                "load_profile": {...},
                "solar": {...},
                "energy": {...},
                "price": {...},
                "economic": {...}
            }
        }
        ```

    Notes:
        - Hardware IDs (inverter_id, battery_id, etc.) link to hardware catalog
        - When executed, IDs are hydrated to current hardware specs from database
        - This ensures scenarios always use up-to-date hardware specifications
    """

    id: int
    name: str = Field(..., description="Unique configuration name")
    config_type: str = Field(..., description="Configuration type: 'scenario' or 'campaign'")
    data: Dict[str, Any] = Field(..., description="Configuration data (structure varies by type)")

    class Config:
        """Pydantic configuration for ORM mode support."""
        orm_mode = True


class SavedConfigurationCreate(BaseModel):
    """
    Schema for creating or updating a saved configuration.

    Input schema for POST /api/configurations endpoint. Creates a new
    configuration or updates an existing one (upsert by name).

    This is the primary way to save scenarios and campaigns for later
    execution via the /scenarios/{id}/run or /campaigns/{id}/run endpoints.

    Attributes:
        name: Unique configuration identifier (required).
        config_type: Type of configuration (required).
            Must be "scenario" or "campaign".
        data: Complete configuration data (required).
            Should include all required sections for the config_type.

    Example:
        ```python
        # POST /api/configurations (scenario)
        {
            "name": "My Home System",
            "config_type": "scenario",
            "data": {
                "inverter_id": 5,
                "battery_id": 3,
                "load_profile": {...},
                "solar": {...},
                "energy": {...},
                "price": {...},
                "economic": {...}
            }
        }

        # POST /api/configurations (campaign)
        {
            "name": "System Optimization",
            "config_type": "campaign",
            "data": {
                "hardware_selections": {
                    "inverter_ids": [1, 2],
                    "panel_ids": [1],
                    "battery_ids": [1, 2]
                },
                "optimization": {...},
                "load_profile": {...},
                "solar": {...},
                "energy": {...},
                "price": {...},
                "economic": {...}
            }
        }
        ```

    Raises:
        ValidationError: If name is empty, config_type is invalid, or data is missing.
    """

    name: str = Field(..., min_length=1, description="Unique configuration name")
    config_type: str = Field(..., description="Configuration type: 'scenario' or 'campaign'")
    data: Dict[str, Any] = Field(..., description="Complete configuration data")


class ScenarioResponse(BaseModel):
    """
    Historical scenario record response schema.

    Represents a scenario configuration that was executed and stored
    in the database. Different from SavedConfiguration - this is the
    execution history table.

    Note: This schema is somewhat redundant with SavedConfigurationResponse
    and may be consolidated in future API versions. Currently maintained
    for backward compatibility.

    Attributes:
        id: Unique database identifier.
        name: Scenario name.
        config: Scenario configuration as JSON.
        created_at: Timestamp when scenario was created.

    Example:
        ```python
        {
            "id": 15,
            "name": "3kW PV + 5kWh Battery",
            "config": {
                "load_profile": {...},
                "solar": {...},
                "energy": {...},
                "price": {...},
                "economic": {...}
            },
            "created_at": "2025-01-15T10:30:45.123456Z"
        }
        ```

    Notes:
        - This represents executed scenarios from the scenarios table
        - SavedConfigurationResponse represents configurations table
        - Consider using SavedConfiguration for new code
    """

    id: int
    name: str
    config: Dict[str, Any]
    created_at: datetime

    class Config:
        """Pydantic configuration for ORM mode support."""
        orm_mode = True
