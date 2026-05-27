"""
Hardware option dataclasses for optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

@dataclass(frozen=True)
class InverterOption:
    """
    PV inverter hardware specification for scenario optimization.

    Encapsulates the technical and cost parameters of a specific inverter
    model. Used to generate hardware configurations during optimization.

    Attributes:
        name: Human-readable inverter name/model identifier.
        p_ac_max_kw: Maximum AC output power (kW). Determines grid injection limit.
        price_eur: Inverter hardware cost (EUR), excluding installation.
        p_dc_max_kw: Maximum DC input power (kW), or None for unlimited.
        install_cost_eur: Installation cost (EUR), or None for auto-estimate.
        integrated_battery_specs: Battery specs if inverter has integrated storage.
        integrated_battery_price_eur: Cost per integrated battery module (EUR).
        integrated_battery_count_options: Possible integrated battery counts.
        manufacturer: Manufacturer name (optional metadata).
        model_number: Manufacturer model number (optional metadata).
        datasheet: Additional technical data (optional).

    Example:
        ```python
        # Standard inverter without battery
        inverter_basic = InverterOption(
            name="Fronius Primo 5.0",
            p_ac_max_kw=5.0,
            price_eur=1500.0,
            p_dc_max_kw=5.5,
            manufacturer="Fronius"
        )

        # Hybrid inverter with integrated battery
        inverter_hybrid = InverterOption(
            name="Huawei SUN2000-5KTL-L1",
            p_ac_max_kw=5.0,
            price_eur=2000.0,
            integrated_battery_specs=BatterySpecs(capacity_kwh=5.0, cycles_life=6000),
            integrated_battery_price_eur=1200.0,
            integrated_battery_count_options=[1, 2, 3],
            manufacturer="Huawei"
        )
        ```
    """
    name: str
    p_ac_max_kw: float
    price_eur: float
    p_dc_max_kw: Optional[float] = None
    install_cost_eur: Optional[float] = None
    integrated_battery_specs: Optional["BatterySpecs"] = None
    integrated_battery_price_eur: Optional[float] = None
    integrated_battery_count_options: Optional[List[int]] = None
    manufacturer: Optional[str] = None
    model_number: Optional[str] = None
    datasheet: Optional[Any] = None

    @property
    def total_cost_eur(self) -> float:
        """
        Total inverter cost: hardware price plus explicit installation cost.

        Returns ``price_eur + install_cost_eur`` when ``install_cost_eur`` is
        set, otherwise returns ``price_eur`` alone (installation cost = 0).

        This is the value used by ``ScenarioDefinition.investment_eur`` for
        budget calculations.  The legacy ``installation_cost()`` method that
        returned heuristic estimates (1 000 / 2 000 EUR) is intentionally *not*
        called here to keep costs transparent and predictable.

        Returns:
            float: Total inverter cost in EUR (hardware + explicit install).

        Example:
            ```python
            inv_with_install = InverterOption(
                name="Fronius 5kW", p_ac_max_kw=5.0,
                price_eur=1500.0, install_cost_eur=300.0
            )
            assert inv_with_install.total_cost_eur == 1800.0

            inv_no_install = InverterOption(
                name="SMA 3kW", p_ac_max_kw=3.0, price_eur=1200.0
            )
            assert inv_no_install.total_cost_eur == 1200.0
            ```
        """
        return self.price_eur + (self.install_cost_eur or 0.0)

    def total_cost(self) -> float:
        """Calculate total inverter cost (hardware + installation). Alias for total_cost_eur."""
        return self.total_cost_eur

    def installation_cost(self) -> float:
        """
        Return the explicit installation cost, or 0 when not set.

        Note: In previous versions this method returned heuristic estimates
        (1 000 EUR for inverters ≤0.8 kW, 2 000 EUR for larger units) when
        ``install_cost_eur`` was None.  That behaviour was removed because it
        produced surprising results in budget calculations.  Explicit costs
        are now always preferred; when no installation cost is known, the
        value is 0.

        Returns:
            float: ``install_cost_eur`` when set, otherwise 0.0.
        """
        return self.install_cost_eur or 0.0


@dataclass(frozen=True)
class PanelOption:
    """
    PV panel hardware specification for scenario optimization.

    Encapsulates the technical and cost parameters of a specific solar
    panel model. Used to generate PV array configurations during optimization.

    Attributes:
        name: Human-readable panel name/model identifier.
        power_w: Nominal peak power per panel (Watts, STC conditions).
        price_eur: Cost per panel (EUR).
        manufacturer: Manufacturer name (optional metadata).
        model_number: Manufacturer model number (optional metadata).
        datasheet: Additional technical data (optional).

    Example:
        ```python
        panel_std = PanelOption(
            name="Longi LR5-72HPH-540M",
            power_w=540.0,
            price_eur=150.0,
            manufacturer="Longi Solar",
            model_number="LR5-72HPH-540M"
        )

        panel_budget = PanelOption(
            name="Trina TSM-DE09.08",
            power_w=400.0,
            price_eur=100.0,
            manufacturer="Trina Solar"
        )
        ```
    """
    name: str
    power_w: float
    price_eur: float
    manufacturer: str | None = None
    model_number: str | None = None
    datasheet: dict[str, Any] | None = None


@dataclass(frozen=True)
class BatteryOption:
    """
    Battery hardware specification for scenario optimization.

    Encapsulates the technical and cost parameters of a specific battery
    model. Used to generate energy storage configurations during optimization.

    Attributes:
        name: Human-readable battery name/model identifier.
        specs: Battery specifications (capacity, cycle life).
        price_eur: Cost per battery module (EUR).
        manufacturer: Manufacturer name (optional metadata).
        model_number: Manufacturer model number (optional metadata).
        datasheet: Additional technical data (optional).

    Example:
        ```python
        battery_lfp = BatteryOption(
            name="BYD Battery-Box Premium LVS 4.0",
            specs=BatterySpecs(capacity_kwh=4.0, cycles_life=8000),
            price_eur=1500.0,
            manufacturer="BYD",
            model_number="LVS 4.0"
        )

        battery_tesla = BatteryOption(
            name="Tesla Powerwall 2",
            specs=BatterySpecs(capacity_kwh=13.5, cycles_life=5000),
            price_eur=8000.0,
            manufacturer="Tesla"
        )
        ```
    """
    name: str
    specs: BatterySpecs
    price_eur: float
    manufacturer: str | None = None
    model_number: str | None = None
    datasheet: dict[str, Any] | None = None


