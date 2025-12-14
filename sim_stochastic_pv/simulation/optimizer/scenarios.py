"""
Scenario definition and evaluation dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .hardware import InverterOption, PanelOption, BatteryOption

@dataclass
class ScenarioDefinition:
    """
    Complete PV system hardware configuration scenario.

    Defines a specific combination of inverter, solar panels, and battery
    for techno-economic evaluation. Represents one point in the optimization
    search space.

    Attributes:
        inverter: Selected inverter option.
        panel: Selected panel option.
        panel_count: Number of panels to install.
        battery_option: Selected external battery option (or None).
        battery_count: Number of external battery modules.
        integrated_battery_specs: Integrated battery specs (if applicable).
        integrated_battery_price_eur: Price per integrated battery.
        integrated_battery_count: Number of integrated batteries.

    Example:
        ```python
        scenario = ScenarioDefinition(
            inverter=InverterOption(name="Fronius", p_ac_max_kw=5.0, price_eur=1500),
            panel=PanelOption(name="Longi 540W", power_w=540, price_eur=150),
            panel_count=10,  # 5.4 kWp system
            battery_option=BatteryOption(
                name="BYD 4kWh",
                specs=BatterySpecs(capacity_kwh=4.0, cycles_life=8000),
                price_eur=1500
            ),
            battery_count=2  # 8 kWh total storage
        )

        print(f"System: {scenario.pv_kwp:.1f} kWp")  # 5.4 kWp
        print(f"Investment: {scenario.investment_eur:.0f} EUR")  # Total cost
        print(scenario.describe())  # Human-readable description
        ```
    """
    inverter: InverterOption
    panel: PanelOption
    panel_count: int
    battery_option: BatteryOption | None
    battery_count: int
    integrated_battery_specs: BatterySpecs | None = None
    integrated_battery_price_eur: float = 0.0
    integrated_battery_count: int = 0

    @property
    def pv_kwp(self) -> float:
        """
        Calculate total PV system peak power in kWp.

        Returns:
            Total nominal peak power (kWp) = panel_power × panel_count / 1000.
        """
        return (self.panel.power_w * self.panel_count) / 1000.0

    @property
    def investment_eur(self) -> float:
        """
        Calculate total upfront investment cost.

        Sums all hardware and installation costs:
        - Inverter (hardware + installation)
        - Solar panels
        - External batteries (if any)
        - Integrated batteries (if any)

        Returns:
            Total investment cost in EUR.
        """
        total = self.inverter.price_eur + self.inverter.installation_cost()
        total += self.panel.price_eur * self.panel_count
        if self.battery_option and self.battery_count > 0:
            total += self.battery_option.price_eur * self.battery_count
        if self.integrated_battery_specs and self.integrated_battery_count > 0:
            total += self.integrated_battery_price_eur * self.integrated_battery_count
        return total

    def describe(self) -> str:
        """
        Generate human-readable scenario description.

        Returns:
            String describing inverter, panel configuration, and battery setup.

        Example:
            "Fronius Primo 5.0 | 10xLongi 540W (5.40 kWp) | 2xBYD 4kWh"
        """
        if self.integrated_battery_specs and self.integrated_battery_count > 0:
            batt_desc = f"{self.integrated_battery_count}x{self.inverter.name}-int"
        elif self.battery_option and self.battery_count > 0:
            batt_desc = f"{self.battery_count}x{self.battery_option.name}"
        else:
            batt_desc = "no-batt"
        return (
            f"{self.inverter.name} | {self.panel_count}x{self.panel.name} "
            f"({self.pv_kwp:.2f} kWp) | {batt_desc}"
        )


@dataclass
class OptimizationRequest:
    """
    Configuration for PV system scenario optimization.

    Defines the search space for hardware combinations to evaluate.
    The optimizer generates all valid combinations of the specified options.

    Attributes:
        scenario_name: Descriptive name for this optimization run.
        inverter_options: List of inverter options to consider.
        panel_options: List of panel options to consider.
        panel_count_options: List of panel counts to try (e.g., [8, 10, 12]).
        battery_options: List of battery options to consider.
        battery_count_options: List of battery counts to try (e.g., [1, 2, 3]).
        include_no_battery: If True, include scenarios without batteries.

    Example:
        ```python
        request = OptimizationRequest(
            scenario_name="Residential PV Optimization 2025",
            inverter_options=[
                InverterOption(name="Fronius 5kW", p_ac_max_kw=5.0, price_eur=1500),
                InverterOption(name="SMA 6kW", p_ac_max_kw=6.0, price_eur=1800),
            ],
            panel_options=[
                PanelOption(name="Panel 400W", power_w=400, price_eur=120),
                PanelOption(name="Panel 540W", power_w=540, price_eur=150),
            ],
            panel_count_options=[8, 10, 12],  # Try 3 different array sizes
            battery_options=[
                BatteryOption(
                    name="Battery 5kWh",
                    specs=BatterySpecs(capacity_kwh=5.0, cycles_life=6000),
                    price_eur=1800
                ),
            ],
            battery_count_options=[1, 2],  # Try 5kWh or 10kWh
            include_no_battery=True  # Also test without battery
        )
        # This generates: 2 inverters × 2 panels × 3 counts × 3 battery configs = 36 scenarios
        ```

    Notes:
        - Total scenarios = len(inverters) × len(panels) × len(panel_counts) ×
          (1 + len(batteries) × len(battery_counts)) if include_no_battery
        - Large search spaces can be computationally expensive
        - Each scenario runs full Monte Carlo simulation
    """
    scenario_name: str
    inverter_options: List[InverterOption]
    panel_options: List[PanelOption]
    panel_count_options: List[int]
    battery_options: List[BatteryOption]
    battery_count_options: List[int]
    include_no_battery: bool = True


@dataclass
class ScenarioEvaluation:
    """
    Complete evaluation results for one hardware scenario.

    Contains the scenario configuration, simulation parameters, and
    Monte Carlo results for techno-economic analysis.

    Attributes:
        definition: Hardware configuration that was evaluated.
        economic_config: Economic parameters used (investment, MC count).
        energy_config: Energy system configuration used.
        results: Full Monte Carlo simulation results.
        break_even_month: Month when cumulative gain reaches zero (payback), or None.
        final_gain_eur: Net profit/loss at end of simulation period (EUR).

    Example:
        ```python
        # After running optimizer
        for evaluation in optimizer.run():
            print(f"Scenario: {evaluation.definition.describe()}")
            print(f"Investment: {evaluation.definition.investment_eur:.0f} EUR")
            if evaluation.break_even_month:
                years = evaluation.break_even_month / 12
                print(f"Payback: {years:.1f} years")
            else:
                print("No payback within simulation period")
            print(f"Final gain: {evaluation.final_gain_eur:.0f} EUR")
            print(f"Mean IRR: {evaluation.results.df_irr['irr_mean'].iloc[-1]:.2%}")
        ```

    Notes:
        - break_even_month is None if investment never pays back
        - final_gain_eur can be negative (loss) or positive (profit)
        - results contains full time series and statistical distributions
    """
    definition: ScenarioDefinition
    economic_config: EconomicConfig
    energy_config: EnergySystemConfig
    results: MonteCarloResults
    break_even_month: int | None
    final_gain_eur: float


