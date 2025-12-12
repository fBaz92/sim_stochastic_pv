"""
Scenario optimization and hardware configuration helpers.

Provides tools for evaluating multiple PV system configurations (inverters,
panels, batteries) through automated Monte Carlo simulations. Enables
comparison of different hardware combinations to find optimal economic outcomes.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from .battery import BatterySpecs
from .energy_simulator import EnergySystemConfig, EnergySystemSimulator
from .load_profiles import LoadProfile
from .monte_carlo import EconomicConfig, MonteCarloResults, MonteCarloSimulator
from .prices import PriceModel
from .solar import SolarModel


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
    p_dc_max_kw: float | None = None
    install_cost_eur: float | None = None
    integrated_battery_specs: BatterySpecs | None = None
    integrated_battery_price_eur: float | None = None
    integrated_battery_count_options: List[int] | None = None
    manufacturer: str | None = None
    model_number: str | None = None
    datasheet: dict[str, Any] | None = None

    def total_cost(self) -> float:
        """Calculate total inverter cost (hardware + installation)."""
        return self.price_eur + self.installation_cost()

    def installation_cost(self) -> float:
        """
        Calculate or return installation cost.

        Returns install_cost_eur if specified, otherwise estimates based on
        inverter size (small ≤0.8kW: 1000 EUR, larger: 2000 EUR).
        """
        if self.install_cost_eur is not None:
            return self.install_cost_eur
        return 1000.0 if self.p_ac_max_kw <= 0.8 else 2000.0


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


class ScenarioOptimizer:
    """
    Automated PV system scenario optimizer with Monte Carlo evaluation.

    Generates and evaluates all hardware combinations specified in an
    OptimizationRequest, running full Monte Carlo simulations for each
    to determine techno-economic performance. Provides progress tracking
    and comparative results for decision support.

    The optimizer systematically explores the configuration space defined
    by combinations of:
    - Inverter options
    - Panel types and counts
    - Battery types and counts
    - Integrated battery configurations

    Each valid combination is simulated using the provided weather, load,
    and price models to estimate financial performance (NPV, IRR, payback).

    Attributes:
        request: Optimization configuration defining search space.
        base_energy_config: Template energy system configuration.
        economic_config_template: Template economic parameters (MC count).
        price_model: Electricity price model for economic calculations.
        load_profile_factory: Factory creating load profile instances.
        solar_model: Solar production model (location, weather data).

    Example:
        ```python
        from sim_stochastic_pv.simulation.optimizer import (
            ScenarioOptimizer,
            OptimizationRequest,
            InverterOption,
            PanelOption,
            BatteryOption
        )
        from sim_stochastic_pv.simulation.battery import BatterySpecs
        from sim_stochastic_pv.simulation.monte_carlo import EconomicConfig
        from sim_stochastic_pv.simulation.energy_simulator import EnergySystemConfig

        # Define hardware options
        request = OptimizationRequest(
            scenario_name="Residential PV 2025",
            inverter_options=[
                InverterOption(name="Inv5kW", p_ac_max_kw=5.0, price_eur=1500),
            ],
            panel_options=[
                PanelOption(name="Panel400W", power_w=400, price_eur=120),
            ],
            panel_count_options=[10, 12, 14],  # Try different array sizes
            battery_options=[
                BatteryOption(
                    name="Bat5kWh",
                    specs=BatterySpecs(capacity_kwh=5.0, cycles_life=6000),
                    price_eur=1800
                ),
            ],
            battery_count_options=[1, 2],
            include_no_battery=True
        )

        # Create optimizer
        optimizer = ScenarioOptimizer(
            request=request,
            base_energy_config=base_energy_config,
            economic_config_template=EconomicConfig(investment_eur=0, n_mc=100),
            price_model=price_model,
            load_profile_factory=lambda: load_profile,
            solar_model=solar_model
        )

        # Run optimization
        evaluations = optimizer.run(seed=42)

        # Analyze results
        best = max(evaluations, key=lambda e: e.final_gain_eur)
        print(f"Best scenario: {best.definition.describe()}")
        print(f"Final gain: {best.final_gain_eur:.0f} EUR")
        print(f"Payback: {best.break_even_month/12:.1f} years")
        ```

    Notes:
        - Computational cost scales with number of scenarios × MC paths
        - Progress tracking displays real-time ETA and scenario details
        - Results can be sorted/filtered by various metrics (gain, IRR, payback)
        - Each scenario uses independent random seed for reproducibility
    """

    def __init__(
        self,
        request: OptimizationRequest,
        base_energy_config: EnergySystemConfig,
        economic_config_template: EconomicConfig,
        price_model: PriceModel,
        load_profile_factory: Callable[[], LoadProfile],
        solar_model: SolarModel,
    ) -> None:
        """
        Initialize scenario optimizer with configuration and models.

        Args:
            request: Optimization request defining hardware search space.
            base_energy_config: Template energy system configuration.
                Hardware-specific parameters (pv_kwp, battery_specs, etc.)
                will be overridden for each scenario.
            economic_config_template: Template economic parameters.
                investment_eur is overridden per scenario, n_mc is preserved.
            price_model: Electricity price model for economic calculations.
                Should be configured with desired escalation and stochasticity.
            load_profile_factory: Factory creating fresh load profile instances.
                Called once per scenario to ensure independent stochastic state.
            solar_model: Solar production model with weather data and location.

        Example:
            ```python
            optimizer = ScenarioOptimizer(
                request=OptimizationRequest(...),
                base_energy_config=EnergySystemConfig(
                    pv_kwp=0.0,  # Overridden per scenario
                    inverter_p_ac_max_kw=5.0,  # Can be overridden
                    battery_specs=BatterySpecs(...),  # Can be overridden
                    n_batteries=0  # Overridden per scenario
                ),
                economic_config_template=EconomicConfig(
                    investment_eur=0.0,  # Overridden per scenario
                    n_mc=500  # Used for all scenarios
                ),
                price_model=EscalatingPriceModel(...),
                load_profile_factory=lambda: create_load_profile(),
                solar_model=SolarModel(...)
            )
            ```

        Notes:
            - load_profile_factory should return fresh instances (not shared)
            - price_model is shared across scenarios (reset per MC path)
            - solar_model is shared (contains deterministic weather data)
        """
        self.request = request
        self.base_energy_config = base_energy_config
        self.economic_config_template = economic_config_template
        self.price_model = price_model
        self.load_profile_factory = load_profile_factory
        self.solar_model = solar_model
        self._progress_active = False

    def _generate_scenarios(self) -> List[ScenarioDefinition]:
        scenarios: List[ScenarioDefinition] = []
        battery_options = list(self.request.battery_options)
        if self.request.include_no_battery:
            battery_options = [None] + battery_options

        for inv, panel, panel_count in product(
            self.request.inverter_options,
            self.request.panel_options,
            self.request.panel_count_options,
        ):
            if panel_count <= 0:
                continue
            if inv.integrated_battery_specs is not None:
                if inv.integrated_battery_price_eur is None:
                    raise ValueError(
                        f"Inverter {inv.name} richiede integrated_battery_price_eur definito"
                    )
                counts = inv.integrated_battery_count_options or [1]
                for integ_count in counts:
                    if integ_count <= 0:
                        continue
                    scenarios.append(
                        ScenarioDefinition(
                            inverter=inv,
                            panel=panel,
                            panel_count=panel_count,
                            battery_option=None,
                            battery_count=0,
                            integrated_battery_specs=inv.integrated_battery_specs,
                            integrated_battery_price_eur=inv.integrated_battery_price_eur or 0.0,
                            integrated_battery_count=integ_count,
                        )
                    )
            else:
                for batt_option in battery_options:
                    battery_counts = (
                        self.request.battery_count_options
                        if batt_option is not None
                        else [0]
                    )
                    for batt_count in battery_counts:
                        if batt_option is None and batt_count != 0:
                            continue
                        if batt_option is not None and batt_count <= 0:
                            continue
                        scenarios.append(
                            ScenarioDefinition(
                                inverter=inv,
                                panel=panel,
                                panel_count=panel_count,
                                battery_option=batt_option,
                                battery_count=batt_count,
                            )
                        )
        return scenarios

    def _build_energy_config(self, definition: ScenarioDefinition) -> EnergySystemConfig:
        if definition.integrated_battery_specs is not None:
            battery_specs = definition.integrated_battery_specs
            n_batteries = definition.integrated_battery_count
        else:
            battery_specs = (
                definition.battery_option.specs
                if definition.battery_option is not None
                else self.base_energy_config.battery_specs
            )
            n_batteries = definition.battery_count if definition.battery_option else 0

        return replace(
            self.base_energy_config,
            pv_kwp=definition.pv_kwp,
            inverter_p_ac_max_kw=definition.inverter.p_ac_max_kw,
            inverter_p_dc_max_kw=definition.inverter.p_dc_max_kw,
            battery_specs=battery_specs,
            n_batteries=n_batteries,
        )

    def _build_economic_config(self, definition: ScenarioDefinition) -> EconomicConfig:
        return EconomicConfig(
            investment_eur=definition.investment_eur,
            n_mc=self.economic_config_template.n_mc,
        )

    @staticmethod
    def _format_bar(frac: float, length: int = 30) -> str:
        frac = max(0.0, min(1.0, frac))
        filled = int(length * frac)
        return "#" * filled + "-" * (length - filled)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0.0, seconds)
        total_seconds = int(round(seconds))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        if minutes > 0:
            return f"{minutes:02d}:{secs:02d}"
        return f"{secs:02d}s"

    def _format_line(
        self,
        scenario_idx: int,
        total_scens: int,
        scenario_desc: str,
        mc_done: int,
        mc_total: int,
        scenario_elapsed: float,
        scenario_eta: float,
        global_elapsed: float,
        global_eta: float,
    ) -> str:
        frac = (scenario_idx + mc_done / mc_total) / total_scens if total_scens > 0 else 0.0
        bar = self._format_bar(frac)
        desc = scenario_desc
        if len(desc) > 60:
            desc = desc[:57] + "..."
        scenario_elapsed_str = self._format_duration(scenario_elapsed)
        scenario_eta_str = self._format_duration(scenario_eta)
        global_elapsed_str = self._format_duration(global_elapsed)
        global_eta_str = self._format_duration(global_eta)
        return (
            f"OPT {scenario_idx+1:4d}/{total_scens:<4d} "
            f"MC {mc_done:4d}/{mc_total:<4d} "
            f"[{bar}] {frac*100:6.2f}%  "
            f"MC elapsed: {scenario_elapsed_str:>8}  ETA scen: {scenario_eta_str:>8}  "
            f"glob elapsed: {global_elapsed_str:>8}  ETA glob: {global_eta_str:>8}  | {desc}"
        )

    def _render_progress(self, line: str, final: bool = False) -> None:
        sys.stdout.write("\r\x1b[2K" + line)
        if final:
            sys.stdout.write("\n")
        sys.stdout.flush()

    def _evaluate_scenario(
        self,
        definition: ScenarioDefinition,
        seed: int,
        progress_callback: Callable[[int, int, float, float], None] | None = None,
    ) -> ScenarioEvaluation:
        energy_config = self._build_energy_config(definition)
        economic_config = self._build_economic_config(definition)
        load_profile = self.load_profile_factory()
        energy_simulator = EnergySystemSimulator(
            config=energy_config,
            solar_model=self.solar_model,
            load_profile=load_profile,
        )
        mc = MonteCarloSimulator(
            energy_simulator=energy_simulator,
            price_model=self.price_model,
            economic_config=economic_config,
        )
        results = mc.run(
            seed=seed,
            progress_callback=progress_callback,
            show_progress=False,
        )
        break_even_month = self._compute_break_even(results.df_profit)
        final_gain = float(results.df_profit["mean_gain_eur"].iloc[-1])
        return ScenarioEvaluation(
            definition=definition,
            economic_config=economic_config,
            energy_config=energy_config,
            results=results,
            break_even_month=break_even_month,
            final_gain_eur=final_gain,
        )

    @staticmethod
    def _compute_break_even(df_profit) -> int | None:
        mask = df_profit["mean_gain_eur"] >= 0.0
        if not mask.any():
            return None
        return int(df_profit[mask]["month_index"].iloc[0])

    def run(self, seed: int = 1234) -> List[ScenarioEvaluation]:
        scenarios = self._generate_scenarios()
        if not scenarios:
            return []

        evaluations: List[ScenarioEvaluation] = []
        start_time = time.time()
        completed = 0
        total = len(scenarios)

        def update_progress(
            mc_done: int,
            mc_total: int,
            mc_elapsed: float,
            mc_eta: float,
            *,
            scenario_idx: int,
            scenario_desc: str,
        ) -> None:
            global_elapsed = time.time() - start_time
            overall_progress = (scenario_idx + mc_done / mc_total) / total
            global_eta = (
                (global_elapsed / overall_progress - global_elapsed)
                if overall_progress > 0
                else 0.0
            )
            line = self._format_line(
                scenario_idx=scenario_idx,
                total_scens=total,
                scenario_desc=scenario_desc,
                mc_done=mc_done,
                mc_total=mc_total,
                scenario_elapsed=mc_elapsed,
                scenario_eta=mc_eta,
                global_elapsed=global_elapsed,
                global_eta=global_eta,
            )
            self._render_progress(line)

        for idx, definition in enumerate(scenarios):
            scenario_seed = seed + idx * 1000
            scenario_desc = definition.describe()

            def mc_callback(done, total_mc, elapsed_mc, eta_mc, *, idx=idx, desc=scenario_desc):
                update_progress(
                    mc_done=done,
                    mc_total=total_mc,
                    mc_elapsed=elapsed_mc,
                    mc_eta=eta_mc,
                    scenario_idx=idx,
                    scenario_desc=desc,
                )

            evaluation = self._evaluate_scenario(
                definition,
                scenario_seed,
                progress_callback=mc_callback,
            )
            evaluations.append(evaluation)
            completed += 1
            completed_line = f"Scenario {idx+1}/{total} completato."
            self._render_progress(completed_line)

        final_line = "Tutti gli scenari completati."
        self._render_progress(final_line, final=True)
        return evaluations
