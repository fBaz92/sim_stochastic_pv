from __future__ import annotations

import sys
import time
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from .battery import BatterySpecs
from .energy_simulator import EnergySystemConfig, EnergySystemSimulator
from .load_profiles import LoadProfile
from .monte_carlo import EconomicConfig, MonteCarloResults, MonteCarloSimulator
from .prices import PriceModel
from .solar import SolarModel


@dataclass(frozen=True)
class InverterOption:
    name: str
    p_ac_max_kw: float
    price_eur: float
    p_dc_max_kw: float | None = None
    install_cost_eur: float | None = None
    integrated_battery_specs: BatterySpecs | None = None
    integrated_battery_price_eur: float | None = None
    integrated_battery_count_options: List[int] | None = None

    def total_cost(self) -> float:
        return self.price_eur + self.installation_cost()

    def installation_cost(self) -> float:
        if self.install_cost_eur is not None:
            return self.install_cost_eur
        return 1000.0 if self.p_ac_max_kw <= 0.8 else 2000.0


@dataclass(frozen=True)
class PanelOption:
    name: str
    power_w: float
    price_eur: float


@dataclass(frozen=True)
class BatteryOption:
    name: str
    specs: BatterySpecs
    price_eur: float


@dataclass
class ScenarioDefinition:
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
        return (self.panel.power_w * self.panel_count) / 1000.0

    @property
    def investment_eur(self) -> float:
        total = self.inverter.price_eur + self.inverter.installation_cost()
        total += self.panel.price_eur * self.panel_count
        if self.battery_option and self.battery_count > 0:
            total += self.battery_option.price_eur * self.battery_count
        if self.integrated_battery_specs and self.integrated_battery_count > 0:
            total += self.integrated_battery_price_eur * self.integrated_battery_count
        return total

    def describe(self) -> str:
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
    scenario_name: str
    inverter_options: List[InverterOption]
    panel_options: List[PanelOption]
    panel_count_options: List[int]
    battery_options: List[BatteryOption]
    battery_count_options: List[int]
    include_no_battery: bool = True


@dataclass
class ScenarioEvaluation:
    definition: ScenarioDefinition
    economic_config: EconomicConfig
    energy_config: EnergySystemConfig
    results: MonteCarloResults
    break_even_month: int | None
    final_gain_eur: float


class ScenarioOptimizer:
    def __init__(
        self,
        request: OptimizationRequest,
        base_energy_config: EnergySystemConfig,
        economic_config_template: EconomicConfig,
        price_model: PriceModel,
        load_profile_factory: Callable[[], LoadProfile],
        solar_model: SolarModel,
    ) -> None:
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
