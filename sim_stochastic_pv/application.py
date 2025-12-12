from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .simulation.monte_carlo import EconomicConfig, MonteCarloSimulator
from .simulation.optimizer import ScenarioOptimizer, ScenarioEvaluation
from .persistence import PersistenceService
from .result_builder import ResultBuilder
try:
    from .scenario_setup import (
        build_default_economic_config,
        build_default_energy_config,
        build_default_load_profile,
        build_default_optimization_request,
        build_default_price_model,
        build_default_solar_model,
        load_scenario_data,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback for top-level scripts
    from scenario_setup import (  # type: ignore
        build_default_economic_config,
        build_default_energy_config,
        build_default_load_profile,
        build_default_optimization_request,
        build_default_price_model,
        build_default_solar_model,
        load_scenario_data,
    )
from .simulation.energy_simulator import EnergySystemSimulator
from .simulation.prices import PriceModel

ScenarioData = Mapping[str, Any] | str | Path | None


def _build_summary(evaluation: ScenarioEvaluation) -> Dict[str, Any]:
    """
    Extract the final Monte Carlo metrics for a scenario evaluation.

    Args:
        evaluation: ScenarioEvaluation to summarize.

    Returns:
        Dictionary with final gain, probability of profit, and break-even month.
    """
    df_profit = evaluation.results.df_profit
    final_row = df_profit.iloc[-1]
    return {
        "scenario": evaluation.definition.describe(),
        "final_gain_mean_eur": float(final_row["mean_gain_eur"]),
        "final_gain_real_mean_eur": float(final_row["mean_gain_real_eur"]),
        "prob_gain": float(final_row["prob_gain"]),
        "break_even_month": evaluation.break_even_month,
    }


class SimulationApplication:
    """
    High-level orchestrator used by the CLI and (future) FastAPI surface.
    """

    def __init__(
        self,
        *,
        save_outputs: bool = False,
        persistence: PersistenceService | None = None,
        result_builder: ResultBuilder | None = None,
    ) -> None:
        """
        Args:
            save_outputs: When True, ResultBuilder saves plots/reports.
            persistence: Optional PersistenceService for DB storage.
            result_builder: Optional ResultBuilder for CLI outputs.
        """
        self.save_outputs = save_outputs
        self.persistence = persistence
        self.result_builder = result_builder

    def run_analysis(
        self,
        *,
        n_mc: int | None = None,
        seed: int = 123,
        scenario_data: ScenarioData = None,
    ) -> Dict[str, Any]:
        """
        Execute the single-scenario Monte Carlo analysis.

        Args:
            n_mc: Number of Monte Carlo paths (defaults to scenario setup).
            seed: RNG seed for reproducibility.
            scenario_data: Optional mapping/path overriding the default scenario definition.

        Returns:
            Summary dictionary with economic metrics and optional output path.
        """
        scenario_payload = load_scenario_data(scenario_data)
        load_profile = build_default_load_profile(scenario_payload)
        solar_model = build_default_solar_model(scenario_payload)
        energy_cfg = build_default_energy_config(scenario_payload)
        price_model = build_default_price_model(scenario_data=scenario_payload)
        econ_cfg = build_default_economic_config(n_mc=n_mc, scenario_data=scenario_payload)

        energy_sim = EnergySystemSimulator(
            config=energy_cfg,
            solar_model=solar_model,
            load_profile=load_profile,
        )

        mc = MonteCarloSimulator(
            energy_simulator=energy_sim,
            price_model=price_model,
            economic_config=econ_cfg,
        )

        results = mc.run(seed=seed)
        scenario_name = scenario_payload.get("scenario_name", "custom_scenario")

        summary = {
            "scenario": scenario_name,
            "final_gain_mean_eur": float(results.df_profit["mean_gain_eur"].iloc[-1]),
            "final_gain_real_mean_eur": float(results.df_profit["mean_gain_real_eur"].iloc[-1]),
            "prob_gain": float(results.df_profit["prob_gain"].iloc[-1]),
            "plots_data": {
                "profit": {
                    "months": results.df_profit["month_index"].tolist(),
                    "mean_gain_eur": results.df_profit["mean_gain_eur"].tolist(),
                    "p05_gain_eur": results.df_profit["p05_gain_eur"].tolist(),
                    "p95_gain_eur": results.df_profit["p95_gain_eur"].tolist(),
                    "mean_gain_real_eur": results.df_profit["mean_gain_real_eur"].tolist(),
                },
                "energy_monthly": {
                    "months": results.df_energy["month_index"].tolist(),
                    "pv_prod_mean_kwh": results.df_energy["pv_prod_mean_kwh"].tolist(),
                    "solar_used_mean_kwh": results.df_energy["solar_used_mean_kwh"].tolist(),
                    "grid_import_mean_kwh": results.df_energy["grid_import_mean_kwh"].tolist(),
                    "savings_mean_kwh": results.df_energy["savings_mean_kwh"].tolist(),
                },
                "soc_profile": {
                    "hours": list(range(24)),
                    "months_data": [
                        {
                            "month": m,
                            "soc_mean": results.df_soc[results.df_soc["month_in_year"] == m].sort_values("hour")["soc_mean"].tolist()
                        }
                        for m in range(12)
                    ]
                }
            }
        }

        output_dir = None
        if self.save_outputs and self.result_builder:
            output_dir = self.result_builder.build_analysis(
                scenario_name,
                results=results,
                energy_config=energy_cfg,
                economic_config=econ_cfg,
                price_model=price_model,
            )

        if self.persistence:
            scenario_record = self.persistence.record_scenario(
                scenario_name,
                config=scenario_payload,
                metadata={
                    "economic": asdict(econ_cfg),
                    "scenario_name": scenario_name,
                },
            )
            self.persistence.record_run_result(
                "analysis",
                summary,
                scenario=scenario_record,
                output_dir=str(output_dir) if output_dir else None,
            )

        summary["output_dir"] = str(output_dir) if output_dir else None
        return summary

    def run_optimization(
        self,
        *,
        seed: int = 321,
        n_mc: int | None = None,
        scenario_data: ScenarioData = None,
    ) -> Dict[str, Any]:
        """
        Execute the optimization batch covering all configured scenarios.

        Args:
            seed: RNG seed propagated to ScenarioOptimizer.
            n_mc: Override for Monte Carlo paths used per scenario.
            scenario_data: Optional mapping/path overriding the default scenario definition.

        Returns:
            Dictionary containing the number of evaluations and optional output dir.
        """
        scenario_payload = load_scenario_data(scenario_data)
        request = build_default_optimization_request(scenario_payload)
        base_energy_cfg = build_default_energy_config(scenario_payload)
        econ_cfg = build_default_economic_config(n_mc=n_mc, scenario_data=scenario_payload)
        price_model = build_default_price_model(scenario_data=scenario_payload)
        solar_model = build_default_solar_model(scenario_payload)

        optimizer = ScenarioOptimizer(
            request=request,
            base_energy_config=base_energy_cfg,
            economic_config_template=econ_cfg,
            price_model=price_model,
            load_profile_factory=lambda payload=scenario_payload: build_default_load_profile(payload),
            solar_model=solar_model,
        )
        evaluations = optimizer.run(seed=seed)

        output_dir = None
        if self.save_outputs and self.result_builder:
            output_dir = self.result_builder.build_optimization_bundle(
                request.scenario_name,
                evaluations,
                price_model,
            )

        metadata = {"evaluations": len(evaluations)}
        optimization_record = None
        if self.persistence:
            optimization_record = self.persistence.record_optimization(
                request.scenario_name,
                request_payload={
                    "scenario": scenario_payload,
                    "request": asdict(request),
                },
                metadata=metadata,
            )
            for ev in evaluations:
                inverter_record = self.persistence.upsert_inverter(ev.definition.inverter)
                panel_record = self.persistence.upsert_panel(ev.definition.panel)
                battery_payload = None
                if ev.definition.integrated_battery_specs and ev.definition.integrated_battery_count > 0:
                    battery_payload = {
                        "name": f"{ev.definition.inverter.name}-integrated",
                        "manufacturer": getattr(ev.definition.inverter, "manufacturer", None),
                        "model_number": None,
                        "datasheet": getattr(ev.definition.inverter, "datasheet", None),
                        "specs": {
                            "capacity_kwh": ev.definition.integrated_battery_specs.capacity_kwh,
                            "cycles_life": ev.definition.integrated_battery_specs.cycles_life,
                        },
                    }
                elif ev.definition.battery_option and ev.definition.battery_count > 0:
                    battery_payload = {
                        "name": ev.definition.battery_option.name,
                        "manufacturer": ev.definition.battery_option.manufacturer,
                        "model_number": ev.definition.battery_option.model_number,
                        "datasheet": ev.definition.battery_option.datasheet,
                        "specs": asdict(ev.definition.battery_option.specs),
                    }
                battery_record = self.persistence.upsert_battery(battery_payload) if battery_payload else None

                scenario_record = self.persistence.record_scenario(
                    ev.definition.describe(),
                    config={
                        "pv_kwp": ev.definition.pv_kwp,
                        "investment_eur": ev.definition.investment_eur,
                        "panel_count": ev.definition.panel_count,
                        "battery_count": ev.definition.battery_count,
                    },
                    metadata={
                        "integrated_battery_count": ev.definition.integrated_battery_count,
                    },
                    inverter=inverter_record,
                    panel=panel_record,
                    battery=battery_record,
                )
                summary = _build_summary(ev)
                self.persistence.record_run_result(
                    "optimization",
                    summary,
                    scenario=scenario_record,
                    optimization=optimization_record,
                    output_dir=str(output_dir) if output_dir else None,
                )

        return {
            "evaluations": len(evaluations),
            "output_dir": str(output_dir) if output_dir else None,
        }
