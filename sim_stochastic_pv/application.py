from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np

from .simulation.monte_carlo import (
    EconomicConfig,
    MonteCarloResults,
    MonteCarloSimulator,
)
from .simulation.optimizer import ScenarioOptimizer, ScenarioEvaluation

# Number of full price trajectories returned to the UI for fan-chart rendering.
# Capped to keep the JSON response light: a fan chart loses visual value beyond
# a few dozen lines anyway, and the bands (mean/p05/p95) carry the statistical
# story already.
_PRICE_SAMPLE_PATHS_LIMIT: int = 20
from .persistence import PersistenceService
from .output import ResultBuilder
from .scenario_builder import (
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


def _build_price_plot_payload(
    results: MonteCarloResults,
    sample_paths_limit: int = _PRICE_SAMPLE_PATHS_LIMIT,
) -> Dict[str, Any]:
    """
    Build the JSON-friendly price block of the analysis response.

    Produces the data needed by the Dashboard "Prezzo energia" tab:

    * Per-month statistics across all Monte Carlo paths (mean, p05, p95)
      to draw the uncertainty band as a filled area.
    * A capped sample of full trajectories so the Dashboard can overlay
      a fan chart of representative paths on top of the band.

    The sample is selected via a deterministic stride
    ``stride = max(1, n_mc // sample_paths_limit)`` so the chosen paths
    are spread across the index space and the result is reproducible
    when the Monte Carlo seed is fixed.

    Args:
        results: The MonteCarloResults bundle from
            :meth:`MonteCarloSimulator.run`.
        sample_paths_limit: Maximum number of full paths to include in
            the response. Hard-capped to a reasonable visualisation
            density. Defaults to ``_PRICE_SAMPLE_PATHS_LIMIT`` (20).

    Returns:
        Dictionary with the following keys:

        - ``months``: list[int], 0-based month indices
        - ``mean_eur_per_kwh``: list[float], mean price per month
        - ``p05_eur_per_kwh``: list[float], 5th-percentile price per month
        - ``p95_eur_per_kwh``: list[float], 95th-percentile price per month
        - ``sample_paths``: list[list[float]] — each inner list is one
          full ``(n_months,)`` price trajectory in EUR/kWh.

    Example:
        ```python
        payload = _build_price_plot_payload(results)
        # In the frontend (Chart.js / Svelte):
        # - render the (p05, p95) band as a filled area dataset
        # - render `mean_eur_per_kwh` as a solid line on top
        # - render each `sample_paths[i]` as a thin semi-transparent line
        ```

    Notes:
        - For deterministic price models (volatility=0, escalation no-jitter)
          the band collapses to a line and the sample paths are all
          identical: the fan chart visually degenerates, which is the
          correct signal that the model has no uncertainty.
        - The function is intentionally pure (no I/O); callers control
          where the payload is serialised (API response, file, etc.).
    """
    df_price = results.df_price
    months = df_price["month_index"].tolist()
    mean = df_price["price_mean_eur_per_kwh"].tolist()
    p05 = df_price["price_p05_eur_per_kwh"].tolist()
    p95 = df_price["price_p95_eur_per_kwh"].tolist()

    paths = results.price_paths_eur_per_kwh
    n_mc, _n_months = paths.shape
    if n_mc <= sample_paths_limit:
        selected_idx = np.arange(n_mc)
    else:
        stride = max(1, n_mc // sample_paths_limit)
        selected_idx = np.arange(0, n_mc, stride)[:sample_paths_limit]

    sample_paths: List[List[float]] = [
        paths[i, :].tolist() for i in selected_idx
    ]

    return {
        "months": months,
        "mean_eur_per_kwh": mean,
        "p05_eur_per_kwh": p05,
        "p95_eur_per_kwh": p95,
        "sample_paths": sample_paths,
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
        # Phase 8: if the scenario references DB entities by ID (load_profile_id,
        # price_profile_id, inverter_id, …) expand them into full specs before
        # the builders try to read inline blocks. Safe no-op when no IDs are
        # present in the payload.
        if self.persistence is not None and any(
            key in scenario_payload
            for key in (
                "load_profile_id",
                "price_profile_id",
                "inverter_id",
                "panel_id",
                "battery_id",
            )
        ):
            scenario_payload = self.persistence.hydrate_scenario_from_ids(
                scenario_payload
            )
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

        price_plot = _build_price_plot_payload(results)

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
                },
                "price": price_plot,
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
        seed: int = 123,  # Default seed unified to 123 for consistency across workflows
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

            # OPTIMIZATION: Collect unique hardware before loop to reduce database calls
            # (100 scenarios × 3 hardware types = 300 calls → ~10-20 calls typically)
            unique_inverters = {}
            unique_panels = {}
            unique_batteries = {}

            for ev in evaluations:
                # Collect unique inverters
                unique_inverters[ev.definition.inverter.name] = ev.definition.inverter

                # Collect unique panels
                unique_panels[ev.definition.panel.name] = ev.definition.panel

                # Collect unique batteries (handle integrated vs separate)
                if ev.definition.integrated_battery_specs and ev.definition.integrated_battery_count > 0:
                    battery_key = f"{ev.definition.inverter.name}-integrated"
                    unique_batteries[battery_key] = {
                        "name": battery_key,
                        "manufacturer": getattr(ev.definition.inverter, "manufacturer", None),
                        "model_number": None,
                        "datasheet": getattr(ev.definition.inverter, "datasheet", None),
                        "specs": {
                            "capacity_kwh": ev.definition.integrated_battery_specs.capacity_kwh,
                            "cycles_life": ev.definition.integrated_battery_specs.cycles_life,
                        },
                    }
                elif ev.definition.battery_option and ev.definition.battery_count > 0:
                    battery_key = ev.definition.battery_option.name
                    unique_batteries[battery_key] = {
                        "name": ev.definition.battery_option.name,
                        "manufacturer": ev.definition.battery_option.manufacturer,
                        "model_number": ev.definition.battery_option.model_number,
                        "datasheet": ev.definition.battery_option.datasheet,
                        "specs": asdict(ev.definition.battery_option.specs),
                    }

            # Batch upsert all unique hardware (reduces N*3 calls to U*3 where U = unique count)
            inverter_map = {
                name: self.persistence.upsert_inverter(inv)
                for name, inv in unique_inverters.items()
            }
            panel_map = {
                name: self.persistence.upsert_panel(panel)
                for name, panel in unique_panels.items()
            }
            battery_map = {
                key: self.persistence.upsert_battery(battery)
                for key, battery in unique_batteries.items()
            }

            # Loop: Use cached hardware records instead of upserting
            for ev in evaluations:
                # Lookup from cached maps
                inverter_record = inverter_map[ev.definition.inverter.name]
                panel_record = panel_map[ev.definition.panel.name]

                # Lookup battery (if applicable)
                battery_record = None
                if ev.definition.integrated_battery_specs and ev.definition.integrated_battery_count > 0:
                    battery_key = f"{ev.definition.inverter.name}-integrated"
                    battery_record = battery_map[battery_key]
                elif ev.definition.battery_option and ev.definition.battery_count > 0:
                    battery_key = ev.definition.battery_option.name
                    battery_record = battery_map[battery_key]

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
