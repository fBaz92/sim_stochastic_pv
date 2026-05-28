from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

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


# Phase 11 — same cap applied to the inflation fan chart, so the JSON
# payload stays well under the SQLite practical limit for `summary` JSON.
_INFLATION_SAMPLE_PATHS_LIMIT: int = _PRICE_SAMPLE_PATHS_LIMIT


def _build_inflation_plot_payload(
    results: MonteCarloResults,
    sample_paths_limit: int = _INFLATION_SAMPLE_PATHS_LIMIT,
) -> Dict[str, Any] | None:
    """
    Build the JSON-friendly inflation block for the Dashboard fan chart.

    Returns ``None`` when the simulator ran in deterministic mode — the
    fan chart degenerates to a single line in that case and is not worth
    rendering. When stochastic, the payload mirrors the price block:
    per-year aggregates plus a capped set of full sample trajectories
    of the cumulative inflation factor.

    Args:
        results: MonteCarloResults from a stochastic-inflation run.
        sample_paths_limit: Maximum number of cumulative-factor sample
            paths included in the payload (deterministic stride).

    Returns:
        Dict with keys ``years``, ``mean_factor``, ``p05_factor``,
        ``p95_factor``, ``mean_rate``, ``sample_paths`` — or None if the
        simulator was deterministic.
    """
    if results.df_inflation is None or results.inflation_annual_rates_paths is None:
        return None

    df_inf = results.df_inflation
    rates_paths = results.inflation_annual_rates_paths  # (n_mc, n_years)
    n_mc, n_years = rates_paths.shape

    # Reconstruct per-path cumulative factor (year 0 -> 1.0, year k ->
    # prod(1+r_1..r_k)). Mirrors the convention in
    # MonteCarloSimulator._build_inflation_factors_stochastic.
    ones_col = np.ones((n_mc, 1))
    cumulative_year_end = np.cumprod(1.0 + rates_paths, axis=1)
    cumulative_per_year = np.concatenate(
        [ones_col, cumulative_year_end[:, :-1]], axis=1
    )  # (n_mc, n_years)

    if n_mc <= sample_paths_limit:
        selected_idx = np.arange(n_mc)
    else:
        stride = max(1, n_mc // sample_paths_limit)
        selected_idx = np.arange(0, n_mc, stride)[:sample_paths_limit]

    sample_paths: List[List[float]] = [
        cumulative_per_year[i, :].tolist() for i in selected_idx
    ]

    return {
        "years": df_inf["year"].tolist(),
        "mean_factor": df_inf["mean_factor"].tolist(),
        "p05_factor": df_inf["p05_factor"].tolist(),
        "p95_factor": df_inf["p95_factor"].tolist(),
        "mean_rate": df_inf["mean_rate"].tolist(),
        "sample_paths": sample_paths,
    }


def _build_cashflow_table_payload(results: MonteCarloResults) -> Dict[str, Any]:
    """
    Build the monthly cash-flow table used by the Excel/PDF exporters.

    All series are *means across Monte Carlo paths* at monthly granularity.
    The table is structured as a "column store" (one key per column) for
    cheap JSON serialisation and for easy reshaping into a pandas
    DataFrame on the consumer side.

    Args:
        results: MonteCarloResults with the full path arrays still attached.

    Returns:
        Dict with keys:
            - ``months``: list[int], 0-based month indices.
            - ``mean_savings_eur``: mean nominal monthly savings (incl. bonus).
            - ``mean_savings_real_eur``: mean inflation-adjusted savings.
            - ``bonus_per_month_eur``: sparse bonus vector (0 where empty).
            - ``mean_profit_cum_eur``: mean cumulative profit, nominal.
            - ``mean_profit_cum_real_eur``: mean cumulative profit, real.
            - ``mean_price_eur_per_kwh``: mean electricity price per month.
            - ``mean_inflation_factor``: mean cumulative inflation factor.

    Notes:
        - The bonus is already folded into ``mean_savings_eur`` (the
          simulator adds it inside the per-month loop). The separate
          ``bonus_per_month_eur`` column is provided so the Excel reader
          can split it out and compute "savings excluding bonus" cheaply.
        - For deterministic-inflation runs the mean inflation factor is a
          simple ``(1+r)^year`` curve, identical across all paths.
    """
    n_months = len(results.df_profit)
    months_list = results.df_profit["month_index"].tolist()

    mean_savings_eur = results.monthly_savings_eur_paths.mean(axis=0).tolist()
    mean_savings_real_eur = results.monthly_savings_real_eur_paths.mean(axis=0).tolist()
    bonus_per_month_eur = (
        results.bonus_per_month_eur.tolist()
        if results.bonus_per_month_eur is not None
        else [0.0] * n_months
    )
    mean_profit_cum_eur = results.df_profit["mean_gain_eur"].tolist()
    mean_profit_cum_real_eur = results.df_profit["mean_gain_real_eur"].tolist()

    if results.df_price is not None:
        mean_price = results.df_price["price_mean_eur_per_kwh"].tolist()
    else:
        mean_price = [None] * n_months

    # Mean cumulative inflation factor at monthly granularity. For
    # stochastic runs we expand the per-year mean factor over 12 months.
    # For deterministic runs we recompute the (1+r)^year curve from
    # ``mean_savings_eur / mean_savings_real_eur`` to avoid storing yet
    # another array on MonteCarloResults — but this can produce NaNs when
    # the nominal savings are zero. Safer: derive from df_inflation when
    # available, else from results.monthly_savings_eur_paths /
    # monthly_savings_real_eur_paths (path-wise division before averaging).
    if results.df_inflation is not None:
        mean_factor_per_year = np.asarray(results.df_inflation["mean_factor"].tolist())
        mean_inflation_factor = np.repeat(mean_factor_per_year, 12).tolist()
    else:
        # Deterministic: derive the factor from the ratio of nominal to
        # real savings PATH-WISE then average. This is robust to zero
        # savings months because the simulator multiplies path-wise by
        # the same factor for nominal and real, so the ratio is the
        # constant deterministic factor for every (path, month).
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                results.monthly_savings_real_eur_paths != 0.0,
                results.monthly_savings_eur_paths
                / results.monthly_savings_real_eur_paths,
                np.nan,
            )
        # Use nanmean to ignore months where nominal=real=0 (no division).
        mean_inflation_factor = np.nanmean(ratio, axis=0)
        # Replace any all-nan column with 1.0 (start of horizon, no
        # inflation effect yet).
        mean_inflation_factor = np.where(
            np.isnan(mean_inflation_factor), 1.0, mean_inflation_factor
        ).tolist()

    return {
        "months": months_list,
        "mean_savings_eur": mean_savings_eur,
        "mean_savings_real_eur": mean_savings_real_eur,
        "bonus_per_month_eur": bonus_per_month_eur,
        "mean_profit_cum_eur": mean_profit_cum_eur,
        "mean_profit_cum_real_eur": mean_profit_cum_real_eur,
        "mean_price_eur_per_kwh": mean_price,
        "mean_inflation_factor": mean_inflation_factor,
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

    def _resolve_location_name(
        self, scenario_payload: Mapping[str, Any]
    ) -> str | None:
        """
        Best-effort lookup of the solar profile's location name.

        Used to enrich the persisted ``summary`` with a string the
        Dashboard can use as a filter facet. Returns None when no profile
        is referenced or the persistence layer is unavailable.
        """
        if not self.persistence:
            return None
        solar_cfg = scenario_payload.get("solar") or {}
        profile_id = solar_cfg.get("solar_profile_id")
        profile = None
        if profile_id is not None:
            try:
                profile = self.persistence.get_solar_profile_by_id(int(profile_id))
            except Exception:  # noqa: BLE001
                profile = None
        if profile is None:
            name = solar_cfg.get("solar_profile_name")
            if name:
                try:
                    profile = self.persistence.get_solar_profile_by_name(name)
                except Exception:  # noqa: BLE001
                    profile = None
        if profile is None:
            return None
        return getattr(profile, "location_name", None) or getattr(profile, "name", None)

    def run_analysis(
        self,
        *,
        n_mc: int | None = None,
        seed: int = 123,
        scenario_data: ScenarioData = None,
        progress_callback: "Callable[[int, int, float, float], None] | None" = None,
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
        solar_model = build_default_solar_model(scenario_payload, self.persistence)
        # Phase 16 — energy_cfg now also wires the optional ThermalModel /
        # ElectricalModel from the scenario JSON (resolved via the
        # persistence service when climate_profile_id is set).
        energy_cfg = build_default_energy_config(scenario_payload, self.persistence)
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

        results = mc.run(
            seed=seed,
            progress_callback=progress_callback,
            show_progress=progress_callback is None,
        )
        scenario_name = scenario_payload.get("scenario_name", "custom_scenario")

        price_plot = _build_price_plot_payload(results)
        # Phase 11 — optional inflation fan chart payload (None in
        # deterministic mode; the Dashboard tab degrades gracefully).
        inflation_plot = _build_inflation_plot_payload(results)
        # Phase 11 — monthly cash-flow table for Excel/PDF exporters.
        cashflow_table = _build_cashflow_table_payload(results)

        # Phase 12 — capture the solar profile location so the Dashboard
        # can offer a "Filter by location" facet without re-reading the
        # scenario JSON. Falls back to None when no profile is wired.
        location_name = self._resolve_location_name(scenario_payload)

        summary = {
            "scenario": scenario_name,
            "location_name": location_name,
            "final_gain_mean_eur": float(results.df_profit["mean_gain_eur"].iloc[-1]),
            "final_gain_real_mean_eur": float(results.df_profit["mean_gain_real_eur"].iloc[-1]),
            "prob_gain": float(results.df_profit["prob_gain"].iloc[-1]),
            # Phase 4 — break-even KPIs and aggregate financial metrics.
            "prob_break_even_within_horizon": results.prob_break_even_within_horizon,
            "break_even_month_median": results.break_even_month_median,
            "break_even_month_p05": results.break_even_month_p05,
            "break_even_month_p95": results.break_even_month_p95,
            "npv_median_eur": results.npv_median_eur,
            "irr_mean": results.irr_mean,
            # Phase 11 — total bonus disbursed. Always present (0.0 when
            # the feature is disabled) so the Dashboard can display it
            # unconditionally.
            "tax_bonus_total_eur": float(results.tax_bonus_total_eur),
            # Phase 16 — aggregated electrical KPIs (None when the
            # detailed model is disabled). The Dashboard reads this
            # block to render the new "Salute hardware" card with
            # hours_dc_overvoltage / hours_outside_mppt / peak voltages.
            "electrical": results.electrical_kpis_summary,
            "plots_data": {
                "profit": {
                    "months": results.df_profit["month_index"].tolist(),
                    "mean_gain_eur": results.df_profit["mean_gain_eur"].tolist(),
                    "p05_gain_eur": results.df_profit["p05_gain_eur"].tolist(),
                    "p95_gain_eur": results.df_profit["p95_gain_eur"].tolist(),
                    "mean_gain_real_eur": results.df_profit["mean_gain_real_eur"].tolist(),
                    # Phase 4 — break-even annotation data for the profit chart.
                    # The frontend uses these values to draw a dashed vertical
                    # line at the median and a shaded band between p05 and p95.
                    "break_even_month_median": results.break_even_month_median,
                    "break_even_month_p05": results.break_even_month_p05,
                    "break_even_month_p95": results.break_even_month_p95,
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
                # Phase 11 — inflation fan chart (None in deterministic mode).
                "inflation": inflation_plot,
                # Phase 11 — monthly cash-flow table used by the Excel/PDF
                # exporters (and any client that needs the raw monthly
                # means, e.g. a future "Tabella cassa" tab).
                "cashflow_table": cashflow_table,
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
            # Phase 6: capture the run record to expose run_id in the API
            # response so the Scenario Wizard can redirect directly to the
            # newly created run in the Dashboard.
            run_record = self.persistence.record_run_result(
                "analysis",
                summary,
                scenario=scenario_record,
                output_dir=str(output_dir) if output_dir else None,
            )
            summary["run_id"] = run_record.id if run_record is not None else None

        summary["output_dir"] = str(output_dir) if output_dir else None
        return summary

    def run_optimization(
        self,
        *,
        seed: int = 123,  # Default seed unified to 123 for consistency across workflows
        n_mc: int | None = None,
        scenario_data: ScenarioData = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
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
        base_energy_cfg = build_default_energy_config(scenario_payload, self.persistence)
        econ_cfg = build_default_economic_config(n_mc=n_mc, scenario_data=scenario_payload)
        price_model = build_default_price_model(scenario_data=scenario_payload)
        solar_model = build_default_solar_model(scenario_payload, self.persistence)

        # Phase 12 — resolved once for the whole sweep so each persisted
        # run inherits the same location label.
        opt_location_name = self._resolve_location_name(scenario_payload)

        optimizer = ScenarioOptimizer(
            request=request,
            base_energy_config=base_energy_cfg,
            economic_config_template=econ_cfg,
            price_model=price_model,
            load_profile_factory=lambda payload=scenario_payload: build_default_load_profile(payload),
            solar_model=solar_model,
        )
        evaluations = optimizer.run(
            seed=seed,
            external_progress_callback=progress_callback,
            show_progress=progress_callback is None,
        )

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
                # Phase 12 — copy the resolved location into the per-run
                # summary so the Dashboard can filter design results by
                # location as well.
                summary["location_name"] = opt_location_name
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
