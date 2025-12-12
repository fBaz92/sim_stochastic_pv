from __future__ import annotations

from sim_stochastic_pv.application import SimulationApplication
from sim_stochastic_pv.persistence import PersistenceService


def test_simulation_application_run_analysis_records_run(
    persistence: PersistenceService,
    simple_scenario_data: dict,
):
    """Run analysis workflow and assert a DB record is inserted."""
    app = SimulationApplication(
        save_outputs=False,
        persistence=persistence,
        result_builder=None,
    )
    summary = app.run_analysis(n_mc=2, seed=1, scenario_data=simple_scenario_data)
    assert "scenario" in summary
    runs = persistence.list_run_results(limit=5)
    assert runs, "Expected a run result stored in the database"


def test_simulation_application_run_optimization(
    persistence: PersistenceService,
    simple_scenario_data: dict,
):
    """Run optimization workflow and ensure results are stored."""
    app = SimulationApplication(
        save_outputs=False,
        persistence=persistence,
        result_builder=None,
    )
    summary = app.run_optimization(seed=1, n_mc=1, scenario_data=simple_scenario_data)
    assert summary["evaluations"] > 0
    runs = persistence.list_run_results(limit=10)
    assert runs, "Expected optimization runs stored"
