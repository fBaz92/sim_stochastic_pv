from __future__ import annotations

from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.application import SimulationApplication
from sim_stochastic_pv.persistence import PersistenceService


def create_test_client(persistence: PersistenceService) -> TestClient:
    """Build a FastAPI test client with dependency overrides for persistence."""
    app = create_app()

    def get_app_service() -> SimulationApplication:
        return SimulationApplication(
            save_outputs=False,
            persistence=persistence,
            result_builder=None,
        )

    app.dependency_overrides[dependencies.get_application_service] = get_app_service
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence
    return TestClient(app)


def test_api_analysis_and_runs(persistence: PersistenceService):
    """Exercise /api/analysis and /api/runs endpoints."""
    client = create_test_client(persistence)
    resp = client.post("/api/analysis", json={"n_mc": 1, "seed": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["scenario"]

    runs_resp = client.get("/api/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()
    assert len(runs) >= 1


def test_api_optimization(persistence: PersistenceService):
    """Exercise the /api/optimization endpoint."""
    client = create_test_client(persistence)
    resp = client.post("/api/optimization", json={"seed": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert data["evaluations"] > 0
