"""
Tests for the background job API (Phase 12).

The submission endpoints return immediately with a job_id; the worker
runs in a ThreadPoolExecutor. We poll the status endpoint until the
job reaches a terminal state and then verify the persisted run_id.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.application import SimulationApplication
from sim_stochastic_pv.persistence import PersistenceService


def _client(persistence: PersistenceService) -> TestClient:
    app = create_app()
    app.dependency_overrides[dependencies.get_application_service] = (
        lambda: SimulationApplication(save_outputs=False, persistence=persistence)
    )
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence
    return TestClient(app)


def _wait_for_terminal(client: TestClient, job_id: str, timeout_s: float = 30.0) -> dict:
    """Poll the job endpoint until status is done or failed."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        if data["status"] in ("done", "failed"):
            return data
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not finish within {timeout_s}s")


class TestAnalysisJob:
    def test_submit_and_complete(
        self, persistence: PersistenceService, simple_scenario_data: dict
    ):
        client = _client(persistence)
        resp = client.post(
            "/api/jobs/analysis",
            json={"n_mc": 4, "seed": 1, "scenario": simple_scenario_data},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["kind"] == "analysis"
        job_id = body["job_id"]

        final = _wait_for_terminal(client, job_id)
        assert final["status"] == "done", final
        # The progress total must equal the requested n_mc (4) and done
        # must equal total at completion.
        assert final["progress_total"] == 4
        assert final["progress_done"] == 4
        # run_id must be set so the frontend can redirect.
        assert isinstance(final["run_id"], int)

    def test_404_on_unknown_job(self, persistence: PersistenceService):
        client = _client(persistence)
        resp = client.get("/api/jobs/does-not-exist")
        assert resp.status_code == 404


class TestOptimizationJob:
    def test_submit_and_complete(
        self, persistence: PersistenceService, simple_scenario_data: dict
    ):
        client = _client(persistence)
        resp = client.post(
            "/api/jobs/optimization",
            json={"n_mc": 2, "seed": 1, "scenario": simple_scenario_data},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        final = _wait_for_terminal(client, job_id, timeout_s=60.0)
        assert final["status"] == "done", final
        assert final["progress_total"] >= 1
        assert final["progress_done"] == final["progress_total"]


class TestJobListing:
    def test_list_recent_jobs(
        self, persistence: PersistenceService, simple_scenario_data: dict
    ):
        client = _client(persistence)
        client.post(
            "/api/jobs/analysis",
            json={"n_mc": 2, "seed": 1, "scenario": simple_scenario_data},
        )
        resp = client.get("/api/jobs")
        assert resp.status_code == 200
        jobs = resp.json()["jobs"]
        assert len(jobs) >= 1
        assert jobs[0]["kind"] in ("analysis", "optimization")
