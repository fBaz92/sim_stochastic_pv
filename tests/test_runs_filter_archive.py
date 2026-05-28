"""
Tests for the Phase 12 Dashboard helpers:

- ``GET /api/runs`` with filters (name, location, date range, archive toggle).
- ``GET /api/runs/locations``.
- ``PATCH /api/runs/{id}/archive`` / ``/unarchive``.
- ``DELETE /api/runs/{id}``.
"""

from __future__ import annotations

from datetime import datetime, timezone

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


def _seed_runs(persistence: PersistenceService) -> dict[str, int]:
    """Insert a handful of runs with different scenarios/locations."""
    a = persistence.record_run_result(
        "analysis",
        {"scenario": "Casa Milano 3kWp", "location_name": "Milano", "prob_gain": 0.7},
    )
    b = persistence.record_run_result(
        "analysis",
        {"scenario": "Casa Roma 5kWp", "location_name": "Roma", "prob_gain": 0.6},
    )
    c = persistence.record_run_result(
        "optimization",
        {"scenario": "Design Pavullo", "location_name": "Pavullo", "evaluations": 12},
    )
    return {"milano": a.id, "roma": b.id, "pavullo": c.id}


class TestFilters:
    def test_no_filter_returns_all_active(self, persistence: PersistenceService):
        _seed_runs(persistence)
        resp = _client(persistence).get("/api/runs")
        assert resp.status_code == 200
        ids = [r["id"] for r in resp.json()]
        assert len(ids) == 3

    def test_filter_by_scenario_substring(self, persistence: PersistenceService):
        ids = _seed_runs(persistence)
        resp = _client(persistence).get("/api/runs", params={"scenario_name": "milano"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["id"] == ids["milano"]

    def test_filter_by_location(self, persistence: PersistenceService):
        ids = _seed_runs(persistence)
        resp = _client(persistence).get("/api/runs", params={"location": "Roma"})
        assert resp.status_code == 200
        body = resp.json()
        assert len(body) == 1
        assert body[0]["id"] == ids["roma"]

    def test_limit_and_offset_paginate(self, persistence: PersistenceService):
        _seed_runs(persistence)
        page1 = _client(persistence).get("/api/runs", params={"limit": 2}).json()
        page2 = _client(persistence).get("/api/runs", params={"limit": 2, "offset": 2}).json()
        assert len(page1) == 2
        assert len(page2) == 1


class TestLocations:
    def test_distinct_locations(self, persistence: PersistenceService):
        _seed_runs(persistence)
        resp = _client(persistence).get("/api/runs/locations")
        assert resp.status_code == 200
        locs = resp.json()
        assert set(locs) == {"Milano", "Roma", "Pavullo"}


class TestArchiveAndDelete:
    def test_archive_hides_by_default_unhides_with_flag(
        self, persistence: PersistenceService
    ):
        ids = _seed_runs(persistence)
        client = _client(persistence)

        resp = client.patch(f"/api/runs/{ids['milano']}/archive")
        assert resp.status_code == 200
        assert resp.json()["archived_at"] is not None

        # Default list hides it.
        active = client.get("/api/runs").json()
        assert all(r["id"] != ids["milano"] for r in active)

        # include_archived=true brings it back.
        all_runs = client.get("/api/runs", params={"include_archived": True}).json()
        assert any(r["id"] == ids["milano"] for r in all_runs)

        # Unarchive.
        resp = client.patch(f"/api/runs/{ids['milano']}/unarchive")
        assert resp.status_code == 200
        assert resp.json()["archived_at"] is None
        active_again = client.get("/api/runs").json()
        assert any(r["id"] == ids["milano"] for r in active_again)

    def test_delete_removes_row(self, persistence: PersistenceService):
        ids = _seed_runs(persistence)
        client = _client(persistence)
        resp = client.delete(f"/api/runs/{ids['roma']}")
        assert resp.status_code == 204
        after = client.get("/api/runs").json()
        assert all(r["id"] != ids["roma"] for r in after)

    def test_archive_unknown_id_404(self, persistence: PersistenceService):
        client = _client(persistence)
        assert client.patch("/api/runs/9999/archive").status_code == 404
        assert client.delete("/api/runs/9999").status_code == 404
