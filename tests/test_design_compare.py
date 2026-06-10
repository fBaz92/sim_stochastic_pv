"""
Tests for the design comparator (Fase 25): paired Monte Carlo over
multiple plant designs with common random numbers, plus the background
job endpoint.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.application import SimulationApplication
from sim_stochastic_pv.persistence import PersistenceService

from conftest import _build_simple_scenario_data


def _make_app_service(persistence: PersistenceService) -> SimulationApplication:
    return SimulationApplication(
        save_outputs=False, persistence=persistence, result_builder=None
    )


def _seed_designs(persistence: PersistenceService) -> tuple[int, int]:
    """Two designs identical except for the turn-key cost (Δ = 2000 €)."""
    a = persistence.designs.upsert_design({
        "name": "Offerta A",
        "design_level": "essential",
        "data": {"p_ac_kw": 1.0, "p_dc_kwp": 1.0, "total_cost_eur": 3000.0},
    })
    b = persistence.designs.upsert_design({
        "name": "Offerta B",
        "design_level": "essential",
        "data": {"p_ac_kw": 1.0, "p_dc_kwp": 1.0, "total_cost_eur": 5000.0},
    })
    return a.id, b.id


def _base_scenario() -> dict:
    """Shared comparison context derived from the lightweight test scenario."""
    scenario = _build_simple_scenario_data()
    # The designs own the system + investment: strip them so the test
    # proves they come from each design.
    del scenario["energy"]["pv_kwp"]
    del scenario["energy"]["inverter_p_ac_max_kw"]
    del scenario["energy"]["battery_specs"]
    scenario["economic"]["investment_eur"] = 1.0
    return scenario


class TestRunDesignComparison:
    def test_cost_only_delta_is_deterministic(
        self, persistence: PersistenceService
    ) -> None:
        """Common random numbers: two designs differing only in cost must
        show a *deterministic* paired delta equal to the cost difference
        — identical weather/price paths cancel out exactly."""
        app_service = _make_app_service(persistence)
        id_a, id_b = _seed_designs(persistence)

        result = app_service.run_design_comparison(
            design_ids=[id_a, id_b],
            base_scenario=_base_scenario(),
            n_mc=8,
            seed=7,
        )

        assert [d["design_id"] for d in result["designs"]] == [id_a, id_b]
        assert result["baseline_design_id"] == id_a
        a, b = result["designs"]
        assert a["capex_eur"] == 3000.0
        assert b["capex_eur"] == 5000.0

        (delta,) = result["deltas"]
        assert delta["design_id"] == id_b
        assert delta["vs_design_id"] == id_a
        # B costs 2000 € more with identical energy → ΔNPV = −2000 on
        # every path: mean, p05 and p95 all collapse onto −2000.
        assert delta["delta_final_gain_mean_eur"] == pytest.approx(-2000.0, abs=1e-6)
        assert delta["delta_final_gain_p05_eur"] == pytest.approx(-2000.0, abs=1e-6)
        assert delta["delta_final_gain_p95_eur"] == pytest.approx(-2000.0, abs=1e-6)
        assert delta["prob_better"] == 0.0

    def test_bigger_system_changes_energy(
        self, persistence: PersistenceService
    ) -> None:
        """A design with twice the DC power must produce more energy."""
        app_service = _make_app_service(persistence)
        small = persistence.designs.upsert_design({
            "name": "Piccolo",
            "data": {"p_ac_kw": 1.0, "p_dc_kwp": 1.0, "total_cost_eur": 3000.0},
        })
        big = persistence.designs.upsert_design({
            "name": "Grande",
            "data": {"p_ac_kw": 2.0, "p_dc_kwp": 2.0, "total_cost_eur": 5500.0},
        })
        result = app_service.run_design_comparison(
            design_ids=[small.id, big.id],
            base_scenario=_base_scenario(),
            n_mc=6,
            seed=11,
        )
        s, b = result["designs"]
        assert b["p_dc_kwp"] == pytest.approx(2.0)
        assert b["annual_pv_kwh_mean"] > s["annual_pv_kwh_mean"] * 1.5

    def test_validation_errors(self, persistence: PersistenceService) -> None:
        app_service = _make_app_service(persistence)
        id_a, _ = _seed_designs(persistence)
        with pytest.raises(ValueError, match="2 to 4"):
            app_service.run_design_comparison(
                design_ids=[id_a],
                base_scenario=_base_scenario(),
                n_mc=5,
            )
        with pytest.raises(ValueError, match="not found"):
            app_service.run_design_comparison(
                design_ids=[id_a, 999999],
                base_scenario=_base_scenario(),
                n_mc=5,
            )


class TestCompareJobEndpoint:
    def _create_client(self, persistence: PersistenceService) -> TestClient:
        app = create_app()
        app.dependency_overrides[dependencies.get_persistence_service] = (
            lambda: persistence
        )
        app.dependency_overrides[dependencies.get_application_service] = (
            lambda: _make_app_service(persistence)
        )
        return TestClient(app)

    def test_compare_job_end_to_end(self, persistence: PersistenceService) -> None:
        """Submit → poll until done → inline result with designs+deltas.

        The shared context here is the endpoint's own default (inline
        ARERA load + default price), so this also covers the
        empty-database path of the page.
        """
        client = self._create_client(persistence)
        id_a, id_b = _seed_designs(persistence)

        submit = client.post("/api/jobs/compare", json={
            "design_ids": [id_a, id_b],
            "n_years": 2,
            "n_mc": 10,
            "seed": 5,
        })
        assert submit.status_code == 200, submit.text
        job_id = submit.json()["job_id"]
        assert submit.json()["kind"] == "comparison"

        deadline = time.time() + 120
        status = None
        while time.time() < deadline:
            snap = client.get(f"/api/jobs/{job_id}").json()
            status = snap["status"]
            if status in ("done", "failed"):
                break
            time.sleep(0.3)
        assert status == "done", snap.get("error")

        result = snap["result"]
        assert result is not None
        assert len(result["designs"]) == 2
        assert len(result["deltas"]) == 1
        assert result["deltas"][0]["delta_final_gain_mean_eur"] == pytest.approx(
            -2000.0, abs=1e-6
        )

    def test_compare_job_validates_count(
        self, persistence: PersistenceService
    ) -> None:
        client = self._create_client(persistence)
        resp = client.post("/api/jobs/compare", json={"design_ids": [1]})
        assert resp.status_code == 422
