"""
End-to-end tests for the bill-fit endpoints.

Covers ``POST /api/profiles/load/fit-bolletta`` (stateless auto-fit),
``GET /api/profiles/load/house-types`` (dropdown reference data) and the
existing inline preview applied to a bill-fitted profile.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.persistence import PersistenceService
from sim_stochastic_pv.simulation.load_profiles import ARERA_BASELINE_ANNUAL_KWH


@pytest.fixture()
def client(persistence: PersistenceService) -> TestClient:
    app = create_app()
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence
    return TestClient(app)


class TestFitBolletta:
    def test_valid_annual_fit(self, client: TestClient):
        resp = client.post(
            "/api/profiles/load/fit-bolletta",
            json={"annual_kwh": 2400.0, "house_type": "apartment_standard"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["home_scale_factor"] > 0.0
        # Energy split reconstructs the bill.
        assert body["estimated_home_kwh"] + body["estimated_away_kwh"] == pytest.approx(
            2400.0, rel=1e-3
        )
        data = body["derived_profile_data"]
        assert data["kind"] == "bolletta"
        assert data["input_level"] == "bolletta"
        assert data["bolletta"]["annual_kwh"] == 2400.0
        assert data["bolletta"]["house_type"] == "apartment_standard"
        assert len(body["min_days_home"]) == 12

    def test_bimonthly_is_summed(self, client: TestClient):
        resp = client.post(
            "/api/profiles/load/fit-bolletta",
            json={"bimonthly_kwh": [400, 350, 300, 280, 320, 450]},
        )
        assert resp.status_code == 200
        # 2100 kWh total → both energies present and summing to it.
        body = resp.json()
        assert body["estimated_home_kwh"] + body["estimated_away_kwh"] == pytest.approx(
            2100.0, rel=1e-3
        )

    def test_identity_fit_at_baseline(self, client: TestClient):
        resp = client.post(
            "/api/profiles/load/fit-bolletta",
            json={"annual_kwh": ARERA_BASELINE_ANNUAL_KWH},
        )
        assert resp.status_code == 200
        assert resp.json()["home_scale_factor"] == pytest.approx(1.0, rel=1e-3)

    def test_custom_presence_calendar_is_honoured(self, client: TestClient):
        """A weekend-only calendar lowers presence → higher home scale."""
        weekend = {
            "months": [
                {"weekends": True, "full_weeks": 0, "extra_weekdays": 0, "visit_probability": 0.0}
                for _ in range(12)
            ]
        }
        default = client.post(
            "/api/profiles/load/fit-bolletta", json={"annual_kwh": 3000.0}
        ).json()
        weekend_fit = client.post(
            "/api/profiles/load/fit-bolletta",
            json={"annual_kwh": 3000.0, "presence_calendar": weekend},
        ).json()
        assert weekend_fit["annual_presence_fraction"] < default["annual_presence_fraction"]
        assert weekend_fit["home_scale_factor"] > default["home_scale_factor"]

    def test_negative_kwh_is_422(self, client: TestClient):
        resp = client.post(
            "/api/profiles/load/fit-bolletta", json={"annual_kwh": -100.0}
        )
        assert resp.status_code == 422

    def test_missing_consumption_is_422(self, client: TestClient):
        resp = client.post("/api/profiles/load/fit-bolletta", json={})
        assert resp.status_code == 422

    def test_unknown_house_type_is_422(self, client: TestClient):
        resp = client.post(
            "/api/profiles/load/fit-bolletta",
            json={"annual_kwh": 2400.0, "house_type": "castle"},
        )
        assert resp.status_code == 422

    def test_bad_bimonthly_length_is_422(self, client: TestClient):
        resp = client.post(
            "/api/profiles/load/fit-bolletta",
            json={"bimonthly_kwh": [400, 350, 300]},
        )
        assert resp.status_code == 422


class TestHouseTypes:
    def test_lists_archetypes(self, client: TestClient):
        resp = client.get("/api/profiles/load/house-types")
        assert resp.status_code == 200
        items = resp.json()
        keys = {it["key"] for it in items}
        assert "apartment_standard" in keys
        for it in items:
            assert it["label_it"]
            assert it["floor_area_m2"] > 0
            assert it["baseline_annual_kwh"] > 0


class TestPreviewBollettaProfile:
    def test_preview_runs_for_bolletta_home(self, client: TestClient):
        """A fitted profile previews through the existing inline pipeline."""
        fit = client.post(
            "/api/profiles/load/fit-bolletta", json={"annual_kwh": 2400.0}
        ).json()
        resp = client.post(
            "/api/profiles/load/preview",
            json={
                "profile_type": "bolletta",
                "data": fit["derived_profile_data"],
                "month": 0,
                "regime": "home",
                "n_paths": 10,
                "seed": 1,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["total_kw_mean"]) == 168
        assert body["annual_kwh_mean"] > 0.0

    def test_preview_away_is_lower_than_home(self, client: TestClient):
        fit = client.post(
            "/api/profiles/load/fit-bolletta", json={"annual_kwh": 4000.0}
        ).json()
        common = {
            "profile_type": "bolletta",
            "data": fit["derived_profile_data"],
            "month": 0,
            "n_paths": 10,
            "seed": 1,
        }
        home = client.post(
            "/api/profiles/load/preview", json={**common, "regime": "home"}
        ).json()
        away = client.post(
            "/api/profiles/load/preview", json={**common, "regime": "away"}
        ).json()
        assert away["annual_kwh_mean"] < home["annual_kwh_mean"]
