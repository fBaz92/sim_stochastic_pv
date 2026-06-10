"""
Tests for plant designs ("Impianti"): repository CRUD, the
``plant_design_id`` hydration resolver, the ``/api/designs`` endpoints,
and an end-to-end analysis run driven by an essential design.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.persistence import PersistenceService
from sim_stochastic_pv.persistence.hydration import apply_plant_design

from conftest import _build_simple_scenario_data


def _create_test_client(persistence: PersistenceService) -> TestClient:
    app = create_app()
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence

    from sim_stochastic_pv.application import SimulationApplication

    app.dependency_overrides[dependencies.get_application_service] = (
        lambda: SimulationApplication(
            save_outputs=False, persistence=persistence, result_builder=None
        )
    )
    return TestClient(app)


def _essential_payload(**overrides):
    base = {
        "name": "Offerta Rossi 6kW",
        "design_level": "essential",
        "description": "Offerta chiavi in mano ricevuta a giugno",
        "data": {
            "p_ac_kw": 6.0,
            "p_dc_kwp": 6.6,
            "storage_kwh": 10.0,
            "total_cost_eur": 14500.0,
            "tax_bonus": {
                "enabled": True,
                "fraction_of_investment": 0.5,
                "duration_years": 10,
            },
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class TestRepository:
    def test_upsert_and_get(self, persistence: PersistenceService) -> None:
        record = persistence.designs.upsert_design({
            "name": "Offerta A",
            "design_level": "essential",
            "data": {"p_ac_kw": 3.0, "total_cost_eur": 7000.0},
        })
        assert record.id is not None
        assert persistence.designs.get_design_by_name("Offerta A").id == record.id
        # Upsert by name updates in place.
        again = persistence.designs.upsert_design({
            "name": "Offerta A",
            "data": {"p_ac_kw": 4.0, "total_cost_eur": 8000.0},
        })
        assert again.id == record.id
        assert again.data["p_ac_kw"] == 4.0
        assert len(persistence.designs.list_designs()) == 1

    def test_rename_clash_raises(self, persistence: PersistenceService) -> None:
        a = persistence.designs.upsert_design({
            "name": "A", "data": {"p_ac_kw": 3.0, "total_cost_eur": 1.0},
        })
        persistence.designs.upsert_design({
            "name": "B", "data": {"p_ac_kw": 3.0, "total_cost_eur": 1.0},
        })
        with pytest.raises(ValueError):
            persistence.designs.update_design(a.id, {"name": "B"})

    def test_delete(self, persistence: PersistenceService) -> None:
        record = persistence.designs.upsert_design({
            "name": "Del", "data": {"p_ac_kw": 3.0, "total_cost_eur": 1.0},
        })
        assert persistence.designs.delete_design(record.id) is True
        assert persistence.designs.delete_design(record.id) is False


# ---------------------------------------------------------------------------
# Hydration resolver
# ---------------------------------------------------------------------------


class TestApplyPlantDesign:
    def _design(self, persistence, **data_overrides):
        data = {
            "p_ac_kw": 6.0,
            "p_dc_kwp": 6.6,
            "storage_kwh": 10.0,
            "total_cost_eur": 14500.0,
            "tax_bonus": {
                "enabled": True,
                "fraction_of_investment": 0.5,
                "duration_years": 10,
            },
        }
        data.update(data_overrides)
        # Drop keys explicitly set to None so "absent" cases are testable.
        data = {k: v for k, v in data.items() if v is not None}
        return persistence.designs.upsert_design({
            "name": "Hydration test", "design_level": "essential", "data": data,
        })

    def test_design_fields_overwrite_scenario(
        self, persistence: PersistenceService
    ) -> None:
        design = self._design(persistence)
        scenario = {
            "plant_design_id": design.id,
            "energy": {"n_years": 7, "pv_kwp": 1.0, "inverter_p_ac_max_kw": 1.0},
            "economic": {"investment_eur": 1.0, "n_mc": 42},
        }
        with persistence.session() as session:
            out = apply_plant_design(scenario, session)
        # Design wins on its own fields…
        assert out["energy"]["pv_kwp"] == 6.6
        assert out["energy"]["inverter_p_ac_max_kw"] == 6.0
        assert out["energy"]["battery_specs"]["capacity_kwh"] == 10.0
        assert out["energy"]["n_batteries"] == 1
        assert out["solar"]["pv_kwp"] == 6.6
        assert out["economic"]["investment_eur"] == 14500.0
        assert out["economic"]["tax_bonus"]["fraction_of_investment"] == 0.5
        # …while scenario-owned fields survive untouched.
        assert out["energy"]["n_years"] == 7
        assert out["economic"]["n_mc"] == 42

    def test_dc_defaults_to_ac_and_no_storage(
        self, persistence: PersistenceService
    ) -> None:
        design = self._design(
            persistence, p_dc_kwp=None, storage_kwh=None, tax_bonus=None
        )
        with persistence.session() as session:
            out = apply_plant_design({"plant_design_id": design.id}, session)
        assert out["energy"]["pv_kwp"] == 6.0
        assert out["energy"]["battery_specs"]["capacity_kwh"] == 0.0
        assert out["energy"]["n_batteries"] == 0
        assert "tax_bonus" not in out["economic"]

    def test_location_inheritance(self, persistence: PersistenceService) -> None:
        """A design anchored to a site inherits its solar+climate profiles."""
        location, solar, climate = persistence.locations.persist_import(
            {"name": "Sito", "latitude": 44.3, "longitude": 10.8},
            solar_data={
                "name": "Sito",
                "location_name": "Sito",
                "latitude": 44.3,
                "longitude": 10.8,
                "optimal_tilt_degrees": 35.0,
                "optimal_azimuth_degrees": 180.0,
                "avg_daily_kwh_per_kwp": [3.0] * 12,
                "p_sunny": [0.5] * 12,
            },
            climate_data={
                "name": "Sito",
                "location_name": "Sito",
                "latitude": 44.3,
                "longitude": 10.8,
                "harmonic": {"a0": 12.0, "a1": -10.0, "a2": 0.0},
                "monthly_params": [
                    {
                        "t_std_residual_c": 2.0,
                        "persistence_phi": 0.7,
                        "t_amplitude_c": 5.0,
                        "gpd_upper": None,
                        "gpd_lower": None,
                    }
                ] * 12,
                "climate_trend_c_per_year": 0.0,
            },
        )
        design = persistence.designs.upsert_design({
            "name": "Con sito",
            "design_level": "essential",
            "data": {"p_ac_kw": 3.0, "total_cost_eur": 7000.0},
            "location_id": location.id,
        })
        with persistence.session() as session:
            out = apply_plant_design({"plant_design_id": design.id}, session)
        assert out["solar"]["solar_profile_id"] == solar.id
        assert out["climate_profile_id"] == climate.id

    def test_unknown_id_raises(self, persistence: PersistenceService) -> None:
        with persistence.session() as session:
            with pytest.raises(ValueError, match="not found"):
                apply_plant_design({"plant_design_id": 999999}, session)

    def test_detailed_level_rejected_for_now(
        self, persistence: PersistenceService
    ) -> None:
        record = persistence.designs.upsert_design({
            "name": "Detailed",
            "design_level": "detailed",
            "data": {"p_ac_kw": 3.0, "total_cost_eur": 1.0},
        })
        with persistence.session() as session:
            with pytest.raises(ValueError, match="detailed"):
                apply_plant_design({"plant_design_id": record.id}, session)

    def test_noop_without_reference(self, persistence: PersistenceService) -> None:
        scenario = {"energy": {"pv_kwp": 1.0}}
        with persistence.session() as session:
            assert apply_plant_design(scenario, session) is scenario


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestEndpoints:
    def test_crud_roundtrip(self, persistence: PersistenceService) -> None:
        client = _create_test_client(persistence)

        create = client.post("/api/designs", json=_essential_payload())
        assert create.status_code == 200, create.text
        body = create.json()
        assert body["design_level"] == "essential"
        assert body["data"]["p_ac_kw"] == 6.0

        listing = client.get("/api/designs")
        assert [d["name"] for d in listing.json()] == ["Offerta Rossi 6kW"]

        renamed = client.put(
            f"/api/designs/{body['id']}", json={"name": "Offerta Rossi v2"}
        )
        assert renamed.status_code == 200
        assert renamed.json()["name"] == "Offerta Rossi v2"
        # The data payload survives a rename-only update.
        assert renamed.json()["data"]["total_cost_eur"] == 14500.0

        assert client.delete(f"/api/designs/{body['id']}").status_code == 204
        assert client.delete(f"/api/designs/{body['id']}").status_code == 404

    def test_upsert_by_name(self, persistence: PersistenceService) -> None:
        client = _create_test_client(persistence)
        first = client.post("/api/designs", json=_essential_payload())
        payload = _essential_payload()
        payload["data"]["total_cost_eur"] = 13000.0
        second = client.post("/api/designs", json=payload)
        assert second.json()["id"] == first.json()["id"]
        assert second.json()["data"]["total_cost_eur"] == 13000.0

    def test_validation_rejects_suspicious_dc(
        self, persistence: PersistenceService
    ) -> None:
        client = _create_test_client(persistence)
        payload = _essential_payload()
        payload["data"]["p_dc_kwp"] = 1.0  # < half of 6 kW AC → typo guard
        assert client.post("/api/designs", json=payload).status_code == 422

    def test_validation_requires_cost(self, persistence: PersistenceService) -> None:
        client = _create_test_client(persistence)
        payload = _essential_payload()
        del payload["data"]["total_cost_eur"]
        assert client.post("/api/designs", json=payload).status_code == 422

    def test_rename_conflict_409(self, persistence: PersistenceService) -> None:
        client = _create_test_client(persistence)
        a = client.post("/api/designs", json=_essential_payload(name="A")).json()
        client.post("/api/designs", json=_essential_payload(name="B"))
        assert client.put(f"/api/designs/{a['id']}", json={"name": "B"}).status_code == 409


# ---------------------------------------------------------------------------
# End-to-end: analysis driven by a design
# ---------------------------------------------------------------------------


def test_analysis_run_from_design(persistence: PersistenceService) -> None:
    """A scenario referencing a design runs the MC with the design's
    system + investment, proving the full offer flow end to end."""
    client = _create_test_client(persistence)
    design = client.post(
        "/api/designs",
        json=_essential_payload(name="Offerta e2e"),
    ).json()

    scenario = _build_simple_scenario_data()
    # The design owns system + investment: drop them from the scenario to
    # prove they come from the design, keep load/price/horizon/MC.
    del scenario["energy"]["pv_kwp"]
    del scenario["energy"]["inverter_p_ac_max_kw"]
    del scenario["energy"]["battery_specs"]
    scenario["economic"]["investment_eur"] = 1.0  # will be overwritten
    scenario["plant_design_id"] = design["id"]

    resp = client.post(
        "/api/analysis", json={"n_mc": 5, "seed": 1, "scenario": scenario}
    )
    assert resp.status_code == 200, resp.text
    summary = resp.json()
    # The tax bonus disbursed within the 2-year horizon is 2/10 of 50% of
    # the *design's* investment (14500 €): 14500 × 0.5 × 2/10 = 1450 —
    # proving the design overwrote the scenario's 1 € placeholder.
    assert summary["tax_bonus_total_eur"] == pytest.approx(1450.0)
    # And the run produced sane KPIs over the 2-year horizon.
    assert "final_gain_mean_eur" in summary
    assert 0.0 <= summary["prob_gain"] <= 1.0
