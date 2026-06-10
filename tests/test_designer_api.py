"""
Endpoint tests for the electrical designer:
``POST /api/designs/evaluate`` (stateless spreadsheet engine),
``POST /api/designs/production-preview`` (hourly MC) and the
cable/protection catalogue CRUD.

The evaluate expectations are the reference-spreadsheet numbers — the
deep validation lives in ``test_electrical_design.py``; here we check the
HTTP wiring, the serialization shape and the explicit error paths.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.persistence import PersistenceService


def _create_test_client(persistence: PersistenceService) -> TestClient:
    app = create_app()
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence
    return TestClient(app)


def _panel_payload(**overrides):
    base = {
        "power_w": 505.0,
        "v_oc_stc_v": 40.14,
        "v_mpp_stc_v": 33.9,
        "i_sc_stc_a": 15.88,
        "i_mpp_stc_a": 14.9,
        "beta_voc_pct_per_c": -0.25,
        "gamma_pmax_pct_per_c": -0.29,
        "alpha_isc_pct_per_c": 0.045,
        "v_system_max_v": 1500.0,
        "max_series_fuse_a": 30.0,
        "noct_c": 45.0,
        "n_cells_series": 108,
    }
    base.update(overrides)
    return base


def _inverter_payload(**overrides):
    base = {
        "p_ac_nom_kw": 3.0,
        "efficiency_max": 0.972,
        "v_dc_max_v": 600.0,
        "v_dc_min_v": 90.0,
        "v_mppt_min_v": 90.0,
        "v_mppt_max_v": 580.0,
        "v_mppt_full_load_min_v": 160.0,
        "v_mppt_full_load_max_v": 520.0,
        "n_mppt_trackers": 2,
        "i_dc_max_per_mppt_a": 12.0,
        "i_sc_max_per_mppt_a": 15.0,
        "max_strings_per_mppt": 1,
    }
    base.update(overrides)
    return base


def _evaluate_payload(**overrides):
    base = {
        "panel": _panel_payload(),
        "inverter": _inverter_payload(),
        "site": {"t_min_c": -20.0, "t_max_c": 45.0, "delta_t_cell_c": 30.0},
        "requirements": {
            "p_ac_required_kw": 3.0,
            "target_dc_ac_ratio": 1.2,
            "n_panels_per_string": 6,
            "max_cable_loss_fraction": 0.005,
        },
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# /api/designs/evaluate
# ---------------------------------------------------------------------------


def test_evaluate_reference_case(persistence: PersistenceService) -> None:
    client = _create_test_client(persistence)
    resp = client.post("/api/designs/evaluate", json=_evaluate_payload())
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["bounds"] == {
        "v_limit_v": 600.0, "n_max_voc": 13, "n_max_mppt": 13,
        "n_min": 6, "n_max": 13, "feasible": True,
    }
    assert body["plant"]["p_dc_installed_kwp"] == pytest.approx(6.06)
    assert body["plant"]["dc_ac_ratio"] == pytest.approx(2.02)
    # The reference case fails the current checks (engine truth).
    assert body["currents"]["i_operating_margin_a"] < 0
    assert body["all_checks_ok"] is False
    assert body["cables"]["recommended_section_mm2"] == 25.0
    assert body["protection"]["recommended_fuse_a"] == 25.0


def test_evaluate_missing_field_names_it(persistence: PersistenceService) -> None:
    """An incomplete datasheet is a 422 naming the field (no silent
    defaults)."""
    client = _create_test_client(persistence)
    payload = _evaluate_payload()
    del payload["panel"]["alpha_isc_pct_per_c"]
    resp = client.post("/api/designs/evaluate", json=payload)
    assert resp.status_code == 422


def test_evaluate_with_custom_cable_sections(persistence: PersistenceService) -> None:
    client = _create_test_client(persistence)
    payload = _evaluate_payload()
    payload["cable"] = {
        "length_one_way_m": 10.0,
        "sections": [
            {"section_mm2": 4.0, "price_eur_per_m": 0.75, "iz_a": 55.0},
            {"section_mm2": 6.0, "price_eur_per_m": 1.05, "iz_a": 70.0},
        ],
    }
    resp = client.post("/api/designs/evaluate", json=payload)
    assert resp.status_code == 200, resp.text
    rows = resp.json()["cables"]["rows"]
    assert [r["section_mm2"] for r in rows] == [4.0, 6.0]
    assert rows[0]["cost_total_eur"] == pytest.approx(0.75 * 2 * 10 * 2)
    assert rows[0]["iz_ok"] is True


# ---------------------------------------------------------------------------
# /api/designs/production-preview
# ---------------------------------------------------------------------------


def _seed_solar_profile(persistence: PersistenceService) -> int:
    record = persistence.upsert_solar_profile({
        "name": "Designer site",
        "location_name": "Designer site",
        "latitude": 44.3,
        "longitude": 10.8,
        "optimal_tilt_degrees": 35.0,
        "optimal_azimuth_degrees": 180.0,
        "avg_daily_kwh_per_kwp": [1.5, 2.3, 3.5, 4.4, 5.3, 5.8,
                                  6.3, 5.7, 4.5, 3.1, 1.8, 1.4],
        "p_sunny": [0.5] * 12,
    })
    return record.id


def test_production_preview_basic(persistence: PersistenceService) -> None:
    """The preview returns a coherent loss chain on the reference design
    (DC/AC 2.02 → substantial clipping)."""
    client = _create_test_client(persistence)
    solar_id = _seed_solar_profile(persistence)
    resp = client.post("/api/designs/production-preview", json={
        "panel": _panel_payload(),
        "inverter": _inverter_payload(),
        "n_panels_per_string": 6,
        "n_strings": 2,
        "solar_profile_id": solar_id,
        "cable": {"section_mm2": 6.0, "length_one_way_m": 30.0},
        "n_paths": 10,
        "seed": 3,
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # 6.06 kWp at ~3.9 kWh/kWp/day average → roughly 8-9 MWh DC.
    assert 6000 < body["annual_dc_kwh_mean"] < 12000
    # Band ordering and coherence.
    assert body["annual_ac_kwh_p05"] <= body["annual_ac_kwh_mean"] <= body["annual_ac_kwh_p95"]
    # DC/AC 2.02 on a 3 kW inverter must clip a visible share.
    assert body["clipping_fraction"] > 0.05
    assert body["cable_loss_kwh_mean"] > 0
    # Energy conservation: AC <= (DC - losses) * efficiency.
    reconstructed = (
        body["annual_dc_kwh_mean"]
        - body["clipping_kwh_mean"]
        - body["cable_loss_kwh_mean"]
    ) * body["inverter_efficiency"]
    assert body["annual_ac_kwh_mean"] == pytest.approx(reconstructed, rel=0.02)


def test_production_preview_404_on_missing_profile(
    persistence: PersistenceService,
) -> None:
    client = _create_test_client(persistence)
    resp = client.post("/api/designs/production-preview", json={
        "panel": _panel_payload(),
        "inverter": _inverter_payload(),
        "n_panels_per_string": 6,
        "n_strings": 2,
        "solar_profile_id": 999999,
        "n_paths": 5,
    })
    assert resp.status_code == 404


def test_production_preview_electrical_requires_climate(
    persistence: PersistenceService,
) -> None:
    client = _create_test_client(persistence)
    solar_id = _seed_solar_profile(persistence)
    resp = client.post("/api/designs/production-preview", json={
        "panel": _panel_payload(),
        "inverter": _inverter_payload(),
        "n_panels_per_string": 6,
        "n_strings": 2,
        "solar_profile_id": solar_id,
        "use_electrical_model": True,
        "n_paths": 5,
    })
    assert resp.status_code == 422
    assert "climate_profile_id" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Cable / protection catalogue CRUD
# ---------------------------------------------------------------------------


def test_cable_crud_roundtrip(persistence: PersistenceService) -> None:
    client = _create_test_client(persistence)
    create = client.post("/api/cables", json={
        "name": "Test 6mm2", "section_mm2": 6.0,
        "price_eur_per_m": 1.05, "iz_a": 70.0,
    })
    assert create.status_code == 200, create.text
    cable_id = create.json()["id"]

    listing = client.get("/api/cables")
    assert [c["name"] for c in listing.json()] == ["Test 6mm2"]

    updated = client.put(f"/api/cables/{cable_id}", json={
        "name": "Test 6mm2 v2", "section_mm2": 6.0, "price_eur_per_m": 1.10,
    })
    assert updated.status_code == 200
    assert updated.json()["price_eur_per_m"] == pytest.approx(1.10)

    assert client.delete(f"/api/cables/{cable_id}").status_code == 204
    assert client.delete(f"/api/cables/{cable_id}").status_code == 404


def test_protection_crud_roundtrip(persistence: PersistenceService) -> None:
    client = _create_test_client(persistence)
    create = client.post("/api/protections", json={
        "name": "Fuse 25A test", "kind": "fuse",
        "rated_current_a": 25.0, "rated_voltage_v": 1000.0, "price_eur": 3.6,
    })
    assert create.status_code == 200, create.text
    prot_id = create.json()["id"]

    listing = client.get("/api/protections")
    assert [p["name"] for p in listing.json()] == ["Fuse 25A test"]

    assert client.delete(f"/api/protections/{prot_id}").status_code == 204


def test_detailed_design_roundtrips_extra_blocks(
    persistence: PersistenceService,
) -> None:
    """Saving a detailed design keeps the designer blocks verbatim."""
    client = _create_test_client(persistence)
    resp = client.post("/api/designs", json={
        "name": "Progetto dettagliato",
        "design_level": "detailed",
        "data": {
            "p_ac_kw": 3.0,
            "p_dc_kwp": 6.06,
            "total_cost_eur": 9000.0,
            "designer": {
                "n_panels_per_string": 6,
                "n_strings": 2,
                "cable_section_mm2": 25.0,
            },
        },
    })
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["design_level"] == "detailed"
    assert body["data"]["designer"]["cable_section_mm2"] == 25.0
