"""
Tests for the technical-report PDF endpoint
(``GET /api/designs/{id}/report.pdf``).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.persistence import PersistenceService


def _client(persistence: PersistenceService) -> TestClient:
    app = create_app()
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence
    return TestClient(app)


def _detailed_design(persistence: PersistenceService) -> int:
    """A detailed design with the full designer block (reference case)."""
    record = persistence.designs.upsert_design({
        "name": "Progetto relazione",
        "design_level": "detailed",
        "data": {
            "p_ac_kw": 3.0,
            "p_dc_kwp": 6.06,
            "total_cost_eur": 9000.0,
            "designer": {
                "panel": {
                    "power_w": 505.0, "v_oc_stc_v": 40.14, "v_mpp_stc_v": 33.9,
                    "i_sc_stc_a": 15.88, "i_mpp_stc_a": 14.9,
                    "beta_voc_pct_per_c": -0.25, "gamma_pmax_pct_per_c": -0.29,
                    "alpha_isc_pct_per_c": 0.045, "v_system_max_v": 1500.0,
                    "max_series_fuse_a": 30.0, "noct_c": 45.0,
                    "n_cells_series": 108,
                },
                "inverter": {
                    "p_ac_nom_kw": 3.0, "efficiency_max": 0.972,
                    "v_dc_max_v": 600.0, "v_dc_min_v": 90.0,
                    "v_mppt_min_v": 90.0, "v_mppt_max_v": 580.0,
                    "v_mppt_full_load_min_v": 160.0,
                    "v_mppt_full_load_max_v": 520.0,
                    "n_mppt_trackers": 2, "i_dc_max_per_mppt_a": 12.0,
                    "i_sc_max_per_mppt_a": 15.0, "max_strings_per_mppt": 1,
                },
                "site": {"t_min_c": -20.0, "t_max_c": 45.0, "delta_t_cell_c": 30.0},
                "requirements": {
                    "p_ac_required_kw": 3.0, "target_dc_ac_ratio": 1.2,
                    "n_panels_per_string": 6, "safety_factor_isc": 1.25,
                    "max_cable_loss_fraction": 0.005,
                    "fuse_factor_min": 1.5, "fuse_factor_max": 2.4,
                },
                "n_panels_per_string": 6,
                "n_strings": 2,
                "cable_section_mm2": 10.0,
                "cable_length_one_way_m": 30.0,
                "cable_operating_temperature_c": 70.0,
                "all_checks_ok": False,
            },
        },
    })
    return record.id


def test_report_pdf_renders(persistence: PersistenceService) -> None:
    """A detailed design renders a non-trivial PDF document."""
    client = _client(persistence)
    design_id = _detailed_design(persistence)

    resp = client.get(f"/api/designs/{design_id}/report.pdf")
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"] == "application/pdf"
    assert "relazione_" in resp.headers["content-disposition"]
    assert resp.content.startswith(b"%PDF")
    # Multi-section dossier: well above a trivial one-pager blob.
    assert len(resp.content) > 10_000


def test_report_requires_designer_block(persistence: PersistenceService) -> None:
    """An essential offer has nothing to size: explicit 422."""
    record = persistence.designs.upsert_design({
        "name": "Solo offerta",
        "design_level": "essential",
        "data": {"p_ac_kw": 6.0, "total_cost_eur": 14000.0},
    })
    client = _client(persistence)
    resp = client.get(f"/api/designs/{record.id}/report.pdf")
    assert resp.status_code == 422
    assert "Progettazione" in resp.json()["detail"]


def test_report_missing_design_404(persistence: PersistenceService) -> None:
    client = _client(persistence)
    assert client.get("/api/designs/999999/report.pdf").status_code == 404
