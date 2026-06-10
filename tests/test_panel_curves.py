"""
Tests for the panel product-sheet curves: single-diode fit quality,
physical scaling across conditions and the API endpoint.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.persistence import PersistenceService
from sim_stochastic_pv.simulation.panel_curves import compute_panel_curve_families
from sim_stochastic_pv.simulation.pv_model import PVModelSingleDiode


# TCL HSM-ND54-DR505 — the designer's reference module.
REF = dict(
    isc_stc=15.88, voc_stc=40.14, imp_stc=14.9, vmp_stc=33.9,
    n_cells_series=108, alpha_isc_pct_per_c=0.045, beta_voc_pct_per_c=-0.25,
)


class TestSingleDiodeFit:
    def test_fit_reproduces_datasheet_points(self) -> None:
        """The 5-condition log-space fit passes through Isc, Voc and MPP
        within solver precision — this failed with the previous 3-point
        linear-space fit (0.3 A residuals on 15 A-class modules)."""
        m = PVModelSingleDiode(Isc=15.88, Voc=40.14, Imp=14.9, Vmp=33.9, Ns=108)
        m.solve()
        residuals = m.residuals([m.Iph, m.Is, m.n, m.Rs, m.Rsh])
        assert max(abs(r) for r in residuals) < 1e-5
        assert float(m.i_of_v(0.0)[0]) == pytest.approx(15.88, abs=1e-3)
        assert float(m.i_of_v(33.9)[0]) == pytest.approx(14.9, abs=1e-2)

    def test_nonsense_datasheet_raises(self) -> None:
        m = PVModelSingleDiode(Isc=1.0, Voc=40.0, Imp=5.0, Vmp=39.9, Ns=108)
        with pytest.raises(RuntimeError, match="did not converge"):
            m.solve()


class TestCurveFamilies:
    def test_stc_curve_matches_datasheet(self) -> None:
        irr, _ = compute_panel_curve_families(**REF)
        stc = irr[-1]  # 1000 W/m²
        assert stc.i[0] == pytest.approx(15.88, abs=0.02)
        assert stc.mpp_p == pytest.approx(505.0, rel=0.01)
        assert stc.mpp_v == pytest.approx(33.9, abs=0.5)
        # Open-circuit end of the grid.
        assert stc.v[-1] == pytest.approx(40.14, abs=1e-6)
        assert stc.i[-1] == pytest.approx(0.0, abs=0.05)
        # Currents decrease monotonically along the curve.
        assert all(b <= a + 1e-9 for a, b in zip(stc.i, stc.i[1:]))

    def test_irradiance_scales_current_linearly(self) -> None:
        irr, _ = compute_panel_curve_families(**REF)
        by_g = {c.irradiance_w_m2: c for c in irr}
        assert by_g[200.0].i[0] == pytest.approx(15.88 * 0.2, rel=0.01)
        assert by_g[600.0].i[0] == pytest.approx(15.88 * 0.6, rel=0.01)
        # Voc shrinks logarithmically with irradiance.
        assert by_g[200.0].v[-1] < by_g[1000.0].v[-1]

    def test_temperature_behaviour_consistent_with_gamma(self) -> None:
        """The power loss at 70 °C emerging from the diode physics must
        sit close to the datasheet γ(Pmax) estimate (within 2%) — the
        sheet's γ is −0.29 %/°C → P(70) ≈ 505·(1−0.0029·45) ≈ 439 W."""
        _, temp = compute_panel_curve_families(**REF)
        by_t = {c.t_cell_c: c for c in temp}
        gamma_estimate = 505.0 * (1.0 - 0.0029 * 45.0)
        assert by_t[70.0].mpp_p == pytest.approx(gamma_estimate, rel=0.02)
        # Cold panels produce more, hot panels less; Voc follows β.
        assert by_t[-10.0].mpp_p > by_t[25.0].mpp_p > by_t[70.0].mpp_p
        assert by_t[-10.0].v[-1] == pytest.approx(
            40.14 * (1 + 0.25 / 100 * 35), abs=0.05
        )

    def test_all_seeded_panels_converge(self) -> None:
        """Every shipped catalogue panel fits without errors."""
        catalogue = [
            dict(isc_stc=13.91, voc_stc=37.5, imp_stc=13.06, vmp_stc=31.4,
                 n_cells_series=108, alpha_isc_pct_per_c=0.05,
                 beta_voc_pct_per_c=-0.26),
            dict(isc_stc=13.93, voc_stc=49.75, imp_stc=13.02, vmp_stc=41.86,
                 n_cells_series=144, alpha_isc_pct_per_c=0.045,
                 beta_voc_pct_per_c=-0.272),
            dict(isc_stc=6.58, voc_stc=75.6, imp_stc=6.08, vmp_stc=65.8,
                 n_cells_series=104, alpha_isc_pct_per_c=0.05,
                 beta_voc_pct_per_c=-0.236),
        ]
        for spec in catalogue:
            irr, temp = compute_panel_curve_families(**spec)
            expected_p = spec["imp_stc"] * spec["vmp_stc"]
            assert irr[-1].mpp_p == pytest.approx(expected_p, rel=0.01)

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            compute_panel_curve_families(**{**REF, "isc_stc": -1.0})
        with pytest.raises(ValueError, match="inside"):
            compute_panel_curve_families(**{**REF, "imp_stc": 16.0})


class TestCurvesEndpoint:
    def _client(self, persistence: PersistenceService) -> TestClient:
        app = create_app()
        app.dependency_overrides[dependencies.get_persistence_service] = (
            lambda: persistence
        )
        return TestClient(app)

    def test_curves_for_complete_panel(self, persistence: PersistenceService) -> None:
        # upsert_panel stores the whole flat payload as the specs blob.
        record = persistence.upsert_panel({
            "name": "TCL test",
            "power_w": 505.0,
            "i_sc_stc_a": 15.88, "v_oc_stc_v": 40.14,
            "i_mpp_stc_a": 14.9, "v_mpp_stc_v": 33.9,
            "n_cells_series": 108,
            "alpha_isc_pct_per_c": 0.045, "beta_voc_pct_per_c": -0.25,
        })
        client = self._client(persistence)
        resp = client.get(f"/api/panels/{record.id}/curves")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["irradiance_family"]) == 5
        assert len(body["temperature_family"]) == 4
        stc = body["irradiance_family"][-1]
        assert stc["mpp_p"] == pytest.approx(505.0, rel=0.01)
        assert len(stc["v"]) == len(stc["i"]) == len(stc["p"])

    def test_incomplete_specs_named_in_422(
        self, persistence: PersistenceService
    ) -> None:
        record = persistence.upsert_panel({
            "name": "Vuoto", "power_w": 400.0, "specs": {"power_w": 400.0},
        })
        client = self._client(persistence)
        resp = client.get(f"/api/panels/{record.id}/curves")
        assert resp.status_code == 422
        assert "i_sc_stc_a" in resp.json()["detail"]

    def test_missing_panel_404(self, persistence: PersistenceService) -> None:
        client = self._client(persistence)
        assert client.get("/api/panels/999999/curves").status_code == 404
