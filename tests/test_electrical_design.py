"""
Tests for the electrical designer engine, validated against the
reference sizing spreadsheet ("Dimensionamento_FV.xlsx").

The reference case is the spreadsheet's own example: TCL HSM-ND54-DR505
module on a ZCS 1PH 3000-TLM-V3 inverter at a site with Tmin −20 °C /
Tmax +45 °C / ΔT_cell 30 °C, 6 modules per string, 3 kW AC required at
DC/AC target 1.2. Every expected number below is read straight from the
spreadsheet cells, including its two deliberate current-check failures.
"""

from __future__ import annotations

import pytest

from sim_stochastic_pv.simulation.electrical import (
    InverterElectricalSpecs,
    PanelElectricalSpecs,
)
from sim_stochastic_pv.simulation.electrical_design import (
    CableParams,
    DesignRequirements,
    DesignSite,
    evaluate_design,
)


def _reference_panel() -> PanelElectricalSpecs:
    """TCL HSM-ND54-DR505 datasheet (spreadsheet Database sheet)."""
    return PanelElectricalSpecs(
        power_w=505.0,
        v_oc_stc_v=40.14,
        v_mpp_stc_v=33.9,
        i_sc_stc_a=15.88,
        i_mpp_stc_a=14.9,
        n_cells_series=108,
        beta_voc_pct_per_c=-0.25,
        gamma_pmax_pct_per_c=-0.29,
        noct_c=45.0,
        alpha_isc_pct_per_c=0.045,
        v_system_max_v=1500.0,
        max_series_fuse_a=30.0,
    )


def _reference_inverter() -> InverterElectricalSpecs:
    """ZCS 1PH 3000-TLM-V3 datasheet (spreadsheet Database sheet)."""
    return InverterElectricalSpecs(
        v_dc_min_v=90.0,
        v_dc_max_v=600.0,
        v_mppt_min_v=90.0,
        v_mppt_max_v=580.0,
        n_mppt_trackers=2,
        i_dc_max_per_mppt_a=12.0,
        i_sc_max_per_mppt_a=15.0,
        max_strings_per_mppt=1,
        v_mppt_full_load_min_v=160.0,
        v_mppt_full_load_max_v=520.0,
        p_ac_nom_kw=3.0,
        efficiency_max=0.972,
    )


def _reference_site() -> DesignSite:
    return DesignSite(t_min_c=-20.0, t_max_c=45.0, delta_t_cell_c=30.0)


def _reference_requirements(**overrides) -> DesignRequirements:
    base = dict(
        p_ac_required_kw=3.0,
        target_dc_ac_ratio=1.2,
        n_panels_per_string=6,
        safety_factor_isc=1.25,
        max_cable_loss_fraction=0.005,
        fuse_factor_min=1.5,
        fuse_factor_max=2.4,
    )
    base.update(overrides)
    return DesignRequirements(**base)


@pytest.fixture()
def evaluation():
    return evaluate_design(
        _reference_panel(),
        _reference_inverter(),
        _reference_site(),
        _reference_requirements(),
        CableParams(length_one_way_m=30.0, operating_temperature_c=70.0),
    )


class TestTemperatureCorrected:
    def test_matches_spreadsheet(self, evaluation) -> None:
        c = evaluation.corrected
        assert c.t_cell_cold_c == pytest.approx(-20.0)
        assert c.t_cell_hot_c == pytest.approx(75.0)
        assert c.v_oc_cold_v == pytest.approx(44.65575, abs=1e-4)
        assert c.v_mp_cold_v == pytest.approx(38.32395, abs=1e-4)
        assert c.v_mp_hot_v == pytest.approx(28.9845, abs=1e-4)
        assert c.i_sc_hot_a == pytest.approx(16.2373, abs=1e-4)
        assert c.i_sc_design_a == pytest.approx(20.296625, abs=1e-5)


class TestStringBounds:
    def test_matches_spreadsheet(self, evaluation) -> None:
        b = evaluation.bounds
        assert b.v_limit_v == pytest.approx(600.0)
        assert b.n_max_voc == 13
        assert b.n_max_mppt == 13
        assert b.n_min == 6
        assert b.n_max == 13
        assert b.feasible

    def test_infeasible_pairing_detected(self) -> None:
        """A 1500 V module on a tiny window admits no valid N."""
        from dataclasses import replace

        narrow = replace(
            _reference_inverter(),
            v_mppt_full_load_min_v=500.0,
            v_mppt_full_load_max_v=520.0,
            v_dc_max_v=550.0,
        )
        result = evaluate_design(
            _reference_panel(), narrow, _reference_site(),
            _reference_requirements(),
        )
        assert not result.bounds.feasible
        assert not result.all_checks_ok


class TestStringVoltages:
    def test_matches_spreadsheet(self, evaluation) -> None:
        v = evaluation.voltages
        assert v.n_in_range
        assert v.v_oc_string_cold_v == pytest.approx(267.9345, abs=1e-3)
        assert v.v_oc_margin_v == pytest.approx(332.0655, abs=1e-3)
        assert v.v_mp_string_hot_v == pytest.approx(173.907, abs=1e-3)
        assert v.v_mp_hot_margin_v == pytest.approx(13.907, abs=1e-3)
        assert v.v_mp_string_cold_v == pytest.approx(229.9437, abs=1e-3)
        assert v.v_mp_cold_margin_v == pytest.approx(290.0563, abs=1e-3)


class TestPlantSizing:
    def test_matches_spreadsheet(self, evaluation) -> None:
        p = evaluation.plant
        assert p.p_dc_target_kwp == pytest.approx(3.6)
        assert p.string_power_kwp == pytest.approx(3.03)
        assert p.n_strings == 2
        assert p.total_panels == 12
        assert p.p_dc_installed_kwp == pytest.approx(6.06)
        assert p.dc_ac_ratio == pytest.approx(2.02)


class TestCurrentChecks:
    def test_reference_case_fails_as_in_spreadsheet(self, evaluation) -> None:
        """The spreadsheet's own example exceeds both per-MPPT current
        limits ("SUPERATA I max operativa MPPT!") — the engine must
        reproduce the failure, margins included."""
        c = evaluation.currents
        assert c.strings_per_mppt == 1
        assert c.inputs_ok
        assert c.i_operating_a == pytest.approx(15.23525, abs=1e-4)
        assert c.i_operating_margin_a == pytest.approx(-3.23525, abs=1e-4)
        assert c.i_sc_a == pytest.approx(16.2373, abs=1e-4)
        assert c.i_sc_margin_a == pytest.approx(-1.2373, abs=1e-4)
        assert not evaluation.all_checks_ok

    def test_lower_dc_target_passes(self) -> None:
        """With DC/AC 1.0 a single string fits the current limits...
        almost: the module's I_mp(hot) of 15.2 A still exceeds the
        12 A operating limit — this inverter genuinely cannot exploit
        a 15 A-class module, whatever the layout. The check must stay
        red even at minimum size (engine vs wishful thinking)."""
        result = evaluate_design(
            _reference_panel(), _reference_inverter(), _reference_site(),
            _reference_requirements(target_dc_ac_ratio=1.0),
        )
        assert result.plant.n_strings == 1
        assert result.currents.i_operating_margin_a < 0


class TestTemperatureMargins:
    def test_matches_spreadsheet(self, evaluation) -> None:
        m = evaluation.margins
        assert m.t_min_admissible_c == pytest.approx(-571.512, abs=0.01)
        assert m.margin_cold_c == pytest.approx(551.512, abs=0.01)
        assert m.t_cell_max_admissible_c == pytest.approx(98.5768, abs=0.001)
        assert m.t_amb_max_admissible_c == pytest.approx(68.5768, abs=0.001)
        assert m.margin_hot_c == pytest.approx(23.5768, abs=0.001)
        assert m.t_min_mppt_tracking_c == pytest.approx(-511.7375, abs=0.001)
        assert m.robust


class TestProtections:
    def test_matches_spreadsheet(self, evaluation) -> None:
        p = evaluation.protection
        # 2 parallel strings → protection not required by the norm.
        assert p.protection_required is False
        assert p.i_fuse_min_a == pytest.approx(23.82, abs=1e-3)
        assert p.i_fuse_max_norm_a == pytest.approx(38.112, abs=1e-3)
        assert p.i_fuse_module_max_a == pytest.approx(30.0)
        assert p.recommended_fuse_a == 25.0
        assert p.fuse_within_module_limit is True
        assert p.fuse_within_norm_limit is True

    def test_three_strings_require_protection(self) -> None:
        result = evaluate_design(
            _reference_panel(), _reference_inverter(), _reference_site(),
            _reference_requirements(target_dc_ac_ratio=2.5),
        )
        assert result.plant.n_strings == 3
        assert result.protection.protection_required is True


class TestCables:
    def test_matches_spreadsheet(self, evaluation) -> None:
        t = evaluation.cables
        assert t.resistivity_ohm_mm2_per_m == pytest.approx(0.02093875, abs=1e-7)
        by_section = {row.section_mm2: row for row in t.rows}

        r4 = by_section[4.0]
        assert r4.resistance_ohm == pytest.approx(0.31408125, abs=1e-6)
        assert r4.voltage_drop_v == pytest.approx(4.679810625, abs=1e-6)
        assert r4.voltage_drop_fraction == pytest.approx(0.0269098, abs=1e-6)
        assert r4.loss_per_string_w == pytest.approx(69.7291783, abs=1e-4)
        assert r4.loss_total_kw == pytest.approx(0.139458356, abs=1e-6)
        assert r4.loss_fraction_of_dc == pytest.approx(0.02301293, abs=1e-6)
        assert r4.loss_ok is False

        r25 = by_section[25.0]
        assert r25.loss_fraction_of_dc == pytest.approx(0.003682069, abs=1e-6)
        assert r25.loss_ok is True

        assert t.recommended_section_mm2 == 25.0

    def test_relaxed_threshold_moves_recommendation(self) -> None:
        """At the typical 1% target the 10 mm² section already passes."""
        result = evaluate_design(
            _reference_panel(), _reference_inverter(), _reference_site(),
            _reference_requirements(max_cable_loss_fraction=0.01),
            CableParams(length_one_way_m=30.0),
        )
        assert result.cables.recommended_section_mm2 == 10.0

    def test_iz_rating_gates_recommendation(self) -> None:
        """A section that passes the loss check but cannot carry the
        design I_sc thermally must be skipped."""
        result = evaluate_design(
            _reference_panel(), _reference_inverter(), _reference_site(),
            _reference_requirements(max_cable_loss_fraction=0.05),
            CableParams(
                length_one_way_m=5.0,
                sections_mm2=(2.5, 4.0),
                iz_a=(18.0, 38.0),  # design I_sc = 20.3 A > 18 → skip 2.5
            ),
        )
        assert result.cables.rows[0].loss_ok is True
        assert result.cables.rows[0].iz_ok is False
        assert result.cables.recommended_section_mm2 == 4.0

    def test_cost_column_present_with_price_list(self) -> None:
        result = evaluate_design(
            _reference_panel(), _reference_inverter(), _reference_site(),
            _reference_requirements(),
            CableParams(
                length_one_way_m=30.0,
                sections_mm2=(4.0, 6.0),
                price_eur_per_m=(0.9, 1.3),
            ),
        )
        # 2 strings × 2 × 30 m × €/m
        assert result.cables.rows[0].cost_total_eur == pytest.approx(108.0)
        assert result.cables.rows[1].cost_total_eur == pytest.approx(156.0)


class TestMissingFields:
    def test_missing_datasheet_field_named_in_error(self) -> None:
        panel = PanelElectricalSpecs(power_w=505.0, v_oc_stc_v=40.14)
        with pytest.raises(ValueError, match="v_mpp_stc_v"):
            evaluate_design(
                panel, _reference_inverter(), _reference_site(),
                _reference_requirements(),
            )
