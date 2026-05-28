"""
Tests for Phase 16 — detailed electrical model (MPPT window + DC shutdown).

The model is **opt-in**: scenarios without an ``electrical`` block, or
with ``electrical.mode='off'``, must produce byte-identical results to
the pre-Phase-16 energy path. Only ``mode='mppt_window'`` activates the
new logic; when active, it must reproduce realistic physics:

- string voltage rises at cold T_cell (winter sunrise) → can exceed
  ``v_dc_max`` and shut the inverter down;
- string voltage falls at hot T_cell (summer noon) → can drop below the
  MPPT window and force a soft derating;
- multi-MPPT strings sum correctly into per-tracker DC contributions.

Tests are deterministic (fixed ``rng``) and fast — every assertion runs
on a small horizon with synthetic inputs unless explicitly otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.scenario_builder import (
    build_default_electrical_model,
    build_default_energy_config,
)
from sim_stochastic_pv.simulation.electrical import (
    DEFAULT_DERATING_EXPONENT_K,
    ElectricalKPIs,
    ElectricalModel,
    InverterElectricalSpecs,
    PanelElectricalSpecs,
    PvString,
    aggregate_kpis,
    cell_temperature_c,
    missing_inverter_fields,
    missing_panel_fields,
    v_string_at_cell_temperature,
)
from sim_stochastic_pv.validation import validate_scenario


# ---------------------------------------------------------------------------
# Realistic specs used as test fixtures (Longi LR5 + Fronius Primo).
# ---------------------------------------------------------------------------


def _make_panel(noct_c: float = 45.0) -> PanelElectricalSpecs:
    """Realistic Longi LR5-72HPH-540M (Phase 16 seed catalog)."""
    return PanelElectricalSpecs(
        power_w=540.0,
        v_oc_stc_v=49.5,
        v_mpp_stc_v=41.5,
        i_sc_stc_a=13.92,
        i_mpp_stc_a=13.02,
        n_cells_series=144,
        beta_voc_pct_per_c=-0.27,
        gamma_pmax_pct_per_c=-0.34,
        noct_c=noct_c,
    )


def _make_inverter(
    v_dc_max_v: float = 1000.0,
    v_dc_min_v: float = 80.0,
    v_mppt_min_v: float = 240.0,
    v_mppt_max_v: float = 800.0,
) -> InverterElectricalSpecs:
    """Realistic Fronius Primo 5.0 (Phase 16 seed catalog)."""
    return InverterElectricalSpecs(
        v_dc_min_v=v_dc_min_v,
        v_dc_max_v=v_dc_max_v,
        v_mppt_min_v=v_mppt_min_v,
        v_mppt_max_v=v_mppt_max_v,
        n_mppt_trackers=2,
        i_dc_max_per_mppt_a=18.0,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


class TestPhase16Validation:
    """Spec validation: catch missing datasheet fields explicitly."""

    def test_missing_panel_fields_lists_blank_attrs(self) -> None:
        specs = PanelElectricalSpecs(power_w=540.0, v_oc_stc_v=49.5)
        missing = missing_panel_fields(specs)
        assert "v_mpp_stc_v" in missing
        assert "n_cells_series" in missing
        assert "beta_voc_pct_per_c" in missing
        assert "power_w" not in missing
        assert "v_oc_stc_v" not in missing

    def test_missing_inverter_fields_lists_blank_attrs(self) -> None:
        specs = InverterElectricalSpecs(v_dc_max_v=1000.0)
        missing = missing_inverter_fields(specs)
        assert "v_dc_min_v" in missing
        assert "v_mppt_min_v" in missing
        assert "v_mppt_max_v" in missing
        assert "v_dc_max_v" not in missing

    def test_complete_specs_yield_empty_missing_lists(self) -> None:
        assert missing_panel_fields(_make_panel()) == []
        assert missing_inverter_fields(_make_inverter()) == []


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestPhase16PureHelpers:
    """Pure physics helpers: T_cell, V_string at temperature."""

    def test_cell_temperature_at_zero_irradiance_equals_ambient(self) -> None:
        # NOCT correction is proportional to irradiance — zero W/m² → T_cell == T_ambient.
        result = cell_temperature_c(t_ambient_c=10.0, poa_irradiance_w_per_m2=0.0, noct_c=45.0)
        assert result == pytest.approx(10.0)

    def test_cell_temperature_at_full_sun_warmer_than_ambient(self) -> None:
        # 1000 W/m², NOCT=45 → T_cell = T_amb + (45-20)/800 * 1000 = T_amb + 31.25
        result = cell_temperature_c(t_ambient_c=25.0, poa_irradiance_w_per_m2=1000.0, noct_c=45.0)
        assert result == pytest.approx(25.0 + 25.0 / 800.0 * 1000.0)

    def test_v_string_at_stc_equals_nominal(self) -> None:
        # At 25°C the temperature factor is 1.0 → V_string = n * V_mpp_stc
        v_op = v_string_at_cell_temperature(
            v_mpp_stc_v=41.5,
            v_oc_stc_v=49.5,
            beta_voc_pct_per_c=-0.27,
            n_panels_in_string=12,
            t_cell_c=25.0,
            operating=True,
        )
        assert float(v_op) == pytest.approx(12 * 41.5)

    def test_v_string_rises_at_cold_t_cell(self) -> None:
        # Winter sunrise: T_cell = -10°C → V_oc rises ~10% with beta=-0.27%/°C.
        v_oc_cold = v_string_at_cell_temperature(
            v_mpp_stc_v=41.5,
            v_oc_stc_v=49.5,
            beta_voc_pct_per_c=-0.27,
            n_panels_in_string=12,
            t_cell_c=-10.0,
            operating=False,
        )
        v_oc_stc = 12 * 49.5
        # Δ = -35°C × (-0.27/100) = +0.0945 → V_oc(-10) = V_oc * 1.0945
        assert float(v_oc_cold) > v_oc_stc * 1.09
        assert float(v_oc_cold) < v_oc_stc * 1.10

    def test_v_string_falls_at_hot_t_cell(self) -> None:
        # Summer noon: T_cell = 70°C → V_mpp drops ~12% with beta=-0.27%/°C.
        v_op_hot = v_string_at_cell_temperature(
            v_mpp_stc_v=41.5,
            v_oc_stc_v=49.5,
            beta_voc_pct_per_c=-0.27,
            n_panels_in_string=12,
            t_cell_c=70.0,
            operating=True,
        )
        v_op_stc = 12 * 41.5
        assert float(v_op_hot) < v_op_stc * 0.89


# ---------------------------------------------------------------------------
# Constructor guard
# ---------------------------------------------------------------------------


class TestPhase16ConstructorGuard:
    """ElectricalModel must refuse incomplete specs and empty string list."""

    def test_constructor_rejects_missing_panel_fields(self) -> None:
        incomplete = PanelElectricalSpecs(power_w=540.0, v_oc_stc_v=49.5)
        with pytest.raises(ValueError, match="missing required datasheet fields"):
            ElectricalModel(
                panel=incomplete,
                inverter=_make_inverter(),
                strings=[PvString(n_panels=12)],
            )

    def test_constructor_rejects_empty_strings_list(self) -> None:
        with pytest.raises(ValueError, match="at least one PvString"):
            ElectricalModel(
                panel=_make_panel(),
                inverter=_make_inverter(),
                strings=[],
            )

    def test_constructor_rejects_negative_derating_exponent(self) -> None:
        with pytest.raises(ValueError, match="derating_exponent_k"):
            ElectricalModel(
                panel=_make_panel(),
                inverter=_make_inverter(),
                strings=[PvString(n_panels=12)],
                derating_exponent_k=-0.1,
            )


# ---------------------------------------------------------------------------
# Apply-to-DC behaviour
# ---------------------------------------------------------------------------


class TestPhase16ApplyToPvDc:
    """End-to-end behaviour of ElectricalModel.apply_to_pv_dc."""

    def test_normal_operating_range_passes_through(self) -> None:
        # 12-panel string at 35°C ambient, full sun → ~66°C T_cell.
        # V_op = 12 * 41.5 * (1 - 0.27/100 * 41) ≈ 442 V — inside MPPT
        # window [240..800] and below v_dc_max=1000 → derating only from
        # gamma_pmax (T_cell ~66°C → ~0.86 factor).
        model = ElectricalModel(
            panel=_make_panel(),
            inverter=_make_inverter(),
            strings=[PvString(n_panels=12)],
            n_years=1,
        )
        n_hours = 24
        pv_kw = np.zeros(n_hours)
        pv_kw[12] = 12 * 540 / 1000.0  # noon full sun
        t_amb = np.full(n_hours, 35.0)
        adjusted, kpis = model.apply_to_pv_dc(pv_kw, t_amb)
        # No overvoltage, no MPPT breach.
        assert kpis.hours_dc_overvoltage_per_year == 0
        assert kpis.hours_outside_mppt_per_year == 0
        # Noon hour derated only by gamma_pmax (T_cell elevated).
        assert adjusted[12] < pv_kw[12]
        assert adjusted[12] > 0.80 * pv_kw[12]
        # Nighttime hours untouched.
        assert adjusted[0] == 0.0

    def test_dc_overvoltage_shutdown_in_cold_morning(self) -> None:
        # Synthetic ultra-low V_dc_max so V_oc(15 panels at -10°C) breaches it.
        # 15 * 49.5 * 1.0945 ≈ 813 V → pick v_dc_max_v=600 to force shutdown.
        model = ElectricalModel(
            panel=_make_panel(),
            inverter=_make_inverter(v_dc_max_v=600.0, v_mppt_max_v=560.0),
            strings=[PvString(n_panels=15)],
            n_years=1,
        )
        n_hours = 24
        pv_kw = np.zeros(n_hours)
        # Producing sunrise hour where the inverter should refuse to operate.
        pv_kw[7] = 0.5
        t_amb = np.array([-10.0] * n_hours)
        adjusted, kpis = model.apply_to_pv_dc(pv_kw, t_amb)
        # At least one hour overvoltage; producing hour must be zeroed.
        assert kpis.hours_dc_overvoltage_per_year >= 1
        assert adjusted[7] == 0.0
        # Peak observed V_string must be above the inverter limit.
        assert kpis.peak_v_string_v > 600.0

    def test_mppt_window_breach_above_derates(self) -> None:
        # Construct a single small string whose V_op sits ABOVE MPPT max
        # at full power. 18 panels × 41.5 V × ~0.88 (at 60°C cell) ≈ 657 V
        # → above MPPT max=500 V. Inverter does NOT shut down (still
        # inside v_dc_max=1000) but derates the output.
        inv = _make_inverter(v_mppt_max_v=500.0, v_mppt_min_v=240.0)
        model = ElectricalModel(
            panel=_make_panel(),
            inverter=inv,
            strings=[PvString(n_panels=18)],
            n_years=1,
        )
        n_hours = 24
        pv_kw = np.zeros(n_hours)
        pv_kw[12] = 18 * 540 / 1000.0
        t_amb = np.full(n_hours, 25.0)  # T_cell at noon ~ 25 + (45-20)/800*1000 ≈ 56°C
        adjusted, kpis = model.apply_to_pv_dc(pv_kw, t_amb)
        assert kpis.hours_outside_mppt_per_year >= 1
        assert kpis.hours_dc_overvoltage_per_year == 0
        # Derating should reduce noon output but not zero it.
        assert 0 < adjusted[12] < pv_kw[12]

    def test_zero_derating_exponent_disables_mppt_penalty(self) -> None:
        # Same scenario as the previous test but with k=0 → ratio==1 →
        # no MPPT-window penalty, only the temperature derating.
        inv = _make_inverter(v_mppt_max_v=500.0, v_mppt_min_v=240.0)
        model = ElectricalModel(
            panel=_make_panel(),
            inverter=inv,
            strings=[PvString(n_panels=18)],
            derating_exponent_k=0.0,
            n_years=1,
        )
        n_hours = 24
        pv_kw = np.zeros(n_hours)
        pv_kw[12] = 18 * 540 / 1000.0
        t_amb = np.full(n_hours, 25.0)
        adjusted, kpis = model.apply_to_pv_dc(pv_kw, t_amb)
        # The KPI counter still ticks (we WERE outside the window) but
        # the actual power is left to the temperature derating only.
        assert kpis.hours_outside_mppt_per_year >= 1
        # Temperature factor at 56°C cell: 1 + (-0.34/100)*31 ≈ 0.895
        expected_temp_factor = 1.0 + (-0.34 / 100.0) * (
            25.0 + (45.0 - 20.0) / 800.0 * 1000.0 - 25.0
        )
        assert adjusted[12] == pytest.approx(pv_kw[12] * expected_temp_factor, rel=1e-6)

    def test_shape_mismatch_raises(self) -> None:
        model = ElectricalModel(
            panel=_make_panel(),
            inverter=_make_inverter(),
            strings=[PvString(n_panels=12)],
        )
        with pytest.raises(ValueError, match="same shape"):
            model.apply_to_pv_dc(np.zeros(24), np.zeros(48))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestPhase16Aggregation:
    """`aggregate_kpis` averages per-path counters and takes worst-case voltages."""

    def test_empty_input_yields_all_zero_dict(self) -> None:
        summary = aggregate_kpis([])
        assert summary["hours_dc_overvoltage_per_year_mean"] == 0.0
        assert summary["peak_v_string_v"] == 0.0
        assert summary["min_v_string_v"] == 0.0

    def test_aggregate_takes_worst_peak_and_minmin_voltage(self) -> None:
        a = ElectricalKPIs(
            hours_dc_overvoltage_per_year=2.0,
            hours_outside_mppt_per_year=10.0,
            peak_v_string_v=900.0,
            min_v_string_v=300.0,
        )
        b = ElectricalKPIs(
            hours_dc_overvoltage_per_year=4.0,
            hours_outside_mppt_per_year=20.0,
            peak_v_string_v=950.0,
            min_v_string_v=250.0,
        )
        summary = aggregate_kpis([a, b])
        assert summary["hours_dc_overvoltage_per_year_mean"] == pytest.approx(3.0)
        assert summary["hours_outside_mppt_per_year_mean"] == pytest.approx(15.0)
        assert summary["peak_v_string_v"] == pytest.approx(950.0)
        assert summary["min_v_string_v"] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# Scenario_builder + validation integration
# ---------------------------------------------------------------------------


def _make_minimal_scenario(electrical_block: dict | None = None) -> dict:
    """Build a minimal scenario dict with an optional electrical block."""
    scenario = {
        "scenario_name": "phase16_test",
        "energy": {
            "n_years": 1,
            "pv_kwp": 5.4,
            "n_batteries": 0,
            "inverter_p_ac_max_kw": 5.0,
        },
        "solar": {"pv_kwp": 5.4, "panel_tilt_degrees": 35, "panel_azimuth_degrees": 180},
        "load_profile": {"home_profiles_w": [200] * 12},
        "price": {"base_price_eur_per_kwh": 0.25},
        "economic": {"investment_eur": 10000, "n_mc": 5, "n_years": 1},
    }
    if electrical_block is not None:
        scenario["electrical"] = electrical_block
    return scenario


class TestPhase16ScenarioBuilder:
    """Scenario JSON → ElectricalModel hydration."""

    def test_missing_electrical_block_returns_none(self) -> None:
        scenario = _make_minimal_scenario(None)
        model = build_default_electrical_model(scenario)
        assert model is None

    def test_mode_off_returns_none(self) -> None:
        scenario = _make_minimal_scenario({"mode": "off"})
        model = build_default_electrical_model(scenario)
        assert model is None

    def test_unrecognised_mode_raises(self) -> None:
        scenario = _make_minimal_scenario({"mode": "weird_thing"})
        with pytest.raises(ValueError, match="not recognised"):
            build_default_electrical_model(scenario)

    def test_mppt_mode_with_complete_specs_yields_model(self) -> None:
        scenario = _make_minimal_scenario(
            {
                "mode": "mppt_window",
                "panel": {
                    "power_w": 540.0,
                    "v_oc_stc_v": 49.5,
                    "v_mpp_stc_v": 41.5,
                    "n_cells_series": 144,
                    "beta_voc_pct_per_c": -0.27,
                    "gamma_pmax_pct_per_c": -0.34,
                    "noct_c": 45.0,
                },
                "inverter": {
                    "v_dc_min_v": 80.0,
                    "v_dc_max_v": 1000.0,
                    "v_mppt_min_v": 240.0,
                    "v_mppt_max_v": 800.0,
                    "n_mppt_trackers": 2,
                },
            }
        )
        model = build_default_electrical_model(scenario)
        assert isinstance(model, ElectricalModel)
        # Single default string synthesised from pv_kwp / panel power.
        assert len(model.strings) == 1
        # 5.4 kWp / 540 W = 10 panels.
        assert model.strings[0].n_panels == 10
        assert model.derating_exponent_k == pytest.approx(DEFAULT_DERATING_EXPONENT_K)


class TestPhase16ValidationIntegration:
    """validate_scenario picks up electrical block constraints."""

    def test_validation_rejects_mppt_mode_without_climate_profile(self) -> None:
        scenario = _make_minimal_scenario(
            {
                "mode": "mppt_window",
                "panel": {
                    "power_w": 540.0,
                    "v_oc_stc_v": 49.5,
                    "v_mpp_stc_v": 41.5,
                    "n_cells_series": 144,
                    "beta_voc_pct_per_c": -0.27,
                    "gamma_pmax_pct_per_c": -0.34,
                    "noct_c": 45.0,
                },
                "inverter": {
                    "v_dc_min_v": 80.0,
                    "v_dc_max_v": 1000.0,
                    "v_mppt_min_v": 240.0,
                    "v_mppt_max_v": 800.0,
                },
            }
        )
        errors = validate_scenario(scenario)
        assert any("climate_profile_id" in e for e in errors)

    def test_validation_rejects_mppt_mode_missing_panel_fields(self) -> None:
        scenario = _make_minimal_scenario(
            {
                "mode": "mppt_window",
                "panel": {"power_w": 540.0},  # missing v_oc, v_mpp, ...
                "inverter": {
                    "v_dc_min_v": 80.0,
                    "v_dc_max_v": 1000.0,
                    "v_mppt_min_v": 240.0,
                    "v_mppt_max_v": 800.0,
                },
                "climate_profile_id": 1,
            }
        )
        scenario["climate_profile_id"] = 1
        errors = validate_scenario(scenario)
        assert any("v_oc_stc_v" in e for e in errors)
        assert any("v_mpp_stc_v" in e for e in errors)

    def test_validation_passes_when_mode_off(self) -> None:
        scenario = _make_minimal_scenario({"mode": "off"})
        errors = validate_scenario(scenario)
        # Any electrical-related errors must be absent in off mode.
        assert not any(e.startswith("electrical.") for e in errors)


# ---------------------------------------------------------------------------
# End-to-end byte-identity check (mode=off vs no block)
# ---------------------------------------------------------------------------


class TestPhase16LegacyByteIdentity:
    """When the electrical block is off or missing the simulation must be unchanged."""

    def test_energy_config_without_electrical_block_yields_no_models(self) -> None:
        cfg = build_default_energy_config(_make_minimal_scenario(None))
        assert cfg.electrical_model is None
        assert cfg.thermal_model is None

    def test_energy_config_with_mode_off_yields_no_models(self) -> None:
        cfg = build_default_energy_config(_make_minimal_scenario({"mode": "off"}))
        assert cfg.electrical_model is None
        assert cfg.thermal_model is None

    def test_full_mc_run_with_mode_off_matches_legacy_no_block(self) -> None:
        """End-to-end byte identity: mode='off' == no electrical block at all."""
        from sim_stochastic_pv.application import SimulationApplication

        # Two scenarios that should be byte-identical: one without an
        # electrical block, one with mode='off'. Run both and check that
        # the headline KPIs match exactly.
        legacy_scenario = _make_minimal_scenario(None)
        off_scenario = _make_minimal_scenario({"mode": "off"})

        app = SimulationApplication(save_outputs=False)
        legacy = app.run_analysis(scenario_data=legacy_scenario, seed=42, n_mc=10)
        off = app.run_analysis(scenario_data=off_scenario, seed=42, n_mc=10)
        # The final KPIs (deterministic given the seed) must match.
        assert legacy["final_gain_mean_eur"] == pytest.approx(off["final_gain_mean_eur"])
        assert legacy["prob_gain"] == pytest.approx(off["prob_gain"])
        # And neither summary exposes electrical KPIs.
        assert legacy.get("electrical") is None
        assert off.get("electrical") is None

    def test_mppt_mode_without_climate_profile_raises_in_energy_config(self) -> None:
        # build_default_energy_config refuses to build a partial setup —
        # the user must either downgrade to mode='off' or wire a climate
        # profile.
        scenario = _make_minimal_scenario(
            {
                "mode": "mppt_window",
                "panel": {
                    "power_w": 540.0,
                    "v_oc_stc_v": 49.5,
                    "v_mpp_stc_v": 41.5,
                    "n_cells_series": 144,
                    "beta_voc_pct_per_c": -0.27,
                    "gamma_pmax_pct_per_c": -0.34,
                    "noct_c": 45.0,
                },
                "inverter": {
                    "v_dc_min_v": 80.0,
                    "v_dc_max_v": 1000.0,
                    "v_mppt_min_v": 240.0,
                    "v_mppt_max_v": 800.0,
                },
            }
        )
        with pytest.raises(ValueError, match="climate_profile_id"):
            build_default_energy_config(scenario)
