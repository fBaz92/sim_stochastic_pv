"""
Tests for the stochastic load decorator + HVAC additive load.

The two features are independent toggles. Together they cover most of the
remaining realism gap of the deterministic baseline load profile:

* :class:`StochasticLoadProfile` adds intra-day variability with the
  long-run mean preserved by an Itō-corrected LogN multiplier on top of
  any wrapped :class:`LoadProfile`.
* :class:`HvacController` adds an electric load proportional to
  ``UA × |T_set − T_out|`` divided by COP, capped at ``p_elec_max_kw``
  with a comfort-breach counter.

Both are byte-identical-no-op when their respective toggles stay off
(verified by :class:`TestPhase17LegacyByteIdentity`).
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.scenario_builder import (
    build_default_energy_config,
    build_default_stochastic_load_config,
    build_default_thermal_load_config,
)
from sim_stochastic_pv.simulation import (
    AreraLoadProfile,
    HeatPumpConfig,
    HouseThermalConfig,
    HvacController,
    MonthlyAverageLoadProfile,
    SetpointConfig,
    StochasticLoadConfig,
    StochasticLoadProfile,
    ThermalLoadConfig,
    aggregate_thermal_kpis,
)
from sim_stochastic_pv.simulation.load_profiles.stochastic import (
    _sample_lognormal_ar1_path,
)
from sim_stochastic_pv.simulation.thermal_load import (
    INSULATION_PRESETS,
    ThermalLoadKPIs,
    PRESET_GOOD_W_PER_C_PER_M2,
    PRESET_STANDARD_W_PER_C_PER_M2,
)
from sim_stochastic_pv.validation import validate_scenario


# ---------------------------------------------------------------------------
# StochasticLoadProfile statistical properties
# ---------------------------------------------------------------------------


class TestPhase17StochasticPathStats:
    """Long-run statistical properties of the AR(1) log-multiplier path."""

    def test_mean_eps_close_to_one(self) -> None:
        rng = np.random.default_rng(seed=1)
        eps = _sample_lognormal_ar1_path(
            n_hours=200_000, sigma_log=0.20, phi=0.5, rng=rng
        )
        # Itō correction → E[eps] should be 1 within ~0.5% with 200k samples.
        assert abs(float(eps.mean()) - 1.0) < 0.005

    def test_marginal_log_variance_matches_sigma_log_sq(self) -> None:
        rng = np.random.default_rng(seed=2)
        eps = _sample_lognormal_ar1_path(
            n_hours=200_000, sigma_log=0.25, phi=0.6, rng=rng
        )
        log_eps = np.log(eps)
        # Account for the Itō shift: shift cancels in std.
        assert abs(float(log_eps.std()) - 0.25) < 0.01

    def test_lag1_autocorrelation_recovers_phi(self) -> None:
        rng = np.random.default_rng(seed=3)
        eps = _sample_lognormal_ar1_path(
            n_hours=200_000, sigma_log=0.20, phi=0.7, rng=rng
        )
        log_eps = np.log(eps)
        c = np.corrcoef(log_eps[:-1], log_eps[1:])[0, 1]
        assert abs(c - 0.7) < 0.02

    def test_zero_sigma_returns_unity_path(self) -> None:
        eps = _sample_lognormal_ar1_path(
            n_hours=1000,
            sigma_log=0.0,
            phi=0.5,
            rng=np.random.default_rng(0),
        )
        assert np.allclose(eps, 1.0)


# ---------------------------------------------------------------------------
# StochasticLoadProfile decorator contract
# ---------------------------------------------------------------------------


class TestPhase17StochasticDecorator:
    """Decorator behaviour wrapping a deterministic LoadProfile."""

    def test_long_run_mean_preserved_within_one_percent(self) -> None:
        # Use MonthlyAverageLoadProfile (no calendar limits) — every
        # (month, hour) bucket returns the same value, so the wrapped
        # total is exactly base_total × sum(eps) / N_hours. The Itō
        # correction makes E[sum eps] / N → 1.
        base = MonthlyAverageLoadProfile(np.full((12, 24), 200.0))
        rng = np.random.default_rng(seed=7)
        wrapper = StochasticLoadProfile(
            base, StochasticLoadConfig(enabled=True, sigma_log=0.20, phi_intra_day=0.5)
        )
        wrapper.reset_for_run(rng=rng, n_years=3)

        base_total = 0.0
        stoch_total = 0.0
        for y in range(3):
            for m in range(12):
                for d in range(30):
                    for h in range(24):
                        base_total += base.get_hourly_load_kw(y, m, d, h, weekday=0)
                        stoch_total += wrapper.get_hourly_load_kw(y, m, d, h, weekday=0)
        rel_err = abs(stoch_total - base_total) / base_total
        assert rel_err < 0.01

    def test_zero_sigma_is_byte_identical_to_base(self) -> None:
        base = MonthlyAverageLoadProfile(np.full((12, 24), 250.0))
        wrapper = StochasticLoadProfile(
            base, StochasticLoadConfig(enabled=True, sigma_log=0.0)
        )
        wrapper.reset_for_run(rng=np.random.default_rng(0), n_years=1)
        for h in range(24):
            assert wrapper.get_hourly_load_kw(0, 0, 0, h, 0) == pytest.approx(
                base.get_hourly_load_kw(0, 0, 0, h, 0)
            )

    def test_config_rejects_invalid_sigma_or_phi(self) -> None:
        with pytest.raises(ValueError, match="sigma_log"):
            StochasticLoadConfig(enabled=True, sigma_log=-0.1)
        with pytest.raises(ValueError, match="phi_intra_day"):
            StochasticLoadConfig(enabled=True, phi_intra_day=1.0)


# ---------------------------------------------------------------------------
# HvacController — physics + dimensioning
# ---------------------------------------------------------------------------


def _make_thermal_cfg(
    preset: str = "standard",
    area: float = 100.0,
    cop_h: float = 3.5,
    cop_c: float = 3.0,
    p_max: float = 3.0,
    t_heat: float = 20.0,
    t_cool: float = 26.0,
    away_c: float | None = None,
    dynamic: bool = False,
    internal_gains_kw: float = 0.0,
) -> ThermalLoadConfig:
    return ThermalLoadConfig(
        enabled=True,
        house=HouseThermalConfig(
            floor_area_m2=area,
            insulation_preset=preset,
            internal_gains_kw=internal_gains_kw,
        ),
        heat_pump=HeatPumpConfig(cop_heating=cop_h, cop_cooling=cop_c, p_elec_max_kw=p_max),
        setpoint=SetpointConfig(
            t_setpoint_heating_c=t_heat,
            t_setpoint_cooling_c=t_cool,
            t_setpoint_away_c=away_c,
        ),
        dynamic=dynamic,
    )


class TestPhase17HvacController:
    """Steady-state HVAC controller — physical correctness."""

    def test_hvac_kw_equals_ua_times_delta_t_over_cop_heating(self) -> None:
        ctrl = HvacController(_make_thermal_cfg(preset="standard", area=100.0))
        t_amb = np.full(24, 5.0)
        p_elec, _ = ctrl.compute_hourly_p_elec_kw(t_amb)
        # UA = 1.5 W/°C/m² × 100 m² / 1000 = 0.15 kW/°C.
        expected = 0.15 * (20.0 - 5.0) / 3.5
        assert np.allclose(p_elec, expected)

    def test_hvac_zero_in_dead_band(self) -> None:
        ctrl = HvacController(_make_thermal_cfg())
        t_amb = np.array([21.0, 22.0, 23.0, 24.0, 25.0])
        p_elec, _ = ctrl.compute_hourly_p_elec_kw(t_amb)
        assert np.allclose(p_elec, 0.0)

    def test_hvac_scales_linearly_with_floor_area(self) -> None:
        small = HvacController(_make_thermal_cfg(area=80.0))
        large = HvacController(_make_thermal_cfg(area=160.0))
        t_amb = np.full(24, -5.0)
        p_small, _ = small.compute_hourly_p_elec_kw(t_amb)
        p_large, _ = large.compute_hourly_p_elec_kw(t_amb)
        assert p_large[0] == pytest.approx(2.0 * p_small[0])

    def test_better_insulation_yields_lower_hvac_kwh(self) -> None:
        poor = HvacController(_make_thermal_cfg(preset="poor"))
        good = HvacController(_make_thermal_cfg(preset="good"))
        t_amb = np.full(24 * 30, 0.0)  # full month at 0°C
        p_poor, _ = poor.compute_hourly_p_elec_kw(t_amb)
        p_good, _ = good.compute_hourly_p_elec_kw(t_amb)
        ratio = p_good.sum() / p_poor.sum()
        # good = 0.8 / poor = 2.5 → ratio ≈ 0.32.
        assert PRESET_GOOD_W_PER_C_PER_M2 / 2.5 - 0.01 < ratio < PRESET_GOOD_W_PER_C_PER_M2 / 2.5 + 0.01

    def test_comfort_breach_when_p_elec_max_below_demand(self) -> None:
        # 100m² standard insulation @ -20°C → UA=0.15, ΔT=40, COP=3.5
        # → P_elec=1.71 kW (under cap). Test with a much lower cap.
        ctrl = HvacController(_make_thermal_cfg(p_max=0.5))
        t_amb = np.full(48, -20.0)
        p_elec, kpis = ctrl.compute_hourly_p_elec_kw(t_amb)
        assert np.allclose(p_elec, 0.5)
        # 48 capped hours, but the KPI is per year.
        n_years = max(1, int(round(48 / (365.0 * 24.0))))
        expected_per_year = 48 / n_years
        assert kpis.comfort_breach_hours_per_year == pytest.approx(expected_per_year)
        assert kpis.p_elec_hvac_peak_kw == pytest.approx(0.5)

    def test_away_setpoint_keeps_hvac_off_when_none(self) -> None:
        ctrl = HvacController(_make_thermal_cfg(away_c=None))
        t_amb = np.full(24, -5.0)
        away_mask = np.zeros(24, dtype=bool)  # nobody home all day
        p_elec, _ = ctrl.compute_hourly_p_elec_kw(t_amb, at_home_hourly=away_mask)
        assert np.allclose(p_elec, 0.0)

    def test_away_setpoint_partial_run_when_set(self) -> None:
        # With away_c = 16°C, the heat pump still runs but at the lower
        # demand (UA × (16 − (−5)) instead of UA × (20 − (−5))).
        ctrl = HvacController(_make_thermal_cfg(away_c=16.0))
        t_amb = np.full(24, -5.0)
        away_mask = np.zeros(24, dtype=bool)
        p_elec, _ = ctrl.compute_hourly_p_elec_kw(t_amb, at_home_hourly=away_mask)
        ua = INSULATION_PRESETS["standard"] * 100.0 / 1000.0  # 0.15
        expected = ua * (16.0 - (-5.0)) / 3.5
        assert np.allclose(p_elec, expected)


# ---------------------------------------------------------------------------
# Phase 18 — dynamic RC mode (indoor-temperature trajectory)
# ---------------------------------------------------------------------------


class TestPhase18DynamicRc:
    """Dynamic implicit-Euler RC integration and its invariants."""

    def test_invariant_matches_steady_state_on_constant_tout(self) -> None:
        # Constant T_out below the heating setpoint, uncapped pump, no
        # internal gains, start at setpoint → the deadbeat controller holds
        # the setpoint every hour, so the per-hour electric draw is byte-for
        # -byte the steady-state result and the indoor temp stays at 20°C.
        t_amb = np.full(48, 5.0)
        ss, _ = HvacController(_make_thermal_cfg(p_max=1e6)).compute_hourly_p_elec_kw(t_amb)
        ctrl = HvacController(_make_thermal_cfg(p_max=1e6, dynamic=True))
        dy, _ = ctrl.compute_hourly_p_elec_kw(t_amb)
        assert np.allclose(dy, ss)
        assert np.allclose(ctrl.last_indoor_temp_c, 20.0)

    def test_annual_energy_close_to_steady_state_when_uncapped(self) -> None:
        # With gains=0 and no capping the dynamic energy differs from the
        # steady-state one only by the (small) stored-energy term at mode
        # transitions → within a few % over a full year. This underpins the
        # ROADMAP claim that the dynamic mode is "marginal for the economics".
        hours = np.arange(8760)
        t_amb = 12.0 + 10.0 * np.sin(hours * 2 * np.pi / 8760) + 5.0 * np.sin(hours * 2 * np.pi / 24)
        ss, _ = HvacController(_make_thermal_cfg("poor", 120.0, p_max=1e6)).compute_hourly_p_elec_kw(t_amb)
        dy, _ = HvacController(_make_thermal_cfg("poor", 120.0, p_max=1e6, dynamic=True)).compute_hourly_p_elec_kw(t_amb)
        assert dy.sum() == pytest.approx(ss.sum(), rel=0.05)

    def test_indoor_temp_drops_below_setpoint_when_capped(self) -> None:
        # Poorly insulated 120 m² home, undersized 2 kW pump, week-long cold
        # snap → the heat pump saturates and the house cannot hold 20°C.
        t_amb = np.full(168, -10.0) + 3.0 * np.sin(np.arange(168) * 2 * np.pi / 24)
        poor = HvacController(_make_thermal_cfg("poor", 120.0, p_max=2.0, dynamic=True))
        _, kp = poor.compute_hourly_p_elec_kw(t_amb)
        assert kp.t_in_min_c < 18.0           # visibly cold indoors
        assert kp.comfort_breach_hours_per_year > 0
        assert kp.p_elec_hvac_peak_kw == pytest.approx(2.0)  # pinned at the cap
        # A well-insulated home with the same pump holds the setpoint.
        good = HvacController(_make_thermal_cfg("good", 120.0, p_max=2.0, dynamic=True))
        _, kg = good.compute_hourly_p_elec_kw(t_amb)
        assert kg.t_in_min_c >= 19.0

    def test_implicit_euler_is_stable_with_short_time_constant(self) -> None:
        # Tiny thermal mass + leaky envelope → very short tau. Implicit Euler
        # must stay finite and bounded (no explicit-scheme oscillation).
        cfg = ThermalLoadConfig(
            enabled=True,
            house=HouseThermalConfig(
                floor_area_m2=200.0,
                insulation_preset="poor",
                capacitance_kwh_per_c_per_m2=0.005,  # tau ≈ 2 h
            ),
            heat_pump=HeatPumpConfig(cop_heating=3.5, cop_cooling=3.0, p_elec_max_kw=2.0),
            setpoint=SetpointConfig(t_setpoint_heating_c=20.0, t_setpoint_cooling_c=26.0),
            dynamic=True,
        )
        ctrl = HvacController(cfg)
        t_amb = np.full(72, -15.0)
        ctrl.compute_hourly_p_elec_kw(t_amb)
        t_in = ctrl.last_indoor_temp_c
        assert np.all(np.isfinite(t_in))
        # Heating only → bounded below by outdoor, above by the setpoint.
        assert t_in.min() >= t_amb.min() - 0.5
        assert t_in.max() <= 20.0 + 0.5

    def test_dead_band_no_load_dynamic(self) -> None:
        ctrl = HvacController(_make_thermal_cfg(dynamic=True))
        p_elec, _ = ctrl.compute_hourly_p_elec_kw(np.full(24, 23.0))
        assert np.allclose(p_elec, 0.0)

    def test_internal_gains_reduce_heating_energy(self) -> None:
        # Free internal heat offsets part of the heating demand.
        t_amb = np.full(168, 2.0)
        no_gain, _ = HvacController(
            _make_thermal_cfg("poor", 120.0, p_max=1e6, dynamic=True, internal_gains_kw=0.0)
        ).compute_hourly_p_elec_kw(t_amb)
        with_gain, _ = HvacController(
            _make_thermal_cfg("poor", 120.0, p_max=1e6, dynamic=True, internal_gains_kw=0.5)
        ).compute_hourly_p_elec_kw(t_amb)
        assert with_gain.sum() < no_gain.sum()

    def test_last_indoor_temp_none_in_steady_state(self) -> None:
        ctrl = HvacController(_make_thermal_cfg())  # dynamic=False
        _, kpis = ctrl.compute_hourly_p_elec_kw(np.full(24, 0.0))
        assert ctrl.last_indoor_temp_c is None
        # Steady-state exposes the setpoints as the indoor-temp KPIs.
        assert kpis.t_in_min_c == pytest.approx(20.0)
        assert kpis.t_in_max_c == pytest.approx(26.0)

    def test_aggregate_takes_worst_case_indoor_temp(self) -> None:
        cold = ThermalLoadKPIs(t_in_min_c=14.0, t_in_max_c=27.0)
        mild = ThermalLoadKPIs(t_in_min_c=19.0, t_in_max_c=30.0)
        agg = aggregate_thermal_kpis([cold, mild])
        assert agg["t_in_min_c"] == pytest.approx(14.0)  # coldest path
        assert agg["t_in_max_c"] == pytest.approx(30.0)  # hottest path


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestPhase17ThermalAggregation:
    def test_empty_input_returns_zero_dict(self) -> None:
        s = aggregate_thermal_kpis([])
        assert s["hvac_kwh_annual_mean"] == 0.0
        assert s["comfort_breach_hours_per_year_mean"] == 0.0

    def test_mean_taken_across_paths(self) -> None:
        a = ThermalLoadKPIs(
            hvac_kwh_annual=1000.0,
            hvac_share_of_total_load_pct=25.0,
            comfort_breach_hours_per_year=4.0,
            p_elec_hvac_peak_kw=2.5,
        )
        b = ThermalLoadKPIs(
            hvac_kwh_annual=1200.0,
            hvac_share_of_total_load_pct=30.0,
            comfort_breach_hours_per_year=10.0,
            p_elec_hvac_peak_kw=2.8,
        )
        s = aggregate_thermal_kpis([a, b])
        assert s["hvac_kwh_annual_mean"] == pytest.approx(1100.0)
        assert s["hvac_share_of_total_load_pct_mean"] == pytest.approx(27.5)
        assert s["comfort_breach_hours_per_year_mean"] == pytest.approx(7.0)
        assert s["p_elec_hvac_peak_kw_mean"] == pytest.approx(2.65)


# ---------------------------------------------------------------------------
# Scenario builder + validation
# ---------------------------------------------------------------------------


def _scenario(with_stochastic=False, with_thermal=False, **kw) -> dict:
    s = {
        "scenario_name": "phase17_test",
        "energy": {"n_years": 1, "pv_kwp": 3.0, "n_batteries": 0,
                   "inverter_p_ac_max_kw": 3.0},
        "solar": {"pv_kwp": 3.0},
        "load_profile": {"home_profiles_w": [200] * 12},
        "price": {"base_price_eur_per_kwh": 0.25},
        "economic": {"investment_eur": 10000, "n_mc": 5, "n_years": 1},
    }
    if with_stochastic:
        s["load_profile"]["stochastic"] = {"enabled": True, "sigma_log": 0.20}
    if with_thermal:
        s["thermal_load"] = {
            "enabled": True,
            "house": {"floor_area_m2": 100, "insulation_preset": "standard"},
            "heat_pump": {"cop_heating": 3.5, "cop_cooling": 3.0, "p_elec_max_kw": 3.0},
            "setpoint": {"t_setpoint_heating_c": 20, "t_setpoint_cooling_c": 26},
        }
    s.update(kw)
    return s


class TestPhase17ScenarioBuilder:
    def test_missing_blocks_return_none(self) -> None:
        s = _scenario()
        assert build_default_stochastic_load_config(s) is None
        assert build_default_thermal_load_config(s) is None

    def test_stochastic_block_hydrates(self) -> None:
        s = _scenario(with_stochastic=True)
        cfg = build_default_stochastic_load_config(s)
        assert cfg is not None
        assert cfg.enabled is True
        assert cfg.sigma_log == pytest.approx(0.20)

    def test_thermal_block_hydrates_into_three_subobjects(self) -> None:
        s = _scenario(with_thermal=True)
        cfg = build_default_thermal_load_config(s)
        assert cfg is not None
        assert cfg.enabled is True
        assert cfg.house.floor_area_m2 == pytest.approx(100)
        assert cfg.house.ua_kw_per_c == pytest.approx(
            PRESET_STANDARD_W_PER_C_PER_M2 * 100 / 1000.0
        )
        assert cfg.heat_pump.cop_heating == pytest.approx(3.5)
        assert cfg.setpoint.t_setpoint_heating_c == pytest.approx(20)


class TestPhase17ValidationIntegration:
    def test_validation_accepts_stochastic_block(self) -> None:
        s = _scenario(with_stochastic=True)
        errors = validate_scenario(s)
        assert not any(e.startswith("stochastic_load") for e in errors)

    def test_validation_rejects_thermal_without_climate(self) -> None:
        s = _scenario(with_thermal=True)
        errors = validate_scenario(s)
        assert any("climate_profile_id" in e for e in errors)

    def test_validation_rejects_setpoint_order(self) -> None:
        s = _scenario(with_thermal=True, climate_profile_id=1)
        s["thermal_load"]["setpoint"] = {
            "t_setpoint_heating_c": 28,
            "t_setpoint_cooling_c": 22,
        }
        errors = validate_scenario(s)
        assert any("dead-band" in e for e in errors)

    def test_validation_rejects_negative_cop(self) -> None:
        s = _scenario(with_thermal=True, climate_profile_id=1)
        s["thermal_load"]["heat_pump"] = {"cop_heating": -1.0}
        errors = validate_scenario(s)
        assert any("cop_heating" in e for e in errors)


# ---------------------------------------------------------------------------
# Byte-identity (legacy path preserved)
# ---------------------------------------------------------------------------


class TestPhase17LegacyByteIdentity:
    def test_no_phase17_block_yields_no_models(self) -> None:
        cfg = build_default_energy_config(_scenario())
        assert cfg.stochastic_load_config is None
        assert cfg.thermal_load_config is None

    def test_stochastic_disabled_yields_no_decorator(self) -> None:
        s = _scenario()
        s["load_profile"]["stochastic"] = {"enabled": False, "sigma_log": 0.20}
        cfg = build_default_energy_config(s)
        # The dataclass is present but disabled → simulator skips the wrap.
        assert cfg.stochastic_load_config is not None
        assert cfg.stochastic_load_config.enabled is False

    def test_thermal_enabled_without_climate_raises(self) -> None:
        s = _scenario(with_thermal=True)
        with pytest.raises(ValueError, match="climate_profile_id"):
            build_default_energy_config(s)

    def test_full_mc_run_with_blocks_off_matches_legacy_no_blocks(self) -> None:
        """End-to-end byte identity: no blocks ≡ blocks with enabled=false."""
        from sim_stochastic_pv.application import SimulationApplication

        legacy = _scenario()
        off = _scenario()
        off["load_profile"]["stochastic"] = {"enabled": False, "sigma_log": 0.20}

        app = SimulationApplication(save_outputs=False)
        a = app.run_analysis(scenario_data=legacy, seed=42, n_mc=8)
        b = app.run_analysis(scenario_data=off, seed=42, n_mc=8)

        assert a["final_gain_mean_eur"] == pytest.approx(b["final_gain_mean_eur"])
        assert a["prob_gain"] == pytest.approx(b["prob_gain"])
        assert a.get("thermal") is None
        assert b.get("thermal") is None
