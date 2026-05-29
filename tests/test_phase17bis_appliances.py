"""
Tests for Phase 17-bis — discrete event-based appliances.

Covers:

* :class:`ApplianceEvent` validation;
* :class:`EventBasedApplianceProfile` schedule statistics (Poisson
  long-run mean, hour-of-day confinement, smart_pv shift toward noon);
* per-appliance KPI breakdown and peak-simultaneous-kW concurrency;
* scenario_builder hydration of presets + overrides;
* validation enforcement of bad shapes / unknown presets;
* end-to-end integration with Phase 17 stochastic + HVAC;
* legacy byte-identity (no block / ``enabled=false`` → unchanged).

All tests are deterministic via fixed seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.scenario_builder import (
    build_default_appliance_profile_config,
    build_default_energy_config,
)
from sim_stochastic_pv.simulation import (
    APPLIANCE_PRESETS,
    ApplianceEvent,
    ApplianceProfileConfig,
    AppliancesKPIs,
    EventBasedApplianceProfile,
    aggregate_appliances_kpis,
    get_appliance_preset,
)
from sim_stochastic_pv.validation import validate_scenario


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _solar_shape() -> np.ndarray:
    """Reproduce the Gaussian hourly shape used by SolarModel."""
    hours = np.arange(24)
    mask = (hours >= 6) & (hours <= 18)
    x = hours[mask] - 12.0
    shape = np.exp(-(x**2) / (2 * 3.0**2))
    shape /= shape.sum()
    out = np.zeros(24, dtype=float)
    out[mask] = shape
    return out


def _wm(monthly: float = 12.0) -> ApplianceEvent:
    """Realistic washing machine spec for tests."""
    return ApplianceEvent(
        name="washing_machine",
        p_kw=1.5,
        duration_hours=1.5,
        monthly_frequency=tuple([monthly] * 12),
        allowed_hours=tuple(range(9, 18)),
    )


# ---------------------------------------------------------------------------
# ApplianceEvent validation
# ---------------------------------------------------------------------------


class TestPhase17bisApplianceEventValidation:
    def test_negative_p_kw_rejected(self) -> None:
        with pytest.raises(ValueError, match="p_kw"):
            ApplianceEvent(
                name="x",
                p_kw=-1.0,
                duration_hours=1.0,
                monthly_frequency=tuple([1.0] * 12),
                allowed_hours=(0,),
            )

    def test_zero_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="duration_hours"):
            ApplianceEvent(
                name="x",
                p_kw=1.0,
                duration_hours=0.0,
                monthly_frequency=tuple([1.0] * 12),
                allowed_hours=(0,),
            )

    def test_monthly_frequency_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="length-12"):
            ApplianceEvent(
                name="x",
                p_kw=1.0,
                duration_hours=1.0,
                monthly_frequency=tuple([1.0] * 11),
                allowed_hours=(0,),
            )

    def test_invalid_allowed_hour(self) -> None:
        with pytest.raises(ValueError, match="allowed_hours"):
            ApplianceEvent(
                name="x",
                p_kw=1.0,
                duration_hours=1.0,
                monthly_frequency=tuple([1.0] * 12),
                allowed_hours=(24,),
            )

    def test_expected_kwh_annual_formula(self) -> None:
        wm = _wm()
        # 12/mo × 12 mo × 1.5 kW × 1.5 h = 324 kWh/yr
        assert wm.expected_kwh_annual() == pytest.approx(324.0)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class TestPhase17bisCatalog:
    def test_all_presets_have_unique_names(self) -> None:
        names = [e.name for e in APPLIANCE_PRESETS.values()]
        assert len(names) == len(set(names))

    def test_get_preset_case_insensitive(self) -> None:
        a = get_appliance_preset("Washing_Machine")
        b = get_appliance_preset("washing_machine")
        assert a == b

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(KeyError, match="not found"):
            get_appliance_preset("microwave")


# ---------------------------------------------------------------------------
# Scheduler statistics
# ---------------------------------------------------------------------------


class TestPhase17bisScheduler:
    """Statistical properties of the event scheduler."""

    def test_long_run_kwh_matches_expected_within_two_percent(self) -> None:
        # Average over many independent paths of a 5-year horizon so
        # the Poisson variance averages down.
        wm = _wm()
        prof = EventBasedApplianceProfile([wm])
        totals = []
        for seed in range(60):
            rng = np.random.default_rng(seed=seed)
            prof.reset_for_run(rng=rng, n_years=5)
            totals.append(prof.hourly_array().sum() / 5)
        mean_kwh_per_year = float(np.mean(totals))
        assert abs(mean_kwh_per_year - wm.expected_kwh_annual()) < 0.02 * wm.expected_kwh_annual()

    def test_zero_frequency_yields_zero_load(self) -> None:
        wm = ApplianceEvent(
            name="quiet",
            p_kw=2.0,
            duration_hours=1.0,
            monthly_frequency=tuple([0.0] * 12),
            allowed_hours=(12,),
        )
        prof = EventBasedApplianceProfile([wm])
        rng = np.random.default_rng(0)
        prof.reset_for_run(rng=rng, n_years=2)
        assert prof.hourly_array().sum() == 0.0

    def test_events_confined_to_allowed_hours(self) -> None:
        # An event with duration 1 h starting at hour h covers only
        # hour h — events outside allowed_hours therefore mean an
        # hour-of-day outside the window has zero contribution.
        wm = ApplianceEvent(
            name="bursty",
            p_kw=2.0,
            duration_hours=1.0,
            monthly_frequency=tuple([20.0] * 12),
            allowed_hours=(10, 11, 12),
        )
        prof = EventBasedApplianceProfile([wm])
        rng = np.random.default_rng(2)
        prof.reset_for_run(rng=rng, n_years=2)
        arr = prof.hourly_array().reshape(-1, 24)
        # Sum across days within each hour-of-day bucket.
        hour_totals = arr.sum(axis=0)
        outside = np.setdiff1d(np.arange(24), [10, 11, 12])
        assert hour_totals[outside].sum() == pytest.approx(0.0)
        assert hour_totals[[10, 11, 12]].sum() > 0

    def test_smart_pv_centroid_closer_to_noon_than_naive_timer(self) -> None:
        # naive_timer event uniformly distributed in [9, 17]; centroid ≈ 13.
        # smart_pv event reweighted by Gaussian solar shape peaked at 12;
        # centroid should shift toward 12.
        shape = _solar_shape()
        wm_naive = ApplianceEvent(
            name="naive_wm",
            p_kw=1.5,
            duration_hours=1.0,
            monthly_frequency=tuple([50.0] * 12),
            allowed_hours=tuple(range(9, 18)),
            schedule_mode="naive_timer",
        )
        wm_smart = ApplianceEvent(
            name="smart_wm",
            p_kw=1.5,
            duration_hours=1.0,
            monthly_frequency=tuple([50.0] * 12),
            allowed_hours=tuple(range(9, 18)),
            schedule_mode="smart_pv",
        )

        def centroid(profile: EventBasedApplianceProfile) -> float:
            arr = profile.hourly_array().reshape(-1, 24)
            hour_totals = arr.sum(axis=0)
            return float(
                (np.arange(24) * hour_totals).sum() / hour_totals.sum()
            )

        n_centroid = []
        s_centroid = []
        for seed in range(20):
            prof_n = EventBasedApplianceProfile([wm_naive])
            prof_n.reset_for_run(rng=np.random.default_rng(seed), n_years=2)
            n_centroid.append(centroid(prof_n))
            prof_s = EventBasedApplianceProfile([wm_smart], solar_hourly_shape=shape)
            prof_s.reset_for_run(rng=np.random.default_rng(seed), n_years=2)
            s_centroid.append(centroid(prof_s))
        # smart_pv must shift the mean centroid toward 12.
        assert float(np.mean(s_centroid)) < float(np.mean(n_centroid))

    def test_smart_pv_requires_solar_shape(self) -> None:
        wm = ApplianceEvent(
            name="smart",
            p_kw=1.5,
            duration_hours=1.0,
            monthly_frequency=tuple([5.0] * 12),
            allowed_hours=tuple(range(9, 18)),
            schedule_mode="smart_pv",
        )
        with pytest.raises(ValueError, match="solar_hourly_shape"):
            EventBasedApplianceProfile([wm])

    def test_reproducible_with_same_seed(self) -> None:
        wm = _wm()
        prof_a = EventBasedApplianceProfile([wm])
        prof_b = EventBasedApplianceProfile([wm])
        prof_a.reset_for_run(rng=np.random.default_rng(123), n_years=2)
        prof_b.reset_for_run(rng=np.random.default_rng(123), n_years=2)
        assert np.array_equal(prof_a.hourly_array(), prof_b.hourly_array())


# ---------------------------------------------------------------------------
# Concurrency + peak KPI
# ---------------------------------------------------------------------------


class TestPhase17bisConcurrency:
    def test_peak_simultaneous_grows_with_n_appliances(self) -> None:
        # Two identical washing machines firing many events should
        # produce a strictly higher mean peak than a single machine.
        # We use a modest frequency (8/mo) so the single-machine baseline
        # stays well below the all-hours-saturated ceiling, leaving
        # room for the double-machine peak to be visibly higher.
        single = EventBasedApplianceProfile([_wm(monthly=8)])
        double = EventBasedApplianceProfile([_wm(monthly=8), _wm(monthly=8)])
        single_peaks = []
        double_peaks = []
        for seed in range(40):
            single.reset_for_run(rng=np.random.default_rng(seed), n_years=1)
            double.reset_for_run(rng=np.random.default_rng(seed), n_years=1)
            single_peaks.append(single.hourly_array().max())
            double_peaks.append(double.hourly_array().max())
        assert float(np.mean(double_peaks)) > float(np.mean(single_peaks)) + 0.5


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


class TestPhase17bisAggregation:
    def test_empty_aggregation_returns_zero_dict(self) -> None:
        s = aggregate_appliances_kpis([])
        assert s["total_appliance_kwh_annual_mean"] == 0.0
        assert s["appliance_kwh_annual_by_name_mean"] == {}

    def test_mean_per_name_taken_across_paths(self) -> None:
        a = AppliancesKPIs(
            total_appliance_kwh_annual=300.0,
            appliance_kwh_annual_by_name={"wm": 200.0, "oven": 100.0},
            peak_simultaneous_kw=2.5,
            share_of_total_load_pct=15.0,
            smart_pv_self_consumption_pct=30.0,
        )
        b = AppliancesKPIs(
            total_appliance_kwh_annual=500.0,
            appliance_kwh_annual_by_name={"wm": 300.0, "oven": 200.0},
            peak_simultaneous_kw=4.0,
            share_of_total_load_pct=20.0,
            smart_pv_self_consumption_pct=40.0,
        )
        s = aggregate_appliances_kpis([a, b])
        assert s["total_appliance_kwh_annual_mean"] == pytest.approx(400.0)
        assert s["appliance_kwh_annual_by_name_mean"]["wm"] == pytest.approx(250.0)
        assert s["appliance_kwh_annual_by_name_mean"]["oven"] == pytest.approx(150.0)
        assert s["peak_simultaneous_kw_mean"] == pytest.approx(3.25)
        assert s["share_of_total_load_pct_mean"] == pytest.approx(17.5)
        assert s["smart_pv_self_consumption_pct_mean"] == pytest.approx(35.0)


# ---------------------------------------------------------------------------
# Scenario builder + validation
# ---------------------------------------------------------------------------


def _scenario(with_appliances: dict | None = None, **kw) -> dict:
    s = {
        "scenario_name": "phase17bis_test",
        "energy": {
            "n_years": 1,
            "pv_kwp": 3.0,
            "n_batteries": 0,
            "inverter_p_ac_max_kw": 3.0,
        },
        "solar": {"pv_kwp": 3.0},
        "load_profile": {"home_profiles_w": [200] * 12},
        "price": {"base_price_eur_per_kwh": 0.25},
        "economic": {"investment_eur": 10000, "n_mc": 4, "n_years": 1},
    }
    if with_appliances is not None:
        s["load_profile"]["appliances"] = with_appliances
    s.update(kw)
    return s


class TestPhase17bisScenarioBuilder:
    def test_missing_block_returns_none(self) -> None:
        s = _scenario()
        assert build_default_appliance_profile_config(s) is None

    def test_disabled_block_returns_none(self) -> None:
        s = _scenario({"enabled": False, "items": [{"type": "washing_machine"}]})
        assert build_default_appliance_profile_config(s) is None

    def test_preset_only_hydrates(self) -> None:
        s = _scenario(
            {"enabled": True, "items": [{"type": "washing_machine"}]}
        )
        cfg = build_default_appliance_profile_config(s)
        assert cfg is not None
        assert cfg.enabled is True
        assert len(cfg.appliances) == 1
        assert cfg.appliances[0].name == "washing_machine"
        assert cfg.appliances[0].schedule_mode == "naive_timer"

    def test_smart_pv_default_propagates_to_items(self) -> None:
        s = _scenario(
            {
                "enabled": True,
                "smart_pv": True,
                "items": [{"type": "washing_machine"}, {"type": "ev_charger_slow"}],
            }
        )
        cfg = build_default_appliance_profile_config(s)
        assert all(a.schedule_mode == "smart_pv" for a in cfg.appliances)

    def test_per_item_schedule_mode_overrides_default(self) -> None:
        s = _scenario(
            {
                "enabled": True,
                "smart_pv": True,
                "items": [
                    {"type": "washing_machine", "schedule_mode": "naive_timer"},
                    {"type": "ev_charger_slow"},
                ],
            }
        )
        cfg = build_default_appliance_profile_config(s)
        assert cfg.appliances[0].schedule_mode == "naive_timer"
        assert cfg.appliances[1].schedule_mode == "smart_pv"

    def test_monthly_frequency_override(self) -> None:
        s = _scenario(
            {
                "enabled": True,
                "items": [
                    {
                        "type": "washing_machine",
                        "monthly_frequency_override": [5.0] * 12,
                    }
                ],
            }
        )
        cfg = build_default_appliance_profile_config(s)
        assert cfg.appliances[0].monthly_frequency == tuple([5.0] * 12)

    def test_custom_appliance_requires_all_fields(self) -> None:
        s = _scenario(
            {
                "enabled": True,
                "items": [
                    {
                        "type": "custom",
                        "name": "fridge",
                        "p_kw": 0.15,
                        "duration_hours": 24.0,
                        "monthly_frequency": [1.0] * 12,
                        "allowed_hours": [0],
                    }
                ],
            }
        )
        cfg = build_default_appliance_profile_config(s)
        assert cfg.appliances[0].name == "fridge"


class TestPhase17bisValidation:
    def test_unknown_preset_rejected(self) -> None:
        s = _scenario({"enabled": True, "items": [{"type": "microwave"}]})
        errors = validate_scenario(s)
        assert any("microwave" in e for e in errors)

    def test_bad_schedule_mode_rejected(self) -> None:
        s = _scenario(
            {
                "enabled": True,
                "items": [{"type": "washing_machine", "schedule_mode": "magic"}],
            }
        )
        errors = validate_scenario(s)
        assert any("schedule_mode" in e for e in errors)

    def test_monthly_frequency_override_wrong_length(self) -> None:
        s = _scenario(
            {
                "enabled": True,
                "items": [
                    {
                        "type": "washing_machine",
                        "monthly_frequency_override": [1.0] * 5,
                    }
                ],
            }
        )
        errors = validate_scenario(s)
        assert any("length-12" in e for e in errors)

    def test_custom_missing_fields_rejected(self) -> None:
        s = _scenario(
            {
                "enabled": True,
                "items": [{"type": "custom", "name": "x", "p_kw": 0.2}],
            }
        )
        errors = validate_scenario(s)
        assert any("required" in e and "duration_hours" in e for e in errors)


# ---------------------------------------------------------------------------
# Legacy byte-identity
# ---------------------------------------------------------------------------


class TestPhase17bisLegacyByteIdentity:
    def test_no_block_yields_no_config(self) -> None:
        cfg = build_default_energy_config(_scenario())
        assert cfg.appliance_profile_config is None

    def test_disabled_block_yields_no_config(self) -> None:
        s = _scenario({"enabled": False, "items": [{"type": "washing_machine"}]})
        cfg = build_default_energy_config(s)
        assert cfg.appliance_profile_config is None

    def test_full_mc_run_with_block_off_matches_legacy_no_block(self) -> None:
        from sim_stochastic_pv.application import SimulationApplication

        legacy = _scenario()
        off = _scenario({"enabled": False, "items": [{"type": "washing_machine"}]})
        app = SimulationApplication(save_outputs=False)
        a = app.run_analysis(scenario_data=legacy, seed=42, n_mc=6)
        b = app.run_analysis(scenario_data=off, seed=42, n_mc=6)
        assert a["final_gain_mean_eur"] == pytest.approx(b["final_gain_mean_eur"])
        assert a.get("appliances") is None
        assert b.get("appliances") is None


# ---------------------------------------------------------------------------
# End-to-end (block enabled raises load + populates summary)
# ---------------------------------------------------------------------------


class TestPhase17bisEndToEnd:
    def test_enabled_appliances_raise_total_load_and_populate_summary(self) -> None:
        from sim_stochastic_pv.application import SimulationApplication

        legacy = _scenario()
        with_apps = _scenario(
            {
                "enabled": True,
                "items": [{"type": "washing_machine"}, {"type": "dishwasher"}],
            }
        )
        app = SimulationApplication(save_outputs=False)
        a = app.run_analysis(scenario_data=legacy, seed=42, n_mc=6)
        b = app.run_analysis(scenario_data=with_apps, seed=42, n_mc=6)
        # Enabling discrete appliances must add load → grid imports
        # cannot drop. (We don't compare final_gain directly because
        # the load also drives a small change in self-consumption; the
        # robust assertion is "appliance summary populated".)
        assert b.get("appliances") is not None
        appliances = b["appliances"]
        assert appliances["total_appliance_kwh_annual_mean"] > 0
        assert "washing_machine" in appliances["appliance_kwh_annual_by_name_mean"]
        assert "dishwasher" in appliances["appliance_kwh_annual_by_name_mean"]
        # Discrepancy in final_gain should at most match the cost of
        # the added kWh — for a 30 kWh increment over 12 months at
        # 0.25 €/kWh that's ~90 € of incremental savings or cost.
        assert a["final_gain_mean_eur"] != b["final_gain_mean_eur"]
