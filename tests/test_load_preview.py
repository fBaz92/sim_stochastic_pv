"""
Tests for the load-profile representative-week preview (library + endpoints).

The preview drives the demand side only (no PV/battery/inverter): it builds the
regime sub-profile, layers daily variability / discrete appliances / HVAC on
top, simulates a typical week of the chosen month across Monte Carlo paths, and
returns the mean + p05/p95 bands, a baseline/appliance/HVAC breakdown, annual
kWh totals, and (when a climate is supplied) the weekly temperature.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sim_stochastic_pv.api import dependencies
from sim_stochastic_pv.api.app import create_app
from sim_stochastic_pv.application import SimulationApplication
from sim_stochastic_pv.persistence import PersistenceService
from sim_stochastic_pv.scenario_builder import (
    build_default_appliance_profile_config,
    build_default_stochastic_load_config,
    build_default_thermal_load_config,
)
from sim_stochastic_pv.simulation import (
    ApplianceProfileConfig,
    HeatPumpConfig,
    HouseThermalConfig,
    MonthlyAverageLoadProfile,
    SetpointConfig,
    StochasticLoadConfig,
    ThermalLoadConfig,
    get_appliance_preset,
)
from sim_stochastic_pv.simulation.load_preview import (
    HOURS_PER_WEEK,
    simulate_load_profile_preview,
)
from sim_stochastic_pv.simulation.thermal import (
    HarmonicSeasonalMean,
    ThermalModel,
    ThermalMonthParams,
)

HOURS_PER_YEAR = 365 * 24


def _const_factory(watts: float = 300.0):
    """Factory for a flat profile of ``watts`` W in every (month, hour)."""
    return lambda: MonthlyAverageLoadProfile(np.full((12, 24), watts))


def _cold_model() -> ThermalModel:
    """A cold-year thermal model (a0 low) so January heating is non-trivial."""
    return ThermalModel(
        HarmonicSeasonalMean(a0=2.0, a1=-8.0, a2=0.0),
        [
            ThermalMonthParams(
                t_std_residual_c=1.0, persistence_phi=0.8, t_amplitude_c=4.0
            )
            for _ in range(12)
        ],
    )


def _client(persistence: PersistenceService) -> TestClient:
    """FastAPI test client with persistence overridden to the temp DB."""
    app = create_app()
    app.dependency_overrides[dependencies.get_application_service] = (
        lambda: SimulationApplication(
            save_outputs=False, persistence=persistence, result_builder=None
        )
    )
    app.dependency_overrides[dependencies.get_persistence_service] = lambda: persistence
    return TestClient(app)


# --------------------------------------------------------------------------
# Library: simulate_load_profile_preview
# --------------------------------------------------------------------------


class TestLoadPreviewLibrary:
    def test_constant_profile_flat_and_collapsed_bands(self):
        res = simulate_load_profile_preview(
            base_profile_factory=_const_factory(300.0),
            regime="home",
            month=0,
            n_paths=8,
            seed=1,
        )
        assert len(res.total_kw_mean) == HOURS_PER_WEEK == 168
        # 300 W → 0.3 kW everywhere.
        assert all(abs(x - 0.3) < 1e-9 for x in res.total_kw_mean)
        # No stochasticity → the band collapses onto the mean.
        assert all(
            abs(p95 - p05) < 1e-9
            for p05, p95 in zip(res.total_kw_p05, res.total_kw_p95)
        )
        # Deterministic annual baseline = 0.3 kW × 8760 h.
        assert abs(res.baseline_kwh_annual - 0.3 * HOURS_PER_YEAR) < 1e-6
        assert abs(res.annual_kwh_mean - 0.3 * HOURS_PER_YEAR) < 1e-6
        assert not res.has_appliances and not res.has_hvac and not res.has_thermal
        assert res.temp_out_c_mean is None and res.temp_in_c_mean is None

    def test_stochastic_opens_bands_keeps_mean(self):
        cfg = StochasticLoadConfig(enabled=True, sigma_log=0.3, phi_intra_day=0.5)
        res = simulate_load_profile_preview(
            base_profile_factory=_const_factory(300.0),
            regime="home",
            month=0,
            n_paths=80,
            seed=2,
            stochastic_config=cfg,
        )
        spreads = [p95 - p05 for p05, p95 in zip(res.total_kw_p05, res.total_kw_p95)]
        assert max(spreads) > 0.01  # bands genuinely open up
        # Unit-mean multiplier keeps the central tendency near 0.3 kW.
        assert abs(float(np.mean(res.total_kw_mean)) - 0.3) < 0.03

    def test_appliances_add_load_and_annual_kwh(self):
        wm = get_appliance_preset("washing_machine")
        cfg = ApplianceProfileConfig(
            enabled=True, smart_pv_default=False, appliances=(wm,)
        )
        res = simulate_load_profile_preview(
            base_profile_factory=_const_factory(300.0),
            regime="home",
            month=0,
            n_paths=40,
            seed=3,
            appliance_config=cfg,
        )
        assert res.has_appliances
        assert sum(res.appliance_kw_mean) > 0.0
        assert wm.name in res.appliance_kwh_annual_by_name
        # Annual appliance energy matches the deterministic expectation.
        assert abs(res.appliance_kwh_annual - wm.expected_kwh_annual()) < 1e-6
        # The total annual = baseline + appliances.
        assert abs(
            res.annual_kwh_mean
            - (res.baseline_kwh_annual + res.appliance_kwh_annual)
        ) < 1e-6

    def test_no_layers_means_no_appliances(self):
        res = simulate_load_profile_preview(
            base_profile_factory=_const_factory(),
            regime="home",
            month=5,
            n_paths=4,
            seed=4,
        )
        assert not res.has_appliances
        assert all(x == 0.0 for x in res.appliance_kw_mean)
        assert all(x == 0.0 for x in res.hvac_kw_mean)

    def test_reproducible_for_same_seed(self):
        cfg = StochasticLoadConfig(enabled=True, sigma_log=0.2, phi_intra_day=0.5)
        kwargs = dict(
            base_profile_factory=_const_factory(),
            regime="home",
            month=3,
            n_paths=20,
            seed=7,
            stochastic_config=cfg,
        )
        a = simulate_load_profile_preview(**kwargs)
        b = simulate_load_profile_preview(**kwargs)
        assert a.total_kw_mean == b.total_kw_mean
        assert a.total_kw_p95 == b.total_kw_p95

    def test_hvac_and_temperature_in_cold_winter(self):
        tl = ThermalLoadConfig(
            enabled=True,
            house=HouseThermalConfig(floor_area_m2=100.0, insulation_preset="poor"),
            heat_pump=HeatPumpConfig(
                cop_heating=3.5, cop_cooling=3.0, p_elec_max_kw=5.0
            ),
            setpoint=SetpointConfig(
                t_setpoint_heating_c=20.0, t_setpoint_cooling_c=26.0
            ),
            dynamic=False,
        )
        res = simulate_load_profile_preview(
            base_profile_factory=_const_factory(300.0),
            regime="home",
            month=0,
            n_paths=15,
            seed=9,
            thermal_load_config=tl,
            thermal_model=_cold_model(),
        )
        assert res.has_hvac and res.has_thermal
        assert res.hvac_kwh_annual_mean > 0.0
        assert sum(res.hvac_kw_mean) > 0.0  # heating draw in January
        assert res.temp_out_c_mean is not None
        assert len(res.temp_out_c_mean) == 168
        assert float(np.mean(res.temp_out_c_mean)) < 15.0  # cold winter
        # Steady-state HVAC → no indoor-temperature trajectory.
        assert res.temp_in_c_mean is None

    def test_temperature_without_hvac(self):
        res = simulate_load_profile_preview(
            base_profile_factory=_const_factory(),
            regime="home",
            month=6,
            n_paths=10,
            seed=5,
            thermal_model=_cold_model(),
        )
        assert res.has_thermal and not res.has_hvac
        assert res.temp_out_c_mean is not None
        assert all(x == 0.0 for x in res.hvac_kw_mean)

    def test_invalid_month_raises(self):
        with pytest.raises(ValueError):
            simulate_load_profile_preview(
                base_profile_factory=_const_factory(), regime="home", month=12
            )


# --------------------------------------------------------------------------
# API endpoints
# --------------------------------------------------------------------------


class TestLoadPreviewEndpoints:
    def test_inline_preview(self, persistence: PersistenceService):
        client = _client(persistence)
        payload = {
            "profile_type": "custom_24h",
            "data": {"monthly_24h_w": [[300.0] * 24 for _ in range(12)]},
            "month": 0,
            "regime": "home",
            "n_paths": 5,
        }
        r = client.post("/api/profiles/load/preview", json=payload)
        assert r.status_code == 200, r.text
        body = r.json()
        assert len(body["total_kw_mean"]) == 168
        assert abs(body["annual_kwh_mean"] - 0.3 * HOURS_PER_YEAR) < 1.0

    def test_saved_preview_home_vs_away_drops_appliances(
        self, persistence: PersistenceService
    ):
        rec = persistence.upsert_load_profile(
            "Preview test",
            "home_away",
            {
                "kind": "home_away",
                "home": {"monthly_24h_w": [[300.0] * 24 for _ in range(12)]},
                "away": {"monthly_24h_w": [[100.0] * 24 for _ in range(12)]},
                "appliances": {
                    "enabled": True,
                    "smart_pv": False,
                    "items": [{"type": "washing_machine"}],
                },
            },
        )
        client = _client(persistence)

        home = client.post(
            f"/api/profiles/load/{rec.id}/preview",
            json={"month": 0, "regime": "home", "n_paths": 5},
        )
        assert home.status_code == 200, home.text
        assert home.json()["has_appliances"] is True

        # The away regime is intentionally simple: appliances are home-only.
        away = client.post(
            f"/api/profiles/load/{rec.id}/preview",
            json={"month": 0, "regime": "away", "n_paths": 5},
        )
        assert away.status_code == 200, away.text
        assert away.json()["has_appliances"] is False

    def test_saved_preview_missing_profile_404(
        self, persistence: PersistenceService
    ):
        client = _client(persistence)
        r = client.post("/api/profiles/load/999999/preview", json={"month": 0})
        assert r.status_code == 404

    def test_hvac_without_climate_is_400(self, persistence: PersistenceService):
        rec = persistence.upsert_load_profile(
            "HVAC no climate",
            "custom_24h",
            {
                "monthly_24h_w": [[300.0] * 24 for _ in range(12)],
                "thermal": {
                    "enabled": True,
                    "house": {"floor_area_m2": 100, "insulation_preset": "poor"},
                    "heat_pump": {"p_elec_max_kw": 3.0},
                    "setpoint": {
                        "t_setpoint_heating_c": 20,
                        "t_setpoint_cooling_c": 26,
                    },
                },
            },
        )
        client = _client(persistence)
        r = client.post(
            f"/api/profiles/load/{rec.id}/preview",
            json={"month": 0, "regime": "home", "n_paths": 3},
        )
        assert r.status_code == 400


class TestScenarioInheritsProfilePersonality:
    """A scenario referencing a load profile inherits its variability /
    appliances / HVAC blocks (hydration copies the profile ``data`` into
    ``data["load_profile"]``)."""

    def test_stochastic_inherited_from_profile(self):
        data = {
            "load_profile": {
                "kind": "home_away",
                "home": {"monthly_24h_w": [[300.0] * 24 for _ in range(12)]},
                "away": {"type": "arera"},
                "stochastic": {
                    "enabled": True,
                    "sigma_log": 0.25,
                    "phi_intra_day": 0.4,
                },
            }
        }
        cfg = build_default_stochastic_load_config(data)
        assert cfg is not None and cfg.enabled
        assert abs(cfg.sigma_log - 0.25) < 1e-9

    def test_appliances_inherited_from_profile(self):
        data = {
            "load_profile": {
                "appliances": {
                    "enabled": True,
                    "smart_pv": False,
                    "items": [{"type": "washing_machine"}],
                }
            }
        }
        cfg = build_default_appliance_profile_config(data)
        assert cfg is not None and cfg.enabled
        assert len(cfg.appliances) == 1

    def test_thermal_inherited_from_profile(self):
        data = {
            "load_profile": {
                "thermal": {
                    "enabled": True,
                    "house": {"floor_area_m2": 100, "insulation_preset": "poor"},
                    "heat_pump": {"p_elec_max_kw": 3.0},
                    "setpoint": {
                        "t_setpoint_heating_c": 20,
                        "t_setpoint_cooling_c": 26,
                    },
                }
            }
        }
        cfg = build_default_thermal_load_config(data)
        assert cfg is not None and cfg.enabled

    def test_scenario_thermal_overrides_profile(self):
        # An explicit scenario-level block wins over the profile's block.
        data = {
            "thermal_load": {"enabled": False},
            "load_profile": {
                "thermal": {
                    "enabled": True,
                    "house": {"floor_area_m2": 100, "insulation_preset": "poor"},
                    "heat_pump": {"p_elec_max_kw": 3.0},
                    "setpoint": {
                        "t_setpoint_heating_c": 20,
                        "t_setpoint_cooling_c": 26,
                    },
                }
            },
        }
        assert build_default_thermal_load_config(data) is None

    def test_no_blocks_means_legacy_none(self):
        data = {
            "load_profile": {
                "kind": "home_away",
                "home": {"monthly_24h_w": [[100.0] * 24 for _ in range(12)]},
                "away": {"type": "arera"},
            }
        }
        assert build_default_stochastic_load_config(data) is None
        assert build_default_appliance_profile_config(data) is None
        assert build_default_thermal_load_config(data) is None
