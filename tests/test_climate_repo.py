"""
Tests for the Phase-15 ClimateProfileRepository and the
serialize/deserialize round-trip between :class:`ThermalModel` and
the JSON-backed DB row.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.persistence import (
    ClimateProfileRepository,
    PersistenceService,
)
from sim_stochastic_pv.persistence.climate_repo import (
    deserialize_thermal_model,
    serialize_thermal_model,
)
from sim_stochastic_pv.simulation.thermal import (
    GPDTail,
    HarmonicSeasonalMean,
    ThermalModel,
    ThermalMonthParams,
)


def _sample_model() -> ThermalModel:
    """Build a non-trivial ThermalModel with GPD tails on some months."""
    harmonic = HarmonicSeasonalMean(a0=12.0, a1=-10.0, a2=1.5)
    params = []
    for m in range(12):
        upper = (
            GPDTail(threshold=4.0, shape=0.05, scale=2.0, exceedance_prob=0.08)
            if 5 <= m <= 8     # summer: heatwave tail
            else None
        )
        lower = (
            GPDTail(threshold=4.5, shape=-0.1, scale=1.8, exceedance_prob=0.07)
            if m in (0, 1, 11) # winter: cold-snap tail
            else None
        )
        params.append(
            ThermalMonthParams(
                t_std_residual_c=2.0 + 0.1 * m,
                persistence_phi=0.7,
                t_amplitude_c=4.0 + 0.2 * m,
                gpd_upper=upper,
                gpd_lower=lower,
            )
        )
    return ThermalModel(harmonic, params, climate_trend_c_per_year=0.025)


class TestSerializeRoundTrip:
    def test_serialize_then_deserialize_recovers_model(self) -> None:
        model = _sample_model()
        blob = serialize_thermal_model(model)
        restored = deserialize_thermal_model(blob)

        assert restored.harmonic.a0 == pytest.approx(model.harmonic.a0)
        assert restored.harmonic.a1 == pytest.approx(model.harmonic.a1)
        assert restored.harmonic.a2 == pytest.approx(model.harmonic.a2)
        assert restored.climate_trend_c_per_year == pytest.approx(
            model.climate_trend_c_per_year,
        )
        for orig, rest in zip(model.monthly_params, restored.monthly_params):
            assert orig.t_std_residual_c == pytest.approx(rest.t_std_residual_c)
            assert orig.persistence_phi == pytest.approx(rest.persistence_phi)
            assert orig.t_amplitude_c == pytest.approx(rest.t_amplitude_c)
            assert (orig.gpd_upper is None) == (rest.gpd_upper is None)
            assert (orig.gpd_lower is None) == (rest.gpd_lower is None)
            if orig.gpd_upper is not None:
                assert orig.gpd_upper.threshold == pytest.approx(
                    rest.gpd_upper.threshold,
                )
                assert orig.gpd_upper.shape == pytest.approx(rest.gpd_upper.shape)

    def test_round_trip_produces_byte_identical_simulation(self) -> None:
        """A simulation from the restored model must reproduce the original
        for a fixed seed — proves the serialization is lossless."""
        model = _sample_model()
        blob = serialize_thermal_model(model)
        restored = deserialize_thermal_model(blob)
        a = model.simulate_daily_means(365, np.random.default_rng(7))
        b = restored.simulate_daily_means(365, np.random.default_rng(7))
        np.testing.assert_allclose(a, b)


class TestClimateProfileRepository:
    def test_upsert_and_get_by_name(self, sqlite_session_factory) -> None:
        repo = ClimateProfileRepository(sqlite_session_factory)
        blob = serialize_thermal_model(_sample_model())
        record = repo.upsert_climate_profile({
            "name": "Pavullo_thermal",
            "location_name": "Pavullo nel Frignano",
            "latitude": 44.34,
            "longitude": 10.83,
            "elevation_m": 682.0,
            "source": "OpenMeteo Archive",
            "harmonic": blob["harmonic"],
            "monthly_params": blob["monthly_params"],
            "climate_trend_c_per_year": blob["climate_trend_c_per_year"],
            "lookback_window": {"start_year": 2015, "end_year": 2024},
            "notes": "RMSE=2.1°C",
        })
        assert record.id is not None
        assert record.name == "Pavullo_thermal"

        fetched = repo.get_climate_profile_by_name("Pavullo_thermal")
        assert fetched is not None
        assert fetched.latitude == pytest.approx(44.34)
        assert len(fetched.monthly_params) == 12

    def test_upsert_updates_existing(self, sqlite_session_factory) -> None:
        repo = ClimateProfileRepository(sqlite_session_factory)
        blob = serialize_thermal_model(_sample_model())
        repo.upsert_climate_profile({
            "name": "X",
            "location_name": "Loc",
            "latitude": 1.0,
            "longitude": 2.0,
            "harmonic": blob["harmonic"],
            "monthly_params": blob["monthly_params"],
            "climate_trend_c_per_year": 0.0,
        })
        repo.upsert_climate_profile({
            "name": "X",
            "location_name": "Loc2",
            "latitude": 1.0,
            "longitude": 2.0,
            "harmonic": blob["harmonic"],
            "monthly_params": blob["monthly_params"],
            "climate_trend_c_per_year": 0.05,   # update
        })
        records = repo.list_climate_profiles()
        assert len(records) == 1
        assert records[0].location_name == "Loc2"
        assert records[0].climate_trend_c_per_year == pytest.approx(0.05)

    def test_load_thermal_model_round_trip(self, sqlite_session_factory) -> None:
        repo = ClimateProfileRepository(sqlite_session_factory)
        original = _sample_model()
        blob = serialize_thermal_model(original)
        record = repo.upsert_climate_profile({
            "name": "rt",
            "location_name": "loc",
            "latitude": 0.0,
            "longitude": 0.0,
            "harmonic": blob["harmonic"],
            "monthly_params": blob["monthly_params"],
            "climate_trend_c_per_year": blob["climate_trend_c_per_year"],
        })
        loaded = repo.load_thermal_model(record.id)
        assert loaded is not None
        a = original.simulate_daily_means(365, np.random.default_rng(11))
        b = loaded.simulate_daily_means(365, np.random.default_rng(11))
        np.testing.assert_allclose(a, b)

    def test_delete_returns_bool(self, sqlite_session_factory) -> None:
        repo = ClimateProfileRepository(sqlite_session_factory)
        blob = serialize_thermal_model(_sample_model())
        record = repo.upsert_climate_profile({
            "name": "to_delete",
            "location_name": "x",
            "latitude": 0.0,
            "longitude": 0.0,
            "harmonic": blob["harmonic"],
            "monthly_params": blob["monthly_params"],
            "climate_trend_c_per_year": 0.0,
        })
        assert repo.delete_climate_profile(record.id) is True
        assert repo.delete_climate_profile(record.id) is False


class TestPersistenceServiceExposesClimate:
    def test_climate_attribute_present(self, sqlite_session_factory) -> None:
        svc = PersistenceService(session_factory=sqlite_session_factory)
        assert isinstance(svc.climate, ClimateProfileRepository)
