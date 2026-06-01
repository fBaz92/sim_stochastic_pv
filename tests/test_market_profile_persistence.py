"""
Tests for market-profile persistence and scenario hydration.

Covers the round trip from a runtime :class:`MarketPriceProvider` to a stored
``MarketProfileModel`` and back, the repository CRUD, the
``build_default_market_provider`` scenario hydration, the light root-level
validation of ``market_profile_id``, and the default-profile seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim_stochastic_pv.market import PriceSurface
from sim_stochastic_pv.scenario_builder import build_default_market_provider
from sim_stochastic_pv.simulation import MarketPriceProvider
from sim_stochastic_pv.validation import validate_scenario


# ── helpers ────────────────────────────────────────────────────────────────


def _tiny_surface(value: float = 0.05, *, K: int = 3, n_years: int = 2) -> PriceSurface:
    grid = np.full((K, n_years, 12, 24), float(value))
    return PriceSurface(price_eur_per_kwh=grid, n_trajectories=K, n_years=n_years)


def _tiny_provider(**kwargs) -> MarketPriceProvider:
    return MarketPriceProvider(_tiny_surface(), pmg_base_eur_per_kwh=0.07, **kwargs)


# ── provider <-> config dict round trip ────────────────────────────────────


def test_provider_config_dict_round_trip():
    grid = np.linspace(0.01, 0.20, 2 * 3 * 12 * 24).reshape(2, 3, 12, 24)
    surface = PriceSurface(price_eur_per_kwh=grid, n_trajectories=2, n_years=3)
    provider = MarketPriceProvider(
        surface,
        pmg_base_eur_per_kwh=0.0413,
        retail_markup_fraction=0.8,
        retail_fixed_components_eur_per_kwh=0.11,
    )

    blob = provider.to_config_dict(build_config={"mix": "italian"})
    assert blob["version"] == 1
    assert blob["build_config"] == {"mix": "italian"}
    assert blob["surface"]["shape"] == [2, 3, 12, 24]

    clone = MarketPriceProvider.from_config_dict(blob)
    assert clone.pmg_base_eur_per_kwh == pytest.approx(0.0413)
    assert clone.retail_markup_fraction == pytest.approx(0.8)
    assert clone.retail_fixed_components_eur_per_kwh == pytest.approx(0.11)
    assert clone.price_surface.n_trajectories == 2
    assert clone.price_surface.n_years == 3
    assert np.allclose(
        clone.price_surface.price_eur_per_kwh,
        provider.price_surface.price_eur_per_kwh,
        atol=1e-6,
    )


def test_provider_config_dict_round_trip_no_retail():
    provider = _tiny_provider()
    clone = MarketPriceProvider.from_config_dict(provider.to_config_dict())
    assert clone.retail_markup_fraction is None
    assert clone.pmg_base_eur_per_kwh == pytest.approx(0.07)


# ── repository CRUD + hydration ────────────────────────────────────────────


def test_repo_upsert_get_and_load(persistence):
    blob = _tiny_provider().to_config_dict()
    record = persistence.upsert_market_profile(
        {"name": "P1", "description": "test profile", "data": blob}
    )
    assert record.id is not None

    by_id = persistence.get_market_profile_by_id(record.id)
    by_name = persistence.get_market_profile_by_name("P1")
    assert by_id.id == record.id
    assert by_name.id == record.id
    assert by_name.description == "test profile"

    provider = persistence.load_market_provider(record.id)
    assert isinstance(provider, MarketPriceProvider)
    assert provider.pmg_base_eur_per_kwh == pytest.approx(0.07)


def test_repo_list_update_delete(persistence):
    blob = _tiny_provider().to_config_dict()
    r1 = persistence.upsert_market_profile({"name": "A", "data": blob})
    r2 = persistence.upsert_market_profile({"name": "B", "data": blob})

    names = [p.name for p in persistence.list_market_profiles()]
    assert names == ["A", "B"]  # ordered by name

    # Rename A -> C.
    updated = persistence.update_market_profile(r1.id, {"name": "C"})
    assert updated.name == "C"

    # Rename clash is rejected.
    with pytest.raises(ValueError):
        persistence.update_market_profile(r2.id, {"name": "C"})

    assert persistence.delete_market_profile(r2.id) is True
    assert persistence.delete_market_profile(r2.id) is False
    assert [p.name for p in persistence.list_market_profiles()] == ["C"]


def test_load_market_provider_missing_returns_none(persistence):
    assert persistence.load_market_provider(99999) is None


# ── scenario_builder hydration ─────────────────────────────────────────────


def test_run_summary_market_block_has_price_profile_and_fuel(
    persistence, simple_scenario_data
):
    """The run summary's market block carries the hourly price profile + fuel
    prices the Dashboard 'Mercato' tab needs."""
    from sim_stochastic_pv.application import SimulationApplication

    grid = np.full((3, 2, 12, 24), 0.05)
    surface = PriceSurface(price_eur_per_kwh=grid, n_trajectories=3, n_years=2)
    provider = MarketPriceProvider(surface, pmg_base_eur_per_kwh=0.05)
    data = provider.to_config_dict(
        build_config={
            "gas_scenario": "tension",
            "gas_mu_drift_annual": 0.05,
            "co2_scenario": "high",
        }
    )
    rec = persistence.upsert_market_profile({"name": "M", "data": data})

    scenario = dict(simple_scenario_data)
    scenario["solar"]["pv_kwp"] = 6.0
    scenario["energy"]["pv_kwp"] = 6.0
    scenario["energy"]["inverter_p_ac_max_kw"] = 4.0
    scenario["market_profile_id"] = rec.id

    app = SimulationApplication(persistence=persistence)
    res = app.run_analysis(scenario_data=scenario, seed=7)
    m = res["market"]
    assert m is not None
    assert m["surface_n_years"] == 2
    assert np.array(m["price_profile_mean_eur_per_kwh"]).shape == (2, 12, 24)
    # tension gas mu = 55, +5%/yr drift, high CO2 mu = 100.
    assert m["gas_price_by_year_eur_per_mwh"][0] == pytest.approx(55.0)
    assert m["gas_price_by_year_eur_per_mwh"][1] == pytest.approx(57.75)
    assert m["co2_price_by_year_eur_per_ton"][0] == pytest.approx(100.0)


def test_build_default_market_provider_none_without_reference(persistence):
    assert build_default_market_provider({"foo": 1}, persistence) is None


def test_build_default_market_provider_none_without_persistence():
    # A reference but no persistence service (CLI / standalone) → quiet None.
    assert build_default_market_provider({"market_profile_id": 1}, None) is None


def test_build_default_market_provider_resolves_by_id(persistence):
    blob = _tiny_provider().to_config_dict()
    rec = persistence.upsert_market_profile({"name": "P", "data": blob})
    provider = build_default_market_provider(
        {"market_profile_id": rec.id}, persistence
    )
    assert isinstance(provider, MarketPriceProvider)
    assert provider.pmg_base_eur_per_kwh == pytest.approx(0.07)


def test_build_default_market_provider_resolves_by_name(persistence):
    blob = _tiny_provider().to_config_dict()
    persistence.upsert_market_profile({"name": "Mercato X", "data": blob})
    provider = build_default_market_provider(
        {"market_profile_name": "Mercato X"}, persistence
    )
    assert isinstance(provider, MarketPriceProvider)


def test_build_default_market_provider_raises_on_missing_id(persistence):
    with pytest.raises(ValueError):
        build_default_market_provider({"market_profile_id": 12345}, persistence)


def test_build_default_market_provider_raises_on_missing_name(persistence):
    with pytest.raises(ValueError):
        build_default_market_provider(
            {"market_profile_name": "does-not-exist"}, persistence
        )


# ── validation ─────────────────────────────────────────────────────────────


def test_validation_rejects_bad_market_profile_id():
    for bad in (0, -1, True, "x"):
        errs = validate_scenario({"market_profile_id": bad})
        assert any("market_profile_id must be a positive integer" in e for e in errs)


def test_validation_accepts_good_market_profile_id():
    errs = validate_scenario({"market_profile_id": 5})
    assert not any("market_profile_id" in e for e in errs)


def test_validation_rejects_empty_market_profile_name():
    errs = validate_scenario({"market_profile_name": "   "})
    assert any("market_profile_name must be a non-empty string" in e for e in errs)


# ── default profile seed ────────────────────────────────────────────────────


def test_seed_default_market_profile_idempotent_and_usable(
    sqlite_session_factory, persistence
):
    from sim_stochastic_pv.db.seeding import (
        _DEFAULT_MARKET_PROFILE_NAME,
        seed_market_profiles,
    )

    session = sqlite_session_factory()
    try:
        assert seed_market_profiles(session) == 1  # created
        assert seed_market_profiles(session) == 0  # idempotent
    finally:
        session.close()

    record = persistence.get_market_profile_by_name(_DEFAULT_MARKET_PROFILE_NAME)
    assert record is not None
    provider = persistence.load_market_provider(record.id)
    assert provider is not None
    assert provider.pmg_base_eur_per_kwh == pytest.approx(0.04)
    assert provider.price_surface.n_trajectories == 8
    # The hydrated surface is a usable, non-degenerate wholesale grid.
    assert provider.price_surface.price_eur_per_kwh.shape == (8, 20, 12, 24)
    assert float(provider.price_surface.price_eur_per_kwh.mean()) > 0.0
