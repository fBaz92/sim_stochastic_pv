from __future__ import annotations

import pytest
from pathlib import Path
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sim_stochastic_pv.db.session import Base  # noqa: E402
from sim_stochastic_pv.persistence import PersistenceService  # noqa: E402


@pytest.fixture()
def sqlite_session_factory():
    """Provide a session factory bound to an in-memory SQLite database."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )
    yield Session
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture()
def persistence(sqlite_session_factory):
    """Provide a PersistenceService bound to the temporary SQLite DB."""
    return PersistenceService(session_factory=sqlite_session_factory)


def _build_simple_scenario_data() -> dict:
    hourly_profile = [200.0] * 24
    home_profiles = []
    for idx in range(12):
        # Slightly vary each month to ensure stochastic generation behaves deterministically in tests
        month_profile = [value + idx * 5 for value in hourly_profile]
        home_profiles.append(month_profile)

    month_params = [
        {
            "avg_daily_kwh_per_kwp": 2.0 + idx * 0.1,
            "p_sunny": 0.5,
            "sunny_factor": 1.1,
            "cloudy_factor": 0.4,
        }
        for idx in range(12)
    ]

    return {
        "scenario_name": "test_minimal",
        "description": "Lightweight scenario for unit tests",
        "load_profile": {
            "home_profiles_w": home_profiles,
            "min_days_home": [1] * 12,
            "max_days_home": [2] * 12,
            "home_variation_percentiles": [-0.05, 0.05],
            "away_variation_percentiles": [-0.02, 0.02],
            "away_profile": "arera",
        },
        "solar": {
            "pv_kwp": 1.0,
            "degradation_per_year": 0.0,
            "month_params": month_params,
        },
        "energy": {
            "n_years": 2,
            "pv_kwp": 1.0,
            "battery_specs": {
                "capacity_kwh": 1.0,
                "cycles_life": 1000,
            },
            "n_batteries": 0,
            "inverter_p_ac_max_kw": 1.0,
        },
        "price": {
            "base_price_eur_per_kwh": 0.2,
            "annual_escalation": 0.01,
            "use_stochastic_escalation": False,
            "escalation_variation_percentiles": [-0.01, 0.01],
        },
        "economic": {
            "investment_eur": 500.0,
            "n_mc": 5,
        },
        "optimization": {
            "panel_count_options": [1],
            "battery_count_options": [0],
            "include_no_battery": True,
            "inverter_options": [
                {
                    "name": "Test Inverter 1kW",
                    "p_ac_max_kw": 1.0,
                    "p_dc_max_kw": 1.0,
                    "price_eur": 100.0,
                    "install_cost_eur": 50.0,
                }
            ],
            "panel_options": [
                {
                    "name": "Test Panel 400W",
                    "power_w": 400.0,
                    "price_eur": 80.0,
                }
            ],
            "battery_options": [],
        },
    }


@pytest.fixture()
def simple_scenario_data() -> dict:
    """Return a lightweight scenario definition used to speed up tests."""
    return _build_simple_scenario_data()
