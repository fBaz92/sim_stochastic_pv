"""
Database seeding system for initializing default data from JSON files.

Provides functions to populate the database with seed data on first run,
including solar profiles, hardware components, and example configurations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from sqlalchemy.orm import Session

from .models import (
    CableModel,
    InverterModel,
    PanelModel,
    ProtectionModel,
    SolarProfileModel,
)


def seed_solar_profiles(session: Session, seed_dir: Path) -> int:
    """
    Load solar profiles from JSON seed files into database.

    Scans the solar_profiles subdirectory for JSON files and inserts
    each as a new solar profile record. Skips profiles that already
    exist (based on unique name constraint).

    Args:
        session: Active SQLAlchemy session for database operations.
        seed_dir: Root directory containing seed_data/ folder.

    Returns:
        int: Number of new solar profiles inserted.

    Example:
        ```python
        from sim_stochastic_pv.db.session import SessionLocal
        from sim_stochastic_pv.db.seeding import seed_solar_profiles
        from pathlib import Path

        with SessionLocal() as session:
            seed_dir = Path(__file__).parent.parent / "seed_data"
            count = seed_solar_profiles(session, seed_dir)
            print(f"Inserted {count} solar profiles")
        ```

    Notes:
        - Expects JSON files in seed_dir/solar_profiles/*.json
        - Each JSON file must match SolarProfileModel schema
        - Existing profiles (by name) are skipped silently
        - All insertions committed at end (transactional)
    """
    solar_dir = seed_dir / "solar_profiles"
    if not solar_dir.exists():
        return 0

    count = 0
    for json_file in solar_dir.glob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            # Check if already exists (by unique name)
            existing = session.query(SolarProfileModel).filter_by(name=data["name"]).first()
            if existing:
                continue  # Skip if already seeded

            # Create and add new profile
            profile = SolarProfileModel(**data)
            session.add(profile)
            count += 1

        except Exception as e:
            print(f"Warning: Failed to load solar profile from {json_file.name}: {e}")
            continue

    # Commit all insertions
    session.commit()
    return count


def backfill_solar_profiles_new_columns(
    session: Session,
    seed_dir: Path | None = None,
) -> int:
    """
    Populate newly-added nullable columns on existing solar profile rows.

    Designed to keep databases that were created before a schema extension
    in sync with the latest seed JSON files. The function does *not* touch
    columns that are already filled — it only writes into the columns that
    appear in the seed JSON but are currently NULL on the DB row.

    Currently handled columns:
        - ``weather_persistence`` (list[float] of length 12)

    Args:
        session: Active SQLAlchemy session. The function commits at the end.
        seed_dir: Root directory containing ``seed_data/``. When omitted the
            package default (``sim_stochastic_pv/seed_data``) is used.

    Returns:
        int: Number of records that received at least one new value.

    Example:
        ```python
        from sim_stochastic_pv.db.session import SessionLocal
        from sim_stochastic_pv.db.seeding import backfill_solar_profiles_new_columns

        with SessionLocal() as session:
            touched = backfill_solar_profiles_new_columns(session)
            print(f"Backfilled {touched} solar profiles")
        ```

    Notes:
        - Idempotent: if every column is already populated, no UPDATE is run.
        - Records whose ``name`` is not present in the seed directory are
          left untouched (we cannot infer values for user-created sites).
        - Failures on a single file are logged and skipped — the function
          never raises so that ``init_db`` keeps booting the application.
    """
    if seed_dir is None:
        seed_dir = Path(__file__).parent.parent / "seed_data"

    solar_dir = seed_dir / "solar_profiles"
    if not solar_dir.exists():
        return 0

    # Build a lookup table {name: seed_dict} once
    seeds: Dict[str, Dict] = {}
    for json_file in solar_dir.glob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            if "name" in data:
                seeds[data["name"]] = data
        except Exception as exc:
            print(f"Warning: cannot read seed {json_file.name}: {exc}")

    if not seeds:
        return 0

    touched = 0
    for record in session.query(SolarProfileModel).all():
        seed_data = seeds.get(record.name)
        if seed_data is None:
            continue

        updated_this_row = False

        # weather_persistence: backfill only when currently NULL
        if (
            getattr(record, "weather_persistence", None) is None
            and "weather_persistence" in seed_data
        ):
            record.weather_persistence = seed_data["weather_persistence"]
            updated_this_row = True

        if updated_this_row:
            touched += 1

    if touched:
        session.commit()
    return touched


def seed_panels(session: Session, seed_dir: Path) -> int:
    """
    Load panel records from JSON seed files into the database.

    Phase 16 — every shipped panel JSON carries the full electrical
    datasheet (V_oc, V_mpp, temperature coefficients, NOCT, …) inside
    its ``specs`` blob so the MPPT-window model has realistic
    parameters out of the box.

    Args:
        session: Active SQLAlchemy session for database operations.
        seed_dir: Root directory containing ``panels/`` subfolder.

    Returns:
        Number of new panel records inserted (existing names skipped).
    """
    panels_dir = seed_dir / "panels"
    if not panels_dir.exists():
        return 0
    count = 0
    for json_file in panels_dir.glob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            existing = (
                session.query(PanelModel).filter_by(name=data.get("name")).first()
            )
            if existing:
                continue
            record = PanelModel(
                name=data.get("name"),
                manufacturer=data.get("manufacturer"),
                model_number=data.get("model_number"),
                power_w=data.get("power_w"),
                datasheet=data.get("datasheet"),
                specs=data.get("specs", data),
            )
            session.add(record)
            count += 1
        except Exception as exc:
            print(f"Warning: Failed to load panel from {json_file.name}: {exc}")
            continue
    session.commit()
    return count


def seed_inverters(session: Session, seed_dir: Path) -> int:
    """
    Load inverter records from JSON seed files into the database.

    Phase 16 — every shipped inverter JSON carries the full electrical
    datasheet (DC operating window, MPPT window, n_mppt_trackers, …)
    inside its ``specs`` blob.

    Args:
        session: Active SQLAlchemy session.
        seed_dir: Root directory containing ``inverters/`` subfolder.

    Returns:
        Number of new inverter records inserted (existing names skipped).
    """
    inverters_dir = seed_dir / "inverters"
    if not inverters_dir.exists():
        return 0
    count = 0
    for json_file in inverters_dir.glob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            existing = (
                session.query(InverterModel).filter_by(name=data.get("name")).first()
            )
            if existing:
                continue
            record = InverterModel(
                name=data.get("name"),
                manufacturer=data.get("manufacturer"),
                model_number=data.get("model_number"),
                nominal_power_kw=data.get("nominal_power_kw"),
                datasheet=data.get("datasheet"),
                specs=data.get("specs", data),
            )
            session.add(record)
            count += 1
        except Exception as exc:
            print(f"Warning: Failed to load inverter from {json_file.name}: {exc}")
            continue
    session.commit()
    return count


# Default electricity-market profile seeded on first run. Modest dimensions
# keep the one-off surface build fast (~2 s) and the JSON blob small while still
# giving the PV Monte Carlo a realistic hourly wholesale surface to value PV
# export against.
_DEFAULT_MARKET_PROFILE_NAME = "Italia (mercato base)"
_DEFAULT_MARKET_N_YEARS = 20
_DEFAULT_MARKET_N_TRAJECTORIES = 8
_DEFAULT_MARKET_PMG_EUR_PER_KWH = 0.04
_DEFAULT_MARKET_SEED = 42


def seed_market_profiles(session: Session) -> int:
    """
    Seed the default Italian electricity-market profile (idempotent).

    Builds a wholesale price surface once from the packaged Italian generation
    mix + base gas scenario, wraps it in a
    :class:`~sim_stochastic_pv.simulation.market_pricing.MarketPriceProvider`
    with a guaranteed minimum export price (PMG) of 0.04 EUR/kWh, and stores the
    whole provider configuration in a ``MarketProfileModel`` named
    "Italia (mercato base)".

    The surface is built on a few representative years and interpolated, so the
    one-off cost is a couple of seconds rather than a full per-year market Monte
    Carlo over the whole horizon.

    Args:
        session: Active SQLAlchemy session for database operations.

    Returns:
        int: 1 if the profile was created, 0 if it already existed.

    Notes:
        - Idempotent: returns 0 without rebuilding when the named profile is
          already present, so it is safe to call on every startup.
        - Imports the market engine lazily so the db layer carries no hard
          dependency on it at import time.
    """
    from .models import MarketProfileModel

    existing = (
        session.query(MarketProfileModel)
        .filter_by(name=_DEFAULT_MARKET_PROFILE_NAME)
        .first()
    )
    if existing is not None:
        return 0

    from ..market.config import GAS_SCENARIOS, ITALIAN_MIX
    from ..market.horizon import MixTrend, build_price_surface
    from ..simulation.market_pricing import MarketPriceProvider

    n_years = _DEFAULT_MARKET_N_YEARS
    representative_years = [0, n_years // 2, n_years - 1]
    trend = MixTrend(base_mix=ITALIAN_MIX)
    surface = build_price_surface(
        trend,
        GAS_SCENARIOS["base"],
        n_years=n_years,
        n_trajectories=_DEFAULT_MARKET_N_TRAJECTORIES,
        representative_years=representative_years,
        seed=_DEFAULT_MARKET_SEED,
    )
    provider = MarketPriceProvider(
        surface, pmg_base_eur_per_kwh=_DEFAULT_MARKET_PMG_EUR_PER_KWH
    )
    data = provider.to_config_dict(
        build_config={
            "mix": "italian",
            "gas_scenario": "base",
            "representative_years": representative_years,
            "n_trajectories": _DEFAULT_MARKET_N_TRAJECTORIES,
            "n_years": n_years,
            "seed": _DEFAULT_MARKET_SEED,
        }
    )
    record = MarketProfileModel(
        name=_DEFAULT_MARKET_PROFILE_NAME,
        description=(
            "Profilo di mercato elettrico italiano di default: superficie di "
            "prezzo all'ingrosso (mix base, scenario gas base) con ritiro "
            "dedicato a prezzo minimo garantito di 0.04 €/kWh."
        ),
        data=data,
    )
    session.add(record)
    session.commit()
    return 1


def seed_cables(session: Session, seed_dir: Path) -> int:
    """
    Seed the DC cable catalogue from ``seed_data/cables/*.json``.

    Each JSON file carries a ``cables`` list of catalogue rows (name,
    section, €/m, Iz). Existing names are skipped, so the function is
    idempotent and safe to run on every startup.

    Args:
        session: Active SQLAlchemy session.
        seed_dir: Root directory containing the ``cables/`` subfolder.

    Returns:
        Number of new cable records inserted.
    """
    cables_dir = seed_dir / "cables"
    if not cables_dir.exists():
        return 0
    count = 0
    for json_file in cables_dir.glob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                payload = json.load(f)
            for data in payload.get("cables", []):
                existing = (
                    session.query(CableModel).filter_by(name=data.get("name")).first()
                )
                if existing:
                    continue
                session.add(CableModel(
                    name=data["name"],
                    manufacturer=data.get("manufacturer"),
                    section_mm2=float(data["section_mm2"]),
                    material=data.get("material", "copper"),
                    price_eur_per_m=data.get("price_eur_per_m"),
                    iz_a=data.get("iz_a"),
                    notes=data.get("notes"),
                ))
                count += 1
        except Exception as exc:
            print(f"Warning: Failed to load cables from {json_file.name}: {exc}")
            continue
    session.commit()
    return count


def seed_protections(session: Session, seed_dir: Path) -> int:
    """
    Seed the DC protection catalogue from ``seed_data/protections/*.json``.

    Same contract as :func:`seed_cables` (idempotent, name-keyed skip).

    Args:
        session: Active SQLAlchemy session.
        seed_dir: Root directory containing the ``protections/`` subfolder.

    Returns:
        Number of new protection records inserted.
    """
    protections_dir = seed_dir / "protections"
    if not protections_dir.exists():
        return 0
    count = 0
    for json_file in protections_dir.glob("*.json"):
        try:
            with open(json_file, encoding="utf-8") as f:
                payload = json.load(f)
            for data in payload.get("protections", []):
                existing = (
                    session.query(ProtectionModel)
                    .filter_by(name=data.get("name"))
                    .first()
                )
                if existing:
                    continue
                session.add(ProtectionModel(
                    name=data["name"],
                    manufacturer=data.get("manufacturer"),
                    kind=data.get("kind", "fuse"),
                    rated_current_a=data.get("rated_current_a"),
                    rated_voltage_v=data.get("rated_voltage_v"),
                    price_eur=data.get("price_eur"),
                    specs=data.get("specs"),
                    notes=data.get("notes"),
                ))
                count += 1
        except Exception as exc:
            print(f"Warning: Failed to load protections from {json_file.name}: {exc}")
            continue
    session.commit()
    return count


def backfill_hardware_designer_specs(
    session: Session, seed_dir: Path | None = None
) -> int:
    """
    Merge the designer-only datasheet fields into existing hardware rows.

    Databases created before the electrical designer carry panels and
    inverters without the sizing fields (α coefficient, system voltage,
    max series fuse; per-MPPT current limits, full-load MPPT window).
    For every shipped seed component that already exists in the DB by
    name, this backfill copies into its ``specs`` blob **only the keys
    that are missing** — user edits are never overwritten.

    Args:
        session: Active SQLAlchemy session.
        seed_dir: Optional seed-data root (defaults to the packaged one).

    Returns:
        Number of hardware rows that received at least one new key.
    """
    if seed_dir is None:
        seed_dir = Path(__file__).parent.parent / "seed_data"

    touched = 0
    for folder, model in (("panels", PanelModel), ("inverters", InverterModel)):
        directory = seed_dir / folder
        if not directory.exists():
            continue
        for json_file in directory.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                record = (
                    session.query(model).filter_by(name=data.get("name")).first()
                )
                if record is None:
                    continue
                seed_specs = data.get("specs", {})
                specs = dict(record.specs or {})
                missing = {k: v for k, v in seed_specs.items() if k not in specs}
                if missing:
                    specs.update(missing)
                    record.specs = specs
                    touched += 1
            except Exception as exc:
                print(
                    f"Warning: designer-spec backfill failed for {json_file.name}: {exc}"
                )
                continue
    session.commit()
    return touched


def seed_database(session: Session, seed_dir: Path | None = None) -> Dict[str, int]:
    """
    Seed all database tables from JSON seed files.

    Orchestrates seeding of multiple entity types (solar profiles, hardware,
    etc.) from organized seed data directories. Returns counts of inserted
    records for each entity type.

    Args:
        session: Active SQLAlchemy session for database operations.
        seed_dir: Optional root directory containing seed_data/ folder.
                 Defaults to package-relative path ../seed_data/.

    Returns:
        dict: Mapping of entity type to count of inserted records.
              Example: {"solar_profiles": 5, "inverters": 0}

    Example:
        ```python
        from sim_stochastic_pv.db.session import SessionLocal
        from sim_stochastic_pv.db.seeding import seed_database

        with SessionLocal() as session:
            counts = seed_database(session)
            print(f"Database seeded: {counts}")
        ```

    Notes:
        - Auto-locates seed_data/ directory if seed_dir not provided
        - Safe to call multiple times (skips existing records)
        - Prints warning for any failed seed files
        - Future: Add hardware seeding (inverters, panels, batteries)
    """
    if seed_dir is None:
        # Default to seed_data/ in package root
        seed_dir = Path(__file__).parent.parent / "seed_data"

    counts = {
        "solar_profiles": seed_solar_profiles(session, seed_dir),
        # Phase 16 — ship a small but realistic catalog of panels and
        # inverters complete with electrical datasheet specs so the
        # MPPT-window model works out of the box.
        "panels": seed_panels(session, seed_dir),
        "inverters": seed_inverters(session, seed_dir),
        # Ship a default electricity-market profile so the dedicated-withdrawal
        # export valuation is available out of the box.
        "market_profiles": seed_market_profiles(session),
        # Electrical-designer catalogues: DC cables and protections.
        "cables": seed_cables(session, seed_dir),
        "protections": seed_protections(session, seed_dir),
        # Future expansion:
        # "batteries": seed_hardware_batteries(session, seed_dir),
    }

    return counts
