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

from .models import SolarProfileModel


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
        # Future expansion:
        # "inverters": seed_hardware_inverters(session, seed_dir),
        # "panels": seed_hardware_panels(session, seed_dir),
        # "batteries": seed_hardware_batteries(session, seed_dir),
    }

    return counts
