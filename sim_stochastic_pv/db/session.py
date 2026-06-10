from __future__ import annotations

try:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import declarative_base, sessionmaker
except ImportError as exc:  # pragma: no cover - dependency guard
    raise RuntimeError(
        "SQLAlchemy must be installed to use the database features "
        "(pip install sqlalchemy)."
    ) from exc

from ..config import get_database_url

DATABASE_URL = get_database_url()

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    future=True,
)

Base = declarative_base()


def init_db() -> None:
    """
    Import models, create database tables, apply lightweight migrations, and
    seed default data on first run.

    Workflow:
    1. ``create_all`` materializes any missing table.
    2. ``_apply_lightweight_migrations`` adds nullable columns that were
       introduced after the initial schema (e.g. ``solar_profiles.weather_persistence``).
       The check is dialect-aware (SQLite / PostgreSQL) and safe to re-run.
    3. Seeding kicks in only if ``solar_profiles`` is still empty *or* if
       newly-introduced columns are unpopulated on existing rows.

    Example:
        ```python
        from sim_stochastic_pv.db.session import init_db

        # Initialize database and seed defaults
        init_db()
        ```

    Notes:
        - Idempotent: safe to call on every process start.
        - Lightweight migrations are intentionally minimal — for breaking
          schema changes adopt Alembic instead.
        - Seed files expected in package ../seed_data/ directory.
    """
    from . import models  # noqa: F401

    # Create all tables if they don't exist
    Base.metadata.create_all(bind=engine)

    # Apply lightweight ALTER TABLE migrations for nullable columns introduced
    # after the initial schema (e.g. solar_profiles.weather_persistence).
    _apply_lightweight_migrations()

    # Check if seeding needed (solar_profiles table empty)
    with SessionLocal() as session:
        from .models import SolarProfileModel

        count = session.query(SolarProfileModel).count()

        if count == 0:
            # First run - seed database with default data
            try:
                from .seeding import seed_database

                counts = seed_database(session)
                if any(counts.values()):
                    print(f"✅ Database seeded: {counts}")
            except Exception as e:
                print(f"⚠️  Warning: Database seeding failed: {e}")
                print("   You can manually add solar profiles using: pv-sim solar upsert <file>")
        else:
            # Existing DB: backfill new columns on legacy rows from seed JSON
            try:
                from .seeding import backfill_solar_profiles_new_columns

                touched = backfill_solar_profiles_new_columns(session)
                if touched:
                    print(
                        f"✅ Backfilled {touched} solar profile(s) with newly-added columns"
                    )
            except Exception as e:
                print(f"⚠️  Warning: backfill of new columns failed: {e}")

    # Ensure the default electricity-market profile exists even on databases
    # created before this feature: the solar-profile seed gate above only fires
    # on a brand-new DB, so an existing install would otherwise never get it.
    # ``seed_market_profiles`` is idempotent (a quick name lookup) and only
    # pays the surface-build cost the first time it actually inserts.
    with SessionLocal() as session:
        try:
            from .seeding import seed_market_profiles

            if seed_market_profiles(session):
                print("✅ Seeded default market profile 'Italia (mercato base)'")
        except Exception as e:  # pragma: no cover - defensive: don't break startup
            print(f"⚠️  Warning: market profile seeding failed: {e}")


def _apply_lightweight_migrations() -> None:
    """
    Apply non-destructive ``ALTER TABLE`` statements for nullable columns that
    were added after the initial schema.

    The function inspects the live database schema and emits an
    ``ALTER TABLE ... ADD COLUMN ...`` statement only for columns that are
    declared on the SQLAlchemy model but missing from the physical table.
    This avoids the need for a full migration framework while supporting the
    incremental evolution of the model.

    Currently handled:
        - ``solar_profiles.weather_persistence`` (JSON, nullable)
        - ``solar_profiles.location_id`` / ``climate_profiles.location_id``
          (INTEGER FK to ``locations``, nullable)
        - ``run_results.archived_at`` (TIMESTAMP, nullable)

    Notes:
        - Operates transparently on both SQLite and PostgreSQL via the
          dialect-neutral inspector exposed by SQLAlchemy.
        - Does NOT remove columns, change types, or backfill data. Backfill is
          performed afterwards by the seeding module.
        - Failures are logged but never raised: the application must keep
          starting even when the DB is read-only or the user lacks DDL grants.
    """
    from sqlalchemy import inspect, text

    try:
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())

        if "solar_profiles" in tables:
            existing_columns = {
                col["name"] for col in inspector.get_columns("solar_profiles")
            }
            if "weather_persistence" not in existing_columns:
                with engine.begin() as connection:
                    # Both SQLite and PostgreSQL accept the JSON type literal here.
                    connection.execute(
                        text(
                            "ALTER TABLE solar_profiles ADD COLUMN weather_persistence JSON"
                        )
                    )
                print("✅ Migrated solar_profiles: added column 'weather_persistence'")

        # Locations become first-class entities: solar and climate profiles
        # gain a nullable FK to their owning site. Legacy rows keep NULL and
        # behave exactly as before (the simulator never reads this column).
        for table in ("solar_profiles", "climate_profiles"):
            if table in tables:
                existing_columns = {
                    col["name"] for col in inspector.get_columns(table)
                }
                if "location_id" not in existing_columns:
                    with engine.begin() as connection:
                        connection.execute(
                            text(
                                f"ALTER TABLE {table} ADD COLUMN location_id INTEGER "
                                f"REFERENCES locations(id)"
                            )
                        )
                    print(f"✅ Migrated {table}: added column 'location_id'")

        # Phase 12 — add the soft-archive timestamp to run_results.
        if "run_results" in tables:
            existing_columns = {
                col["name"] for col in inspector.get_columns("run_results")
            }
            if "archived_at" not in existing_columns:
                with engine.begin() as connection:
                    # TIMESTAMP / DATETIME both work across SQLite & Postgres.
                    connection.execute(
                        text("ALTER TABLE run_results ADD COLUMN archived_at TIMESTAMP")
                    )
                print("✅ Migrated run_results: added column 'archived_at'")
    except Exception as exc:  # pragma: no cover - defensive: don't break startup
        print(f"⚠️  Warning: lightweight migration skipped: {exc}")
