from __future__ import annotations

import os

from sim_stochastic_pv import config


def test_get_database_url_prefers_postgres(monkeypatch):
    """Ensure PostgreSQL DSN (if provided) takes precedence over SQLite."""
    monkeypatch.setenv("POSTGRES_DSN", "postgresql+psycopg://user:pass@host/db")
    assert config.get_database_url() == "postgresql+psycopg://user:pass@host/db"


def test_get_database_url_falls_back_to_sqlite(monkeypatch, tmp_path):
    """Ensure SQLite path is used when no POSTGRES_DSN is defined."""
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    monkeypatch.setenv("SIM_PV_DB_PATH", str(tmp_path / "custom.db"))
    url = config.get_database_url()
    assert url.startswith("sqlite+pysqlite:///")
    assert "custom.db" in url
