from __future__ import annotations

import os
from pathlib import Path
from typing import Dict


def _load_dotenv(path: str = ".env") -> Dict[str, str]:
    """
    Basic .env loader to populate os.environ when python-dotenv is unavailable.
    Returns a mapping of parsed key/value pairs.
    """
    env_path = Path(path)
    if not env_path.exists():
        return {}

    parsed: Dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        parsed[key] = value
    return parsed


_load_dotenv()


def get_database_url() -> str:
    """
    Determine the SQLAlchemy database URL, preferring PostgreSQL if configured.

    Returns:
        Database connection string compatible with SQLAlchemy.
    """
    dsn = os.getenv("POSTGRES_DSN")
    if dsn:
        return dsn

    db_path = Path(os.getenv("SIM_PV_DB_PATH", "sim_pv.db")).expanduser()
    if not db_path.is_absolute():
        db_path = Path.cwd() / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+pysqlite:///{db_path}"
