from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

_logger = logging.getLogger(__name__)


def _load_dotenv(path: str = ".env", fallback: str = ".env.example") -> Dict[str, str]:
    """
    Populate ``os.environ`` from a dotenv file, with a logged fallback.

    The single source of truth for configuration is ``.env``. When it is
    absent the loader falls back to ``fallback`` (the committed
    ``.env.example`` template) and logs a warning so the substitution is
    visible in the logs. When neither file exists nothing is loaded.

    Existing environment variables always win (values are applied with
    :meth:`os.environ.setdefault`), so an explicit shell/Docker env overrides
    the file.

    Args:
        path: Primary dotenv file. Defaults to ``.env``.
        fallback: File used when ``path`` is missing. Defaults to
            ``.env.example``.

    Returns:
        Mapping of the key/value pairs parsed from whichever file was used
        (empty when neither exists).
    """
    env_path = Path(path)
    if not env_path.exists():
        fallback_path = Path(fallback)
        if not fallback_path.exists():
            return {}
        _logger.warning(
            "Config: '%s' non trovato — uso il fallback '%s' per le variabili "
            "d'ambiente. Copia '%s' in '%s' per personalizzarle.",
            path, fallback, fallback, path,
        )
        env_path = fallback_path

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
