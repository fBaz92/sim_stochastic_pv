"""
Repository + serialization helpers for :class:`ClimateProfileModel`
(Phase 15).

Acts as the bridge between the runtime objects in
:mod:`sim_stochastic_pv.simulation.thermal` (frozen dataclasses + the
:class:`ThermalModel` class) and their JSON-backed persistence in the DB.

The serialization format is intentionally explicit so future schema
changes can be detected via a missing key + the absence of a version
field is treated as ``v1``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from ..db.models import ClimateProfileModel
from ..simulation.thermal import (
    GPDTail,
    HarmonicSeasonalMean,
    ThermalModel,
    ThermalMonthParams,
)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_gpd_tail(tail: GPDTail | None) -> dict[str, float] | None:
    """Return a JSON-friendly dict for a :class:`GPDTail` or ``None``."""
    if tail is None:
        return None
    return {
        "threshold": float(tail.threshold),
        "shape": float(tail.shape),
        "scale": float(tail.scale),
        "exceedance_prob": float(tail.exceedance_prob),
    }


def deserialize_gpd_tail(blob: dict[str, Any] | None) -> GPDTail | None:
    """Round-trip back to :class:`GPDTail` or return ``None`` when missing."""
    if not blob:
        return None
    return GPDTail(
        threshold=float(blob["threshold"]),
        shape=float(blob["shape"]),
        scale=float(blob["scale"]),
        exceedance_prob=float(blob["exceedance_prob"]),
    )


def serialize_thermal_model(model: ThermalModel) -> dict[str, Any]:
    """
    Convert a :class:`ThermalModel` into a JSON-friendly dict.

    The output shape:

        {
          "harmonic": {"a0": ..., "a1": ..., "a2": ...},
          "monthly_params": [
            {
              "t_std_residual_c": ..., "persistence_phi": ...,
              "t_amplitude_c": ...,
              "gpd_upper": {...} | null,
              "gpd_lower": {...} | null
            },
            ... 12 entries total ...
          ],
          "climate_trend_c_per_year": ...
        }
    """
    return {
        "harmonic": {
            "a0": float(model.harmonic.a0),
            "a1": float(model.harmonic.a1),
            "a2": float(model.harmonic.a2),
        },
        "monthly_params": [
            {
                "t_std_residual_c": float(p.t_std_residual_c),
                "persistence_phi": float(p.persistence_phi),
                "t_amplitude_c": float(p.t_amplitude_c),
                "amp_slope_per_c": float(p.amp_slope_per_c),
                "gpd_upper": serialize_gpd_tail(p.gpd_upper),
                "gpd_lower": serialize_gpd_tail(p.gpd_lower),
            }
            for p in model.monthly_params
        ],
        "climate_trend_c_per_year": float(model.climate_trend_c_per_year),
    }


def deserialize_thermal_model(blob: dict[str, Any]) -> ThermalModel:
    """
    Round-trip :func:`serialize_thermal_model` back to a runtime
    :class:`ThermalModel`.

    Raises:
        KeyError: If a required field is missing — the caller should
            then re-fit from Open-Meteo via
            :func:`calibrate_thermal_model`.
    """
    harmonic = HarmonicSeasonalMean(
        a0=float(blob["harmonic"]["a0"]),
        a1=float(blob["harmonic"]["a1"]),
        a2=float(blob["harmonic"]["a2"]),
    )
    monthly_params = [
        ThermalMonthParams(
            t_std_residual_c=float(entry["t_std_residual_c"]),
            persistence_phi=float(entry["persistence_phi"]),
            t_amplitude_c=float(entry["t_amplitude_c"]),
            gpd_upper=deserialize_gpd_tail(entry.get("gpd_upper")),
            gpd_lower=deserialize_gpd_tail(entry.get("gpd_lower")),
            # Legacy profiles predate the clear-sky coupling: 0 keeps the
            # constant-amplitude behaviour bit-identical.
            amp_slope_per_c=float(entry.get("amp_slope_per_c", 0.0)),
        )
        for entry in blob["monthly_params"]
    ]
    return ThermalModel(
        harmonic=harmonic,
        monthly_params=monthly_params,
        climate_trend_c_per_year=float(blob.get("climate_trend_c_per_year", 0.0)),
    )


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class ClimateProfileRepository:
    """
    CRUD operations for :class:`ClimateProfileModel`.

    Mirrors the shape of :class:`SolarProfileRepository` so the
    :class:`PersistenceService` facade can expose them symmetrically.
    """

    def __init__(self, session_factory):
        """Args:
            session_factory: SQLAlchemy ``sessionmaker``.
        """
        self._session_factory = session_factory

    # ------------------------------------------------------------------- CRUD

    def upsert_climate_profile(self, data: dict[str, Any]) -> ClimateProfileModel:
        """
        Insert or update a climate profile by ``name``.

        Args:
            data: dict with keys
                ``{"name", "location_name", "latitude", "longitude",
                "harmonic", "monthly_params"}`` and optional
                ``{"elevation_m", "source", "climate_trend_c_per_year",
                "lookback_window", "notes"}``.

        Returns:
            The created / updated :class:`ClimateProfileModel`.
        """
        with self._session_factory() as session:
            existing = (
                session.query(ClimateProfileModel)
                .filter_by(name=data["name"])
                .first()
            )
            if existing:
                for key, value in data.items():
                    setattr(existing, key, value)
                record = existing
            else:
                record = ClimateProfileModel(**data)
                session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def list_climate_profiles(self) -> list[ClimateProfileModel]:
        with self._session_factory() as session:
            return (
                session.query(ClimateProfileModel)
                .order_by(ClimateProfileModel.name)
                .all()
            )

    def get_climate_profile_by_id(self, profile_id: int) -> ClimateProfileModel | None:
        with self._session_factory() as session:
            return session.get(ClimateProfileModel, profile_id)

    def get_climate_profile_by_name(self, name: str) -> ClimateProfileModel | None:
        with self._session_factory() as session:
            return (
                session.query(ClimateProfileModel)
                .filter_by(name=name)
                .first()
            )

    def update_climate_profile(
        self, profile_id: int, data: dict[str, Any]
    ) -> ClimateProfileModel | None:
        """
        Update an existing climate profile by primary key (allows rename).

        Only the keys present in ``data`` are written to the record. The
        calibrated payload (``harmonic``, ``monthly_params``,
        ``climate_trend_c_per_year``) can be edited via this method too
        but is normally rebuilt by re-running calibration.

        Args:
            profile_id: Primary key of the profile to update.
            data: Mapping of new column values. Allowed keys are any
                column on :class:`ClimateProfileModel`.

        Returns:
            Updated :class:`ClimateProfileModel`, or ``None`` if no such
            ID exists.

        Raises:
            ValueError: If the new ``name`` clashes with another record.
        """
        if not data:
            return self.get_climate_profile_by_id(profile_id)
        with self._session_factory() as session:
            record = session.get(ClimateProfileModel, profile_id)
            if record is None:
                return None
            new_name = data.get("name")
            if new_name and new_name != record.name:
                clash = (
                    session.query(ClimateProfileModel)
                    .filter(ClimateProfileModel.name == new_name)
                    .first()
                )
                if clash is not None and clash.id != profile_id:
                    raise ValueError(
                        f"Climate profile name '{new_name}' is already used by id={clash.id}"
                    )
            for key, value in data.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            session.commit()
            session.refresh(record)
            return record

    def delete_climate_profile(self, profile_id: int) -> bool:
        with self._session_factory() as session:
            record = session.get(ClimateProfileModel, profile_id)
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True

    # ----------------------------------------------------------------- helper

    def load_thermal_model(self, profile_id: int) -> ThermalModel | None:
        """Convenience: fetch by id and return a runtime
        :class:`ThermalModel`, or ``None`` if not found."""
        record = self.get_climate_profile_by_id(profile_id)
        if record is None:
            return None
        blob = {
            "harmonic": record.harmonic,
            "monthly_params": record.monthly_params,
            "climate_trend_c_per_year": record.climate_trend_c_per_year,
        }
        return deserialize_thermal_model(blob)
