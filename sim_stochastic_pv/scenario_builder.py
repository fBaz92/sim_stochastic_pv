from __future__ import annotations

import numpy as np

import json
from pathlib import Path
from typing import Any, Mapping

from sim_stochastic_pv.simulation import (
    APPLIANCE_PRESETS,
    ApplianceEvent,
    ApplianceProfileConfig,
    AreraLoadProfile,
    BatteryOption,
    BatterySpecs,
    DEFAULT_DERATING_EXPONENT_K,
    DEFAULT_PHI_INTRA_DAY,
    DEFAULT_SIGMA_LOG,
    EconomicConfig,
    ElectricalModel,
    EnergySystemConfig,
    EscalatingPriceModel,
    GBMPriceModel,
    HeatPumpConfig,
    HouseThermalConfig,
    InflationConfig,
    InverterElectricalSpecs,
    InverterOption,
    LoadProfile,
    LoadScenarioBlueprint,
    MeanRevertingPriceModel,
    MonthlyAverageLoadProfile,
    OptimizationRequest,
    PanelElectricalSpecs,
    PanelOption,
    PriceModel,
    PvString,
    SetpointConfig,
    SolarModel,
    StochasticLoadConfig,
    TaxBonusConfig,
    ThermalLoadConfig,
    WeeklyPatternLoadProfile,
    WEEKLY_PRESETS,
    get_appliance_preset,
    make_default_solar_params_for_pavullo,
)
from sim_stochastic_pv.simulation.thermal import ThermalModel
DEFAULT_SCENARIO_PATH = Path(__file__).resolve().parent / "examples" / "home_away_default.json"


# Phase 9: default DC overcapacity ratio used by the "simplified" sizing mode.
# Picked as a middle-of-the-road residential value (oversize PV ~+20 % over the
# inverter's AC rating so the inverter sees its nominal power for more hours
# without paying for thermal mismatches on the panels).
DEFAULT_DC_OVERCAPACITY_PCT: float = 0.20


def simplified_panel_count(
    p_ac_max_kw: float,
    panel_power_w: float,
    target_dc_overcapacity_pct: float = DEFAULT_DC_OVERCAPACITY_PCT,
) -> int:
    """
    Smallest panel count that meets a given DC-overcapacity target.

    This is the helper that backs the "simplified" sizing mode (Phase 9):
    the user states *how much* DC overcapacity they want on top of the
    inverter's AC nameplate, and the simulator picks the minimum number
    of panels that satisfies the constraint:

        n_panels * panel_power_w  >=  p_ac_max_kw * 1000 * (1 + overcap)

    The result is rounded up — every additional panel beyond the
    threshold reduces the inverter clipping that would otherwise occur
    at peak irradiance.

    Args:
        p_ac_max_kw: Inverter AC nameplate power (kW).
        panel_power_w: Per-panel nominal STC power (W).
        target_dc_overcapacity_pct: Required excess DC power over the AC
            rating (decimal: 0.20 → 20 %). Defaults to
            :data:`DEFAULT_DC_OVERCAPACITY_PCT`. Must be non-negative.

    Returns:
        int: Number of panels (≥ 1).

    Raises:
        ValueError: When any input is non-positive or the overcapacity is
            below 0.

    Example:
        ```python
        >>> simplified_panel_count(5.0, 400.0, 0.20)  # 5 kW inverter, 400 W panels
        15
        >>> simplified_panel_count(3.0, 540.0)        # default 20 %, 540 W panels
        7
        ```

    Notes:
        - Returns 1 even for absurdly small inverters: in practice the
          user should validate that the resulting nominal DC power is
          reasonable for the inverter's MPPT range — but since the
          simulator does not model MPPT (see ``docs/electrical_simplifications.md``),
          there is no real "wrong" value here.
        - This helper is the only place that knows the overcapacity
          formula; both Python callers and the (future) UI go through
          here for behavioural consistency.
    """
    if p_ac_max_kw <= 0:
        raise ValueError(
            f"p_ac_max_kw must be positive, got {p_ac_max_kw}"
        )
    if panel_power_w <= 0:
        raise ValueError(
            f"panel_power_w must be positive, got {panel_power_w}"
        )
    if target_dc_overcapacity_pct < 0:
        raise ValueError(
            "target_dc_overcapacity_pct must be ≥ 0, "
            f"got {target_dc_overcapacity_pct}"
        )

    required_dc_w = p_ac_max_kw * 1000.0 * (1.0 + target_dc_overcapacity_pct)
    n = int(np.ceil(required_dc_w / panel_power_w))
    return max(1, n)


def load_scenario_data(source: str | Path | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """
    Load scenario data from JSON or return the provided mapping.

    Args:
        source: Path to a JSON file, mapping, or None for default example.

    Returns:
        Dictionary containing scenario configuration.
    """
    if source is None:
        path = DEFAULT_SCENARIO_PATH
        return json.loads(path.read_text(encoding="utf-8"))
    if isinstance(source, (str, Path)):
        path = Path(source)
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(source)


def _build_single_load_profile_factory(sub_cfg: Mapping[str, Any]) -> "callable":
    """
    Build a factory that produces one of the concrete LoadProfile types from a
    "sub-profile" dict (a single side — either home or away — of a richer
    profile, or a flat dict using the legacy keys).

    The accepted sub-config shapes are:

    1. ``{"type": "arera"}`` → returns :class:`AreraLoadProfile`.
    2. ``{"monthly_24h_w": [[24]*12]}`` → returns
       :class:`MonthlyAverageLoadProfile` with the full 12×24 matrix.
    3. ``{"monthly_w": [12 values]}`` → returns
       :class:`MonthlyAverageLoadProfile` with a flat monthly pattern
       expanded to 12×24 (each month constant across hours).
    4. ``{"type": "weekly", "weekly_pattern_w": [[...7 rows × 24...]],
         "monthly_24h_w": [...] | "monthly_w": [...] | "preset": name}``
       → returns :class:`WeeklyPatternLoadProfile`.
       The ``preset`` key selects one of :data:`WEEKLY_PRESETS` as the
       pattern; explicit ``weekly_pattern_w`` overrides it.  One of
       ``monthly_24h_w``, ``monthly_w``, or ``preset`` must provide the
       baseline.  When ``preset`` is the only source the pattern *and* a
       flat 200 W baseline are both taken from the preset name — this is
       only useful as a stand-alone profile; real scenarios should supply
       an explicit baseline.

    Args:
        sub_cfg: Sub-profile configuration dict.

    Returns:
        Zero-argument callable returning a fresh LoadProfile instance.

    Raises:
        ValueError: If none of the supported keys is present.
        ValueError: If ``type="weekly"`` is used but neither a pattern nor a
            recognised preset name is supplied.
        ValueError: If ``type="weekly"`` is used but no monthly baseline is
            supplied (and no preset name is given to derive a flat one from).
    """
    sub_type = str(sub_cfg.get("type", "")).lower()
    if sub_type == "arera":
        return lambda: AreraLoadProfile()

    # -- Phase 5: weekly pattern sub-profile ----------------------------------
    if sub_type == "weekly" or "weekly_pattern_w" in sub_cfg:
        # Resolve the modulation pattern.
        if "weekly_pattern_w" in sub_cfg:
            raw_pattern = np.array(sub_cfg["weekly_pattern_w"], dtype=float)
            if raw_pattern.shape != (7, 24):
                raise ValueError(
                    f"weekly_pattern_w must have shape (7, 24), got {raw_pattern.shape}"
                )
        elif "preset" in sub_cfg:
            preset_name = str(sub_cfg["preset"]).lower()
            if preset_name not in WEEKLY_PRESETS:
                raise ValueError(
                    f"Unknown weekly preset '{preset_name}'. "
                    f"Available: {sorted(WEEKLY_PRESETS)}"
                )
            raw_pattern = WEEKLY_PRESETS[preset_name]
        else:
            raise ValueError(
                "Sub-profile with type='weekly' requires either "
                "'weekly_pattern_w' (7×24 array) or 'preset' (name string)."
            )

        # Resolve the monthly baseline.
        if "monthly_24h_w" in sub_cfg:
            baseline = np.array(sub_cfg["monthly_24h_w"], dtype=float)
            if baseline.shape != (12, 24):
                raise ValueError(
                    f"monthly_24h_w must have shape (12, 24), got {baseline.shape}"
                )
        elif "monthly_w" in sub_cfg:
            flat = np.array(sub_cfg["monthly_w"], dtype=float)
            if flat.ndim != 1 or flat.size != 12:
                raise ValueError(
                    f"monthly_w must be a length-12 list, got shape {flat.shape}"
                )
            baseline = np.tile(flat[:, np.newaxis], (1, 24))
        else:
            raise ValueError(
                "Sub-profile with type='weekly' requires a monthly baseline: "
                "supply 'monthly_24h_w' (12×24) or 'monthly_w' (12 values)."
            )

        # Capture loop variables in default args to avoid closure capture.
        return lambda p=raw_pattern, b=baseline: WeeklyPatternLoadProfile(b, p)

    # -- Existing types -------------------------------------------------------
    if "monthly_24h_w" in sub_cfg:
        arr = np.array(sub_cfg["monthly_24h_w"], dtype=float)
        if arr.shape != (12, 24):
            raise ValueError(
                f"monthly_24h_w must have shape (12, 24), got {arr.shape}"
            )
        return lambda matrix=arr: MonthlyAverageLoadProfile(matrix)

    if "monthly_w" in sub_cfg:
        flat = np.array(sub_cfg["monthly_w"], dtype=float)
        if flat.ndim != 1 or flat.size != 12:
            raise ValueError(
                f"monthly_w must be a length-12 list, got shape {flat.shape}"
            )
        matrix = np.tile(flat[:, np.newaxis], (1, 24))
        return lambda m=matrix: MonthlyAverageLoadProfile(m)

    raise ValueError(
        "Sub-profile must contain one of: type='arera', type='weekly', "
        "monthly_24h_w, monthly_w"
    )


def _looks_like_phase8_single_side(load_cfg: Mapping[str, Any]) -> bool:
    """
    Detect whether a load-profile dict without ``kind`` field is actually a
    single-sided Phase-8 sub-profile (saved by the LoadProfileManager UI)
    rather than the legacy ``home_profiles_w`` / ``away_profiles_w`` shape.

    Returns ``True`` when any of these Phase-8 single-side markers is present
    at the root of ``load_cfg``:

    - ``monthly_w`` — flat 12-value monthly average.
    - ``monthly_24h_w`` — 12×24 monthly × hourly matrix.
    - ``weekly_pattern_w`` — 7×24 weekly pattern.
    - ``type`` ∈ {``arera``, ``weekly``} (case-insensitive).

    Returns ``False`` when the dict looks legacy (``home_profiles_w`` or
    ``away_profiles_w`` keys present) — the legacy branch handles those.

    The check is deliberately conservative: an *unknown* shape (no Phase-8
    marker, no legacy marker) returns ``False`` so the legacy branch fires
    its own informative error message. This keeps a single failure mode
    instead of two for malformed input.

    Args:
        load_cfg: The root dict of a load-profile payload.

    Returns:
        True if the dict should be routed to :func:`_build_single_load_profile_factory`.

    Example:
        ```python
        # Saved DB profile (UI "custom" mode):
        _looks_like_phase8_single_side({"monthly_w": [200]*12})       # True
        # Saved DB profile (UI "custom_24h" mode):
        _looks_like_phase8_single_side({"monthly_24h_w": [[200]*24]*12})  # True
        # Saved DB profile (UI "arera" mode):
        _looks_like_phase8_single_side({"type": "arera"})              # True
        # Legacy inline scenario:
        _looks_like_phase8_single_side({"home_profiles_w": [200]*12})  # False
        ```
    """
    phase8_keys = {"monthly_w", "monthly_24h_w", "weekly_pattern_w"}
    if any(key in load_cfg for key in phase8_keys):
        return True
    declared_type = str(load_cfg.get("type", "")).lower()
    if declared_type in ("arera", "weekly"):
        return True
    return False


def _legacy_side_factory(
    load_cfg: Mapping[str, Any],
    *,
    type_key: str,
    profiles_w_key: str,
    default_type: str,
) -> "callable":
    """
    Backward-compat factory builder for the legacy ``home_profiles_w`` /
    ``away_profiles_w`` schema.

    The legacy keys live at the root of the ``load_profile`` dict (no
    home/away container). This helper extracts and normalises them into a
    ``MonthlyAverageLoadProfile``.

    Accepted shapes for the values:
        - ``(12,)`` 12 monthly average watts
        - ``(24,)`` a single 24-hour pattern replicated 12 times
        - ``(12, 24)`` full monthly × hourly matrix

    Args:
        load_cfg: The full ``load_profile`` dict.
        type_key: Key holding the profile *type* (e.g. ``"home_profile_type"``).
        profiles_w_key: Key holding the array of watts.
        default_type: Default type when ``type_key`` is missing.

    Returns:
        Zero-arg callable producing a LoadProfile instance.

    Raises:
        ValueError: When the requested custom type lacks the watts array.
    """
    declared_type = load_cfg.get(type_key, default_type)
    declared_type_str = (
        declared_type.lower() if isinstance(declared_type, str) else ""
    )

    def factory() -> LoadProfile:
        if declared_type_str == "arera":
            return AreraLoadProfile()

        is_custom = declared_type_str in ("custom", "") or profiles_w_key in load_cfg
        if not is_custom:
            raise ValueError(f"Unsupported profile type: {declared_type}")

        if profiles_w_key not in load_cfg:
            raise ValueError(f"Missing '{profiles_w_key}' for custom profile.")

        arr = np.array(load_cfg[profiles_w_key], dtype=float)
        if arr.ndim == 1 and arr.size == 12:
            arr = np.tile(arr[:, np.newaxis], (1, 24))
        elif arr.ndim == 1 and arr.size == 24:
            arr = np.tile(arr, (12, 1))
        # Otherwise assume (12, 24)
        return MonthlyAverageLoadProfile(arr)

    return factory


def build_default_load_profile(scenario_data: Mapping[str, Any] | str | Path | None = None) -> LoadProfile:
    """
    Build the scenario's :class:`LoadProfile` from a hydrated scenario dict.

    Accepts two structural shapes for ``data["load_profile"]``:

    **New shape (Phase 8 — preferred)** — the load profile is a self-contained
    DB-friendly object that holds both home and away patterns:

    ```
    "load_profile": {
        "kind": "home_away",
        "home": {<sub-profile>},
        "away": {<sub-profile>}
    }
    ```

    A *sub-profile* is one of:

    - ``{"type": "arera"}`` — Italian ARERA standard profile
    - ``{"monthly_24h_w": [[…24…], …12…]}`` — explicit 12×24 matrix
    - ``{"monthly_w": [w0…w11]}`` — flat monthly pattern (expanded)

    In this shape ``min_days_home`` / ``max_days_home`` belong to the
    **scenario** (they describe how the user *uses* the building, not the
    building itself), so they are read from the scenario root first and
    fall back to inside the ``load_profile`` block for compatibility.

    **Legacy shape (still supported)** — flat dict with
    ``home_profile_type``/``away_profile`` selectors and
    ``home_profiles_w``/``away_profiles_w`` arrays at root level. Existing
    scenarios continue to work.

    Args:
        scenario_data: Hydrated scenario dict, JSON path, or ``None`` for
            the packaged example.

    Returns:
        LoadProfile: Fully wired ``HomeAwayLoadProfile`` instance ready for
        the simulator.

    Raises:
        ValueError: When the sub-profile shape is unrecognised or required
            arrays are missing.
    """
    data = load_scenario_data(scenario_data)
    load_cfg = data["load_profile"]

    # min_days_home / max_days_home: prefer scenario-level values
    # (new mental model — they belong to the scenario), fall back to inside
    # the load_profile block (legacy / inline scenarios).
    min_days_home = data.get(
        "min_days_home",
        load_cfg.get("min_days_home", [15] * 12),
    )
    max_days_home = data.get(
        "max_days_home",
        load_cfg.get("max_days_home", min_days_home),
    )
    home_var = tuple(
        data.get(
            "home_variation_percentiles",
            load_cfg.get("home_variation_percentiles", (-0.1, 0.1)),
        )
    )
    away_var = tuple(
        data.get(
            "away_variation_percentiles",
            load_cfg.get("away_variation_percentiles", (-0.05, 0.05)),
        )
    )

    load_kind = str(load_cfg.get("kind", "")).lower()

    if load_kind == "weekly":
        # Phase 5 — standalone weekly-pattern profile (no home/away split).
        # The scenario's occupancy parameters are ignored; the profile is
        # used as-is for all hours of the simulation.
        return _build_single_load_profile_factory(load_cfg)()

    if load_kind == "home_away":
        # Phase 8 schema — explicit home / away sub-profiles.
        if "home" not in load_cfg or "away" not in load_cfg:
            raise ValueError(
                "load_profile with kind='home_away' must contain "
                "both 'home' and 'away' sub-profile dicts"
            )
        home_factory = _build_single_load_profile_factory(load_cfg["home"])
        away_factory = _build_single_load_profile_factory(load_cfg["away"])
    elif _looks_like_phase8_single_side(load_cfg):
        # DB-saved single-sided profiles ("custom", "custom_24h",
        # "weekly", "arera") — the LoadProfileManager stores them as a
        # flat dict with the Phase-8 sub-profile keys at root level
        # (``monthly_w``, ``monthly_24h_w``, ``weekly_pattern_w``,
        # or ``type``). Treat the whole dict as the home sub-profile
        # and fall back to ARERA for the away side.
        home_factory = _build_single_load_profile_factory(load_cfg)
        away_factory = lambda: AreraLoadProfile()
    else:
        # Legacy inline scenario form (pre-Phase 8).
        home_factory = _legacy_side_factory(
            load_cfg,
            type_key="home_profile_type",
            profiles_w_key="home_profiles_w",
            default_type="custom",
        )
        away_factory = _legacy_side_factory(
            load_cfg,
            type_key="away_profile",
            profiles_w_key="away_profiles_w",
            default_type="arera",
        )

    load_blueprint = LoadScenarioBlueprint(
        home_profile_factory=home_factory,
        away_profile_factory=away_factory,
        min_days_home=min_days_home,
        max_days_home=max_days_home,
        home_variation_percentiles=home_var,
        away_variation_percentiles=away_var,
    )
    return load_blueprint.build_load_profile()


def build_default_solar_model(
    scenario_data: Mapping[str, Any] | str | Path | None = None,
    persistence=None,
) -> SolarModel:
    """
    Build SolarModel from scenario configuration with database fallback.

    Supports three ways to specify solar data (in priority order):
    1. Database reference by ID or name (requires persistence parameter)
    2. Inline month_params in scenario data
    3. Fallback to Pavullo defaults (always available)

    Args:
        scenario_data: Scenario configuration (JSON path, dict, or None for default).
        persistence: Optional PersistenceService for loading solar profiles from DB.
            Required if scenario data references solar_profile_id or solar_profile_name.

    Returns:
        SolarModel: Configured solar production model with orientation support.

    Example:
        ```python
        from sim_stochastic_pv.persistence import PersistenceService
        from sim_stochastic_pv.scenario_builder import build_default_solar_model

        # Load from database by name
        persistence = PersistenceService()
        solar_model = build_default_solar_model(
            scenario_data={"solar": {"solar_profile_name": "Milano"}},
            persistence=persistence
        )

        # Inline configuration (no database)
        solar_model = build_default_solar_model(
            scenario_data={"solar": {"month_params": [...]}}
        )

        # Fallback to Pavullo defaults
        solar_model = build_default_solar_model()
        ```

    Notes:
        - Pavullo defaults ALWAYS available as fallback (backward compatible)
        - Orientation parameters (tilt/azimuth) supported from config
        - Database loading requires persistence parameter
    """
    from .simulation.solar import SolarMonthParams

    data = load_scenario_data(scenario_data)
    solar_cfg = data["solar"]

    # Extract common parameters
    pv_kwp = solar_cfg.get("pv_kwp", 2.0)
    degradation_per_year = solar_cfg.get("degradation_per_year", 0.007)
    panel_tilt_degrees = solar_cfg.get("panel_tilt_degrees")
    panel_azimuth_degrees = solar_cfg.get("panel_azimuth_degrees")

    # Priority 1: Load from database by ID
    if "solar_profile_id" in solar_cfg and persistence:
        profile_id = solar_cfg["solar_profile_id"]
        profile = persistence.get_solar_profile_by_id(profile_id)
        if not profile:
            raise ValueError(f"Solar profile ID {profile_id} not found in database")
        return _solar_model_from_db_record(
            profile,
            pv_kwp,
            degradation_per_year,
            panel_tilt_degrees,
            panel_azimuth_degrees,
        )

    # Priority 2: Load from database by name
    if "solar_profile_name" in solar_cfg and persistence:
        name = solar_cfg["solar_profile_name"]
        profile = persistence.get_solar_profile_by_name(name)
        if not profile:
            raise ValueError(
                f"Solar profile '{name}' not found in database. "
                f"Available profiles: {', '.join(p.name for p in persistence.list_solar_profiles())}"
            )
        return _solar_model_from_db_record(
            profile,
            pv_kwp,
            degradation_per_year,
            panel_tilt_degrees,
            panel_azimuth_degrees,
        )

    # Priority 3: Inline month_params
    month_params_raw = solar_cfg.get("month_params")
    if month_params_raw is not None:
        month_params = [
            SolarMonthParams(
                avg_daily_kwh_per_kwp=entry["avg_daily_kwh_per_kwp"],
                p_sunny=entry["p_sunny"],
                sunny_factor=entry["sunny_factor"],
                cloudy_factor=entry["cloudy_factor"],
                weather_persistence=float(entry.get("weather_persistence", 0.0) or 0.0),
            )
            for entry in month_params_raw
        ]
        optimal_tilt = solar_cfg.get("optimal_tilt_degrees", 35.0)
        optimal_azimuth = solar_cfg.get("optimal_azimuth_degrees", 180.0)
        return SolarModel(
            pv_kwp=pv_kwp,
            month_params=month_params,
            degradation_per_year=degradation_per_year,
            optimal_tilt_degrees=optimal_tilt,
            optimal_azimuth_degrees=optimal_azimuth,
            panel_tilt_degrees=panel_tilt_degrees,
            panel_azimuth_degrees=panel_azimuth_degrees,
        )

    # Fallback: Pavullo defaults (PRESERVED - always available as fallback)
    month_params = make_default_solar_params_for_pavullo()
    return SolarModel(
        pv_kwp=pv_kwp,
        month_params=month_params,
        degradation_per_year=degradation_per_year,
        optimal_tilt_degrees=35.0,
        optimal_azimuth_degrees=180.0,
        panel_tilt_degrees=panel_tilt_degrees,
        panel_azimuth_degrees=panel_azimuth_degrees,
    )


def _solar_model_from_db_record(profile, pv_kwp, degradation_per_year, panel_tilt_degrees, panel_azimuth_degrees):
    """
    Create SolarModel from database solar profile record.

    Helper function to construct SolarModel from SolarProfileModel database record.

    Args:
        profile: SolarProfileModel database record.
        pv_kwp: PV system capacity in kWp.
        degradation_per_year: Annual degradation rate.
        panel_tilt_degrees: Actual panel tilt (None = use optimal from profile).
        panel_azimuth_degrees: Actual panel azimuth (None = use optimal from profile).

    Returns:
        SolarModel: Configured solar model with database profile data.
    """
    from .simulation.solar import SolarMonthParams

    # Build month_params from database arrays. ``weather_persistence`` is
    # nullable on legacy records and on profiles that predate the Markov
    # chain feature: in that case we substitute a per-month value of 0.0
    # which collapses the Markov chain to the legacy iid Bernoulli model.
    persistence_array = getattr(profile, "weather_persistence", None)
    if persistence_array is None:
        persistence_array = [0.0] * 12

    month_params = []
    for i in range(12):
        params = SolarMonthParams(
            avg_daily_kwh_per_kwp=profile.avg_daily_kwh_per_kwp[i],
            p_sunny=profile.p_sunny[i],
            sunny_factor=profile.sunny_factor,
            cloudy_factor=profile.cloudy_factor,
            weather_persistence=float(persistence_array[i] or 0.0),
        )
        month_params.append(params)

    # Create SolarModel with database profile's optimal orientation
    return SolarModel(
        pv_kwp=pv_kwp,
        month_params=month_params,
        degradation_per_year=degradation_per_year,
        optimal_tilt_degrees=profile.optimal_tilt_degrees,
        optimal_azimuth_degrees=profile.optimal_azimuth_degrees,
        panel_tilt_degrees=panel_tilt_degrees,
        panel_azimuth_degrees=panel_azimuth_degrees,
    )


def build_default_thermal_model(
    scenario_data: Mapping[str, Any] | str | Path | None = None,
    persistence=None,
) -> ThermalModel | None:
    """
    Resolve the Phase-15 :class:`ThermalModel` referenced by the scenario.

    Looks for a top-level ``climate_profile_id`` (preferred) or
    ``climate_profile_name`` key in the scenario JSON and asks the
    persistence service to hydrate the calibrated model. Returns
    ``None`` when neither key is present, when no persistence service
    is wired (CLI / standalone), or when the referenced profile does
    not exist in the database. The caller decides whether ``None`` is
    a fatal condition (Phase 16 ``mode='mppt_window'`` requires it) or
    a silent legacy fallback (default scenarios).

    Args:
        scenario_data: JSON path, dict, or ``None`` for the packaged
            example.
        persistence: Optional :class:`PersistenceService` used to fetch
            the climate profile. Required when the scenario contains
            ``climate_profile_id`` / ``climate_profile_name``.

    Returns:
        :class:`ThermalModel` instance ready for the simulator, or
        ``None`` when the scenario does not reference any climate
        profile or the lookup yields nothing.

    Raises:
        ValueError: When the referenced profile id/name does not
            resolve to a record (only when ``persistence`` is wired —
            a missing persistence service yields a quiet ``None``).
    """
    data = load_scenario_data(scenario_data)
    profile_id = data.get("climate_profile_id")
    profile_name = data.get("climate_profile_name")
    if profile_id is None and profile_name is None:
        return None
    if persistence is None:
        return None
    if profile_id is not None:
        thermal = persistence.load_thermal_model(int(profile_id))
        if thermal is None:
            raise ValueError(
                f"climate_profile_id={profile_id} not found in the database"
            )
        return thermal
    record = persistence.climate.get_climate_profile_by_name(str(profile_name))
    if record is None:
        raise ValueError(
            f"climate_profile_name={profile_name!r} not found in the database"
        )
    return persistence.load_thermal_model(record.id)


def _coerce_pv_string(raw: Mapping[str, Any]) -> PvString:
    """
    Parse one entry of ``electrical.pv_strings`` into a :class:`PvString`.

    All fields default to a single south-facing string on the first
    MPPT tracker, matching the simulator's legacy behaviour.
    """
    return PvString(
        n_panels=int(raw["n_panels"]),
        tilt_degrees=float(raw.get("tilt_degrees", raw.get("tilt", 35.0))),
        azimuth_degrees=float(raw.get("azimuth_degrees", raw.get("azimuth", 180.0))),
        mppt_id=int(raw.get("mppt_id", 0)),
    )


def build_default_electrical_model(
    scenario_data: Mapping[str, Any] | str | Path | None = None,
    *,
    energy_cfg: Mapping[str, Any] | None = None,
) -> ElectricalModel | None:
    """
    Build the Phase-16 :class:`ElectricalModel` from a scenario JSON.

    The function looks at ``scenario_data["electrical"]`` and returns
    ``None`` whenever the block is missing or sets
    ``mode = "off"`` — the simulator must then run the legacy
    byte-identical energy path. When ``mode = "mppt_window"`` the
    function pulls panel + inverter datasheet specs from inline
    ``panel`` / ``inverter`` sub-blocks (the wizard frontend writes
    them after the user picks hardware from the DB catalog), assembles
    a list of :class:`PvString` from ``pv_strings`` (or synthesises a
    single default string covering all the panels), and wires the
    whole thing into an :class:`ElectricalModel` ready for the
    simulator.

    Args:
        scenario_data: JSON path, dict, or ``None`` for the packaged
            example.
        energy_cfg: Optional pre-extracted ``data["energy"]`` block
            (avoids re-loading the scenario twice when called inside
            :func:`build_default_energy_config`).

    Returns:
        :class:`ElectricalModel` when the block opts in; ``None``
        otherwise.

    Raises:
        ValueError: When ``mode='mppt_window'`` but the required
            datasheet fields are missing or the layout is internally
            inconsistent (e.g. zero panels). The error message lists
            the missing fields so the user can fix the JSON.
    """
    data = load_scenario_data(scenario_data)
    elec_cfg = data.get("electrical")
    if not isinstance(elec_cfg, Mapping):
        return None
    mode = str(elec_cfg.get("mode", "off")).lower()
    if mode in ("off", "", "disabled"):
        return None
    if mode != "mppt_window":
        raise ValueError(
            f"electrical.mode={mode!r} not recognised. "
            "Valid values: 'off', 'mppt_window'."
        )

    panel_raw = elec_cfg.get("panel") or {}
    inverter_raw = elec_cfg.get("inverter") or {}
    panel_specs = PanelElectricalSpecs(
        power_w=panel_raw.get("power_w"),
        v_oc_stc_v=panel_raw.get("v_oc_stc_v"),
        v_mpp_stc_v=panel_raw.get("v_mpp_stc_v"),
        i_sc_stc_a=panel_raw.get("i_sc_stc_a"),
        i_mpp_stc_a=panel_raw.get("i_mpp_stc_a"),
        n_cells_series=panel_raw.get("n_cells_series"),
        beta_voc_pct_per_c=panel_raw.get("beta_voc_pct_per_c"),
        gamma_pmax_pct_per_c=panel_raw.get("gamma_pmax_pct_per_c"),
        noct_c=panel_raw.get("noct_c"),
    )
    inverter_specs = InverterElectricalSpecs(
        v_dc_min_v=inverter_raw.get("v_dc_min_v"),
        v_dc_max_v=inverter_raw.get("v_dc_max_v"),
        v_mppt_min_v=inverter_raw.get("v_mppt_min_v"),
        v_mppt_max_v=inverter_raw.get("v_mppt_max_v"),
        n_mppt_trackers=int(inverter_raw.get("n_mppt_trackers", 1)),
        i_dc_max_per_mppt_a=inverter_raw.get("i_dc_max_per_mppt_a"),
    )

    strings_raw = elec_cfg.get("pv_strings")
    if isinstance(strings_raw, list) and strings_raw:
        strings = [_coerce_pv_string(s) for s in strings_raw]
    else:
        # Synthesise a single default string covering all panels using
        # the energy block's pv_kwp and the panel nameplate.
        cfg = energy_cfg if energy_cfg is not None else data.get("energy", {})
        pv_kwp = float(cfg.get("pv_kwp", 0.0) or 0.0)
        panel_power_w = panel_specs.power_w
        if panel_power_w in (None, 0):
            raise ValueError(
                "electrical.mode='mppt_window' requires either "
                "electrical.pv_strings or a panel.power_w to derive "
                "the default single-string layout."
            )
        if pv_kwp <= 0:
            raise ValueError(
                "electrical.mode='mppt_window' requires energy.pv_kwp > 0 "
                "when electrical.pv_strings is not provided."
            )
        n_panels = max(1, int(round(pv_kwp * 1000.0 / panel_power_w)))
        # Tilt/azimuth taken from the solar block (PVGIS-aligned) or
        # default to optimal south-facing.
        solar_cfg = data.get("solar", {})
        tilt = float(solar_cfg.get("panel_tilt_degrees", 35.0))
        az = float(solar_cfg.get("panel_azimuth_degrees", 180.0))
        strings = [PvString(n_panels=n_panels, tilt_degrees=tilt, azimuth_degrees=az, mppt_id=0)]

    derating_exp = float(elec_cfg.get("derating_exponent_k", DEFAULT_DERATING_EXPONENT_K))
    n_years = int((energy_cfg or data.get("energy", {})).get("n_years", 20))
    return ElectricalModel(
        panel=panel_specs,
        inverter=inverter_specs,
        strings=strings,
        derating_exponent_k=derating_exp,
        n_years=n_years,
    )


def build_default_stochastic_load_config(
    scenario_data: Mapping[str, Any] | str | Path | None = None,
) -> StochasticLoadConfig | None:
    """
    Resolve the Phase-17 ``load_profile.stochastic`` sub-block.

    The block is opt-in. Absent block → ``None`` (legacy deterministic
    behaviour). Present block → :class:`StochasticLoadConfig`.

    Recognises both the canonical nested location
    (``load_profile.stochastic``) and a back-compat root-level key
    (``stochastic_load``) so the user can opt-in from a flat JSON if
    preferred.
    """
    data = load_scenario_data(scenario_data)
    raw: Mapping[str, Any] | None = None
    load_cfg = data.get("load_profile")
    if isinstance(load_cfg, Mapping):
        raw = load_cfg.get("stochastic")
    if raw is None:
        raw = data.get("stochastic_load")
    if not isinstance(raw, Mapping):
        return None
    return StochasticLoadConfig(
        enabled=bool(raw.get("enabled", False)),
        sigma_log=float(raw.get("sigma_log", DEFAULT_SIGMA_LOG)),
        phi_intra_day=float(raw.get("phi_intra_day", DEFAULT_PHI_INTRA_DAY)),
    )


def build_default_thermal_load_config(
    scenario_data: Mapping[str, Any] | str | Path | None = None,
) -> ThermalLoadConfig | None:
    """
    Resolve the Phase-17 ``thermal_load`` block into a runtime config.

    The block is opt-in. Returns ``None`` when missing or
    ``enabled=False`` (the simulator then skips the HVAC controller
    entirely). Otherwise hydrates the three nested dataclasses (house,
    heat_pump, setpoint).

    Args:
        scenario_data: JSON path / mapping / None.

    Returns:
        :class:`ThermalLoadConfig` or ``None``.

    Raises:
        ValueError: When ``thermal_load.enabled=True`` but a required
            sub-block contains a value that fails the dataclass
            ``__post_init__`` checks (delegated to the dataclass).
    """
    data = load_scenario_data(scenario_data)
    raw = data.get("thermal_load")
    if not isinstance(raw, Mapping):
        return None
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return None
    house_raw = raw.get("house") or {}
    hp_raw = raw.get("heat_pump") or {}
    sp_raw = raw.get("setpoint") or {}
    house = HouseThermalConfig(
        floor_area_m2=float(house_raw.get("floor_area_m2", 100.0)),
        insulation_preset=str(house_raw.get("insulation_preset", "standard")),
        ua_w_per_c_per_m2=(
            float(house_raw["ua_w_per_c_per_m2"])
            if house_raw.get("ua_w_per_c_per_m2") is not None
            else None
        ),
        capacitance_kwh_per_c_per_m2=float(
            house_raw.get("capacitance_kwh_per_c_per_m2", 0.05)
        ),
    )
    heat_pump = HeatPumpConfig(
        cop_heating=float(hp_raw.get("cop_heating", 3.5)),
        cop_cooling=float(hp_raw.get("cop_cooling", 3.0)),
        p_elec_max_kw=float(hp_raw.get("p_elec_max_kw", 3.0)),
    )
    setpoint = SetpointConfig(
        t_setpoint_heating_c=float(sp_raw.get("t_setpoint_heating_c", 20.0)),
        t_setpoint_cooling_c=float(sp_raw.get("t_setpoint_cooling_c", 26.0)),
        t_setpoint_away_c=(
            float(sp_raw["t_setpoint_away_c"])
            if sp_raw.get("t_setpoint_away_c") is not None
            else None
        ),
    )
    return ThermalLoadConfig(
        enabled=True,
        house=house,
        heat_pump=heat_pump,
        setpoint=setpoint,
        dynamic=bool(raw.get("dynamic", False)),
    )


def _resolve_appliance_item(
    raw: Mapping[str, Any],
    smart_pv_default: bool,
) -> ApplianceEvent:
    """
    Resolve one ``load_profile.appliances.items[]`` entry into an
    :class:`ApplianceEvent`.

    The entry may either reference a preset (``type: "washing_machine"``)
    or fully describe a custom appliance (``type: "custom"`` plus the
    full set of fields). Per-item overrides win over preset values
    (e.g. ``monthly_frequency_override`` lets the user replace just the
    monthly distribution without touching power or duration).

    Args:
        raw: One ``items[]`` entry from the scenario JSON.
        smart_pv_default: Global scenario default for ``schedule_mode``
            — used only when the entry omits the field explicitly.

    Returns:
        Fully populated :class:`ApplianceEvent` ready for the
        :class:`EventBasedApplianceProfile`.

    Raises:
        ValueError: When ``type`` is missing, references an unknown
            preset, or — for ``type="custom"`` — omits required fields.
    """
    type_str = str(raw.get("type", "")).strip().lower()
    if not type_str:
        raise ValueError("Each appliance item must include a 'type' field.")

    if type_str == "custom":
        # Custom appliance: every field must come from the JSON.
        try:
            base = ApplianceEvent(
                name=str(raw.get("name", "custom")),
                p_kw=float(raw["p_kw"]),
                duration_hours=float(raw["duration_hours"]),
                monthly_frequency=tuple(float(x) for x in raw["monthly_frequency"]),
                allowed_hours=tuple(int(h) for h in raw["allowed_hours"]),
                hour_of_day_weights=(
                    tuple(float(w) for w in raw["hour_of_day_weights"])
                    if raw.get("hour_of_day_weights") is not None
                    else None
                ),
                schedule_mode=raw.get(
                    "schedule_mode",
                    "smart_pv" if smart_pv_default else "naive_timer",
                ),
            )
        except KeyError as e:
            raise ValueError(
                f"appliance type='custom' is missing required field: {e.args[0]}"
            ) from None
        return base

    # Preset path: start from the catalog entry, then apply overrides.
    try:
        base = get_appliance_preset(type_str)
    except KeyError as e:
        raise ValueError(str(e)) from None

    # Build replacement keyword arguments for dataclasses.replace.
    schedule_mode = raw.get("schedule_mode")
    if schedule_mode is None:
        schedule_mode = "smart_pv" if smart_pv_default else base.schedule_mode

    overrides: dict[str, Any] = {"schedule_mode": schedule_mode}
    if "monthly_frequency_override" in raw:
        overrides["monthly_frequency"] = tuple(
            float(x) for x in raw["monthly_frequency_override"]
        )
    if "p_kw" in raw:
        overrides["p_kw"] = float(raw["p_kw"])
    if "duration_hours" in raw:
        overrides["duration_hours"] = float(raw["duration_hours"])
    if "allowed_hours" in raw:
        overrides["allowed_hours"] = tuple(int(h) for h in raw["allowed_hours"])
    if "hour_of_day_weights" in raw:
        overrides["hour_of_day_weights"] = (
            tuple(float(w) for w in raw["hour_of_day_weights"])
            if raw["hour_of_day_weights"] is not None
            else None
        )
    if "name" in raw:
        overrides["name"] = str(raw["name"])
    # ApplianceEvent is a frozen dataclass — rebuild via replace.
    from dataclasses import replace as _dc_replace
    return _dc_replace(base, **overrides)


def build_default_appliance_profile_config(
    scenario_data: Mapping[str, Any] | str | Path | None = None,
) -> ApplianceProfileConfig | None:
    """
    Resolve the Phase-17-bis ``load_profile.appliances`` block.

    Reads the canonical nested location (``load_profile.appliances``)
    and assembles a list of :class:`ApplianceEvent` from the preset
    catalog plus any per-item override. Returns ``None`` when the
    block is missing or ``enabled=false`` — the simulator then
    skips the appliance decorator and the legacy load path stays
    unchanged.

    Args:
        scenario_data: JSON path / dict / None.

    Returns:
        :class:`ApplianceProfileConfig` or ``None``.

    Raises:
        ValueError: When ``items`` references an unknown preset or
            omits required fields for ``type='custom'``.
    """
    data = load_scenario_data(scenario_data)
    load_cfg = data.get("load_profile")
    if not isinstance(load_cfg, Mapping):
        return None
    raw = load_cfg.get("appliances")
    if not isinstance(raw, Mapping):
        return None
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return None
    smart_pv_default = bool(raw.get("smart_pv", False))
    items_raw = raw.get("items")
    if not isinstance(items_raw, list) or not items_raw:
        return None
    appliances = tuple(
        _resolve_appliance_item(item, smart_pv_default) for item in items_raw
    )
    return ApplianceProfileConfig(
        enabled=True,
        smart_pv_default=smart_pv_default,
        appliances=appliances,
    )


def build_default_energy_config(
    scenario_data: Mapping[str, Any] | str | Path | None = None,
    persistence=None,
) -> EnergySystemConfig:
    data = load_scenario_data(scenario_data)
    energy_cfg = data["energy"]
    battery_specs_data = energy_cfg.get("battery_specs", {"capacity_kwh": 0.0, "cycles_life": 6000})
    battery_specs = BatterySpecs(
        capacity_kwh=battery_specs_data.get("capacity_kwh", 0.0),
        # Guard against explicit null stored in DB specs (treat null the same as
        # missing key — fall back to the BatterySpecs class default of 6000).
        cycles_life=battery_specs_data.get("cycles_life") or 6000,
    )
    # Phase 16 — optional detailed electrical model (opt-in via
    # ``electrical.mode='mppt_window'``). The companion ``ThermalModel``
    # is resolved from the scenario's ``climate_profile_id`` so the
    # MPPT-window logic has hourly T_ambient to work with. Both stay
    # ``None`` in legacy scenarios, preserving the byte-identical
    # pre-Phase-16 energy path.
    electrical_model = build_default_electrical_model(data, energy_cfg=energy_cfg)
    # Phase 17 — Thermal HVAC and Phase 16 — Electrical MPPT both
    # consume the same path-level ThermalModel. We resolve it once for
    # either feature requesting it.
    thermal_load_config = build_default_thermal_load_config(data)
    stochastic_load_config = build_default_stochastic_load_config(data)
    needs_thermal = electrical_model is not None or thermal_load_config is not None
    thermal_model = (
        build_default_thermal_model(data, persistence) if needs_thermal else None
    )
    if electrical_model is not None and thermal_model is None:
        raise ValueError(
            "electrical.mode='mppt_window' requires a climate_profile_id "
            "(Phase 15) so the simulator can compute T_cell from hourly "
            "ambient temperatures. Either set climate_profile_id at the "
            "scenario root or downgrade electrical.mode to 'off'."
        )
    if thermal_load_config is not None and thermal_model is None:
        raise ValueError(
            "thermal_load.enabled=true requires a climate_profile_id "
            "(Phase 15) so the HVAC controller has hourly T_ambient. "
            "Wire climate_profile_id at the scenario root or disable "
            "the thermal_load block."
        )

    appliance_profile_config = build_default_appliance_profile_config(data)

    return EnergySystemConfig(
        n_years=energy_cfg.get("n_years", 20),
        pv_kwp=energy_cfg.get("pv_kwp", 2.0),
        battery_specs=battery_specs,
        n_batteries=energy_cfg.get("n_batteries", 0),
        inverter_p_ac_max_kw=energy_cfg.get("inverter_p_ac_max_kw", 1.0),
        electrical_model=electrical_model,
        thermal_model=thermal_model,
        stochastic_load_config=stochastic_load_config,
        thermal_load_config=thermal_load_config,
        appliance_profile_config=appliance_profile_config,
    )


def build_default_price_model(
    use_stochastic_price: bool | None = None,
    escalation_percentiles: tuple[float, float] | None = None,
    scenario_data: Mapping[str, Any] | str | Path | None = None,
) -> PriceModel:
    """
    Build the electricity price model from a scenario configuration.

    Dispatches on ``price.model_type`` to instantiate one of the three
    available price models. When ``model_type`` is omitted the legacy
    :class:`EscalatingPriceModel` is returned for backward compatibility.

    Recognised ``model_type`` values (case-insensitive):

    - ``"escalating"`` (default) — deterministic escalation with iid jitter.
      Honours: ``base_price_eur_per_kwh``, ``annual_escalation``,
      ``use_stochastic_escalation``, ``escalation_variation_percentiles``.
    - ``"gbm"`` / ``"random_walk"`` — geometric Brownian motion in log-price.
      Honours: ``base_price_eur_per_kwh``, ``drift_annual``,
      ``volatility_annual``, ``seasonal_factors``.
    - ``"mean_reverting"`` / ``"ou"`` — Ornstein–Uhlenbeck in log-price.
      Honours: ``base_price_eur_per_kwh``, ``long_term_price_eur_per_kwh``,
      ``mean_reversion_speed_annual``, ``volatility_annual``,
      ``seasonal_factors``.

    Args:
        use_stochastic_price: Legacy kwarg, used only by the ``escalating``
            branch when ``use_stochastic_escalation`` is missing from the
            JSON. Ignored by GBM and OU.
        escalation_percentiles: Legacy kwarg for the ``escalating`` branch.
        scenario_data: JSON path, mapping, or ``None`` (defaults to the
            packaged example scenario).

    Returns:
        PriceModel: A configured concrete model ready to be passed to
        :class:`MonteCarloSimulator`.

    Raises:
        ValueError: If ``model_type`` is set to an unrecognised value.

    Example:
        ```python
        # GBM via JSON
        price_cfg = {"price": {
            "model_type": "gbm",
            "base_price_eur_per_kwh": 0.25,
            "drift_annual": 0.025,
            "volatility_annual": 0.10,
        }}
        model = build_default_price_model(scenario_data=price_cfg)
        ```
    """
    data = load_scenario_data(scenario_data)
    price_cfg = data["price"]
    model_type = str(price_cfg.get("model_type", "escalating")).lower()

    base_price = float(price_cfg.get("base_price_eur_per_kwh", 0.20))
    seasonal_factors = price_cfg.get("seasonal_factors")

    if model_type in ("escalating", "deterministic", "legacy"):
        return EscalatingPriceModel(
            base_price_eur_per_kwh=base_price,
            annual_escalation=float(price_cfg.get("annual_escalation", 0.02)),
            use_stochastic_escalation=price_cfg.get(
                "use_stochastic_escalation",
                True if use_stochastic_price is None else use_stochastic_price,
            ),
            escalation_variation_percentiles=tuple(
                price_cfg.get(
                    "escalation_variation_percentiles",
                    escalation_percentiles or (-0.05, 0.05),
                )
            ),
        )

    if model_type in ("gbm", "random_walk"):
        return GBMPriceModel(
            base_price_eur_per_kwh=base_price,
            drift_annual=float(price_cfg.get("drift_annual", 0.025)),
            volatility_annual=float(price_cfg.get("volatility_annual", 0.08)),
            seasonal_factors=seasonal_factors,
        )

    if model_type in ("mean_reverting", "ou", "ornstein_uhlenbeck"):
        return MeanRevertingPriceModel(
            base_price_eur_per_kwh=base_price,
            long_term_price_eur_per_kwh=float(
                price_cfg.get("long_term_price_eur_per_kwh", base_price)
            ),
            mean_reversion_speed_annual=float(
                price_cfg.get("mean_reversion_speed_annual", 0.30)
            ),
            volatility_annual=float(price_cfg.get("volatility_annual", 0.12)),
            seasonal_factors=seasonal_factors,
        )

    raise ValueError(
        f"Unknown price model_type '{model_type}'. "
        "Valid values: escalating, gbm, mean_reverting."
    )


def build_default_economic_config(
    n_mc: int | None = None,
    scenario_data: Mapping[str, Any] | str | Path | None = None,
) -> EconomicConfig:
    """
    Hydrate an EconomicConfig from a JSON scenario dictionary.

    Reads the ``economic`` block of the scenario and wires up the optional
    sub-blocks introduced in Phase 11:

    - ``tax_bonus`` → TaxBonusConfig. Absent or ``enabled=false`` keeps
      the bonus off and is a true no-op for the simulator.
    - ``inflation`` → InflationConfig. Absent falls back to the legacy
      scalar ``inflation_rate`` (also read here for compatibility with
      pre-Phase-11 JSONs).

    Args:
        n_mc: Optional override for the number of Monte Carlo paths.
            When provided wins over ``economic.n_mc``. When None the
            JSON value is used (with a fallback of 100).
        scenario_data: JSON scenario as dict / path / None. ``None`` uses
            the bundled default scenario at ``examples/home_away_default.json``.

    Returns:
        EconomicConfig fully populated with the inflation and tax-bonus
        sub-objects when present.
    """
    data = load_scenario_data(scenario_data)
    econ_cfg = data["economic"]

    legacy_inflation_rate = float(econ_cfg.get("inflation_rate", 0.025))
    inflation_cfg = _build_inflation_config(
        econ_cfg.get("inflation"), legacy_inflation_rate
    )
    tax_bonus_cfg = _build_tax_bonus_config(econ_cfg.get("tax_bonus"))

    return EconomicConfig(
        investment_eur=econ_cfg.get("investment_eur", 0.0),
        n_mc=n_mc or econ_cfg.get("n_mc", 100),
        inflation_rate=legacy_inflation_rate,
        inflation=inflation_cfg,
        tax_bonus=tax_bonus_cfg,
    )


def _build_inflation_config(
    raw: Mapping[str, Any] | None,
    legacy_inflation_rate: float,
) -> InflationConfig | None:
    """
    Translate the JSON ``economic.inflation`` block into an InflationConfig.

    Returns ``None`` when the block is absent AND the legacy scalar is
    the default (so the simulator stays in the legacy deterministic
    behaviour without allocating an object). Returns a full object when
    the JSON block is present, applying defaults for missing keys.
    """
    if raw is None:
        # Pre-Phase-11 scenarios with the legacy scalar fall back here.
        # We return None so the simulator picks up the scalar via
        # MonteCarloSimulator._resolve_inflation_config().
        return None
    mode = raw.get("mode", "deterministic")
    if mode not in ("deterministic", "stochastic"):
        raise ValueError(
            f"economic.inflation.mode must be 'deterministic' or 'stochastic', "
            f"got '{mode}'."
        )
    return InflationConfig(
        mode=mode,
        mean=float(raw.get("mean", legacy_inflation_rate)),
        std=float(raw.get("std", 0.0)),
        min_clip=float(raw.get("min_clip", -0.02)),
        max_clip=float(raw.get("max_clip", 0.10)),
    )


def _build_tax_bonus_config(
    raw: Mapping[str, Any] | None,
) -> TaxBonusConfig | None:
    """
    Translate the JSON ``economic.tax_bonus`` block into a TaxBonusConfig.

    Returns ``None`` when the block is absent (preserves legacy behaviour).
    """
    if raw is None:
        return None
    return TaxBonusConfig(
        enabled=bool(raw.get("enabled", False)),
        fraction_of_investment=float(raw.get("fraction_of_investment", 0.5)),
        duration_years=int(raw.get("duration_years", 10)),
    )


def _build_inverter(option: Mapping[str, Any]) -> InverterOption:
    battery_specs = option.get("integrated_battery_specs")
    specs_obj = (
        BatterySpecs(
            capacity_kwh=battery_specs.get("capacity_kwh", 0.0),
            cycles_life=battery_specs.get("cycles_life", 0),
        )
        if battery_specs
        else None
    )
    return InverterOption(
        name=option["name"],
        p_ac_max_kw=option["p_ac_max_kw"],
        p_dc_max_kw=option.get("p_dc_max_kw"),
        price_eur=option.get("price_eur", 0.0),
        install_cost_eur=option.get("install_cost_eur"),
        integrated_battery_specs=specs_obj,
        integrated_battery_price_eur=option.get("integrated_battery_price_eur"),
        integrated_battery_count_options=option.get("integrated_battery_count_options"),
        manufacturer=option.get("manufacturer"),
        model_number=option.get("model_number"),
        datasheet=option.get("datasheet"),
    )


def build_default_optimization_request(
    scenario_data: Mapping[str, Any] | str | Path | None = None,
) -> OptimizationRequest:
    """
    Build an :class:`OptimizationRequest` from a hydrated campaign config.

    Phase 9 — adds the ``sizing_mode`` switch in ``optimization``:

    - ``"advanced"`` (default, legacy behaviour): the campaign sweeps the
      ``panel_count_options`` list provided by the user verbatim. Useful
      when the user knows exactly which counts to try, or when a future
      MPPT model makes the count electrically meaningful.
    - ``"simplified"``: ``panel_count_options`` is **ignored**; the
      campaign computes the minimum panel count needed for each
      (inverter, panel) pair to meet the requested DC overcapacity
      (``optimization.target_dc_overcapacity_pct``, default
      :data:`DEFAULT_DC_OVERCAPACITY_PCT` = 0.20). The resulting set of
      counts (union across all pairs) is fed to the optimizer — most
      cross-product evaluations will still be correctly "sized" for at
      least one inverter/panel combo.

    The function does NOT validate semantic consistency between mode and
    options (e.g. providing ``panel_count_options`` in simplified mode):
    the extra options are simply ignored and a warning is implicit via
    the docstring. Strict validation lives in ``validation.py``.

    Args:
        scenario_data: Hydrated campaign config (JSON path, mapping, or
            ``None`` for the packaged example).

    Returns:
        OptimizationRequest: Ready for ``ScenarioOptimizer.run``.

    See also:
        :func:`simplified_panel_count` — the helper that derives a single
        count from inverter AC nameplate and panel power.
    """
    data = load_scenario_data(scenario_data)
    opt_cfg = data["optimization"]

    inverter_options = [_build_inverter(opt) for opt in opt_cfg.get("inverter_options", [])]

    panel_options = [
        PanelOption(
            name=opt["name"],
            power_w=opt.get("power_w", 0.0),
            price_eur=opt.get("price_eur", 0.0),
            manufacturer=opt.get("manufacturer"),
            model_number=opt.get("model_number"),
            datasheet=opt.get("datasheet"),
        )
        for opt in opt_cfg.get("panel_options", [])
    ]

    battery_options = [
        BatteryOption(
            name=opt["name"],
            specs=BatterySpecs(
                capacity_kwh=opt["specs"].get("capacity_kwh", 0.0),
                cycles_life=opt["specs"].get("cycles_life", 0),
            ),
            price_eur=opt.get("price_eur", 0.0),
            manufacturer=opt.get("manufacturer"),
            model_number=opt.get("model_number"),
            datasheet=opt.get("datasheet"),
        )
        for opt in opt_cfg.get("battery_options", [])
    ]

    # Phase 9: derive panel_count_options when in simplified sizing mode.
    sizing_mode = str(opt_cfg.get("sizing_mode", "advanced")).lower()
    if sizing_mode == "simplified" and inverter_options and panel_options:
        overcap = float(
            opt_cfg.get(
                "target_dc_overcapacity_pct", DEFAULT_DC_OVERCAPACITY_PCT
            )
        )
        derived_counts = set()
        for inv in inverter_options:
            for panel in panel_options:
                if panel.power_w <= 0 or inv.p_ac_max_kw <= 0:
                    continue
                derived_counts.add(
                    simplified_panel_count(
                        p_ac_max_kw=inv.p_ac_max_kw,
                        panel_power_w=panel.power_w,
                        target_dc_overcapacity_pct=overcap,
                    )
                )
        panel_count_options = sorted(derived_counts) if derived_counts else [1]
    else:
        panel_count_options = opt_cfg.get("panel_count_options", [1])

    return OptimizationRequest(
        scenario_name=data.get("scenario_name", "custom_scenario"),
        inverter_options=inverter_options,
        panel_options=panel_options,
        panel_count_options=panel_count_options,
        battery_options=battery_options,
        battery_count_options=opt_cfg.get("battery_count_options", [0]),
        include_no_battery=opt_cfg.get("include_no_battery", True),
    )
