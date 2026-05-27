from __future__ import annotations

import numpy as np

import json
from pathlib import Path
from typing import Any, Mapping

from sim_stochastic_pv.simulation import (
    AreraLoadProfile,
    BatteryOption,
    BatterySpecs,
    EconomicConfig,
    EnergySystemConfig,
    EscalatingPriceModel,
    GBMPriceModel,
    InverterOption,
    LoadProfile,
    LoadScenarioBlueprint,
    MeanRevertingPriceModel,
    MonthlyAverageLoadProfile,
    OptimizationRequest,
    PanelOption,
    PriceModel,
    SolarModel,
    make_default_solar_params_for_pavullo,
)
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

    Args:
        sub_cfg: Sub-profile configuration dict.

    Returns:
        Zero-argument callable returning a fresh LoadProfile instance.

    Raises:
        ValueError: If none of the supported keys is present.
    """
    sub_type = str(sub_cfg.get("type", "")).lower()
    if sub_type == "arera":
        return lambda: AreraLoadProfile()

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
        "Sub-profile must contain one of: type='arera', monthly_24h_w, monthly_w"
    )


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

    if str(load_cfg.get("kind", "")).lower() == "home_away":
        # Phase 8 schema — explicit home / away sub-profiles.
        if "home" not in load_cfg or "away" not in load_cfg:
            raise ValueError(
                "load_profile with kind='home_away' must contain "
                "both 'home' and 'away' sub-profile dicts"
            )
        home_factory = _build_single_load_profile_factory(load_cfg["home"])
        away_factory = _build_single_load_profile_factory(load_cfg["away"])
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


def build_default_energy_config(scenario_data: Mapping[str, Any] | str | Path | None = None) -> EnergySystemConfig:
    data = load_scenario_data(scenario_data)
    energy_cfg = data["energy"]
    battery_specs_data = energy_cfg.get("battery_specs", {"capacity_kwh": 0.0, "cycles_life": 0})
    battery_specs = BatterySpecs(
        capacity_kwh=battery_specs_data.get("capacity_kwh", 0.0),
        cycles_life=battery_specs_data.get("cycles_life", 0),
    )
    return EnergySystemConfig(
        n_years=energy_cfg.get("n_years", 20),
        pv_kwp=energy_cfg.get("pv_kwp", 2.0),
        battery_specs=battery_specs,
        n_batteries=energy_cfg.get("n_batteries", 0),
        inverter_p_ac_max_kw=energy_cfg.get("inverter_p_ac_max_kw", 1.0),
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
    data = load_scenario_data(scenario_data)
    econ_cfg = data["economic"]
    return EconomicConfig(
        investment_eur=econ_cfg.get("investment_eur", 0.0),
        n_mc=n_mc or econ_cfg.get("n_mc", 100),
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
