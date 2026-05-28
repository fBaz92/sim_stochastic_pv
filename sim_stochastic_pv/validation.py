"""
Configuration validation for scenarios and optimizations.
"""

from __future__ import annotations

from typing import Any, Dict, List


def validate_scenario(data: Dict[str, Any]) -> List[str]:
    """
    Validate scenario configuration after hydration.

    Checks for required fields and reasonable values to catch configuration
    errors early with clear error messages.

    Args:
        data: Hydrated scenario configuration dict.

    Returns:
        List of error messages. Empty list if valid.

    Example:
        ```python
        errors = validate_scenario(hydrated_config)
        if errors:
            print("Configuration errors:")
            for err in errors:
                print(f"  - {err}")
        ```
    """
    errors = []

    # Check for required top-level sections
    required_sections = ["load_profile", "solar", "energy", "economic", "price"]
    for section in required_sections:
        if section not in data:
            errors.append(f"Missing required section: '{section}'")

    # Validate load_profile
    # Accepted forms:
    #   {"type": "home_away", ...}           – schema-driven form
    #   {"home_profiles_w": [...], ...}       – raw profiles form
    #   {"home_profile_type": "arera", ...}   – shorthand type form
    _LP_ALT_KEYS = {"home_profiles_w", "home_profile_type", "monthly_avg_kwh"}
    if "load_profile" in data:
        if not isinstance(data["load_profile"], dict):
            errors.append("load_profile must be a dict/object")
        elif "type" not in data["load_profile"] and not _LP_ALT_KEYS.intersection(data["load_profile"]):
            errors.append(
                "load_profile.type is required "
                "(or provide home_profiles_w / home_profile_type / monthly_avg_kwh)"
            )

    # Validate solar model
    # Accepted forms:
    #   {"type": "default", ...}   – schema-driven form
    #   {"pv_kwp": 3.5, ...}       – direct parameter form (scenario_builder native)
    if "solar" in data:
        if not isinstance(data["solar"], dict):
            errors.append("solar must be a dict/object")
        elif "type" not in data["solar"] and "pv_kwp" not in data["solar"]:
            errors.append(
                "solar.type is required (or provide pv_kwp directly)"
            )

    # Validate energy system
    if "energy" in data:
        if not isinstance(data["energy"], dict):
            errors.append("energy must be a dict/object")
        else:
            # Check critical energy parameters
            if "pv_kwp" not in data["energy"]:
                errors.append("energy.pv_kwp is required (PV system size)")
            elif not isinstance(data["energy"]["pv_kwp"], (int, float)):
                errors.append("energy.pv_kwp must be a number")
            elif data["energy"]["pv_kwp"] <= 0:
                errors.append("energy.pv_kwp must be positive")

    # Validate economic config
    if "economic" in data:
        if not isinstance(data["economic"], dict):
            errors.append("economic must be a dict/object")
        else:
            # Check n_mc
            if "n_mc" in data["economic"]:
                if not isinstance(data["economic"]["n_mc"], int):
                    errors.append("economic.n_mc must be an integer")
                elif data["economic"]["n_mc"] <= 0:
                    errors.append("economic.n_mc must be positive")

            # Check n_years
            if "n_years" in data["economic"]:
                if not isinstance(data["economic"]["n_years"], int):
                    errors.append("economic.n_years must be an integer")
                elif data["economic"]["n_years"] <= 0:
                    errors.append("economic.n_years must be positive")

            # Phase 11 — tax_bonus and inflation sub-blocks
            if "tax_bonus" in data["economic"]:
                errors.extend(_validate_tax_bonus(data["economic"]["tax_bonus"]))
            if "inflation" in data["economic"]:
                errors.extend(_validate_inflation(data["economic"]["inflation"]))

    # Validate price model
    if "price" in data:
        if not isinstance(data["price"], dict):
            errors.append("price must be a dict/object")
        else:
            errors.extend(_validate_price_model(data["price"]))

    # Phase 16 — detailed electrical model block (opt-in).
    if "electrical" in data:
        errors.extend(_validate_electrical(data["electrical"], data))

    # Phase 17 — stochastic load decorator + thermal HVAC additive load.
    if isinstance(data.get("load_profile"), dict) and "stochastic" in data["load_profile"]:
        errors.extend(_validate_stochastic_load(data["load_profile"]["stochastic"]))
    if "stochastic_load" in data:
        errors.extend(_validate_stochastic_load(data["stochastic_load"]))
    if "thermal_load" in data:
        errors.extend(_validate_thermal_load(data["thermal_load"], data))

    return errors


_VALID_PRICE_MODEL_TYPES = {
    "escalating",
    "deterministic",
    "legacy",
    "gbm",
    "random_walk",
    "mean_reverting",
    "ou",
    "ornstein_uhlenbeck",
}


def _validate_price_model(price_cfg: Dict[str, Any]) -> List[str]:
    """
    Validate the ``price`` block of a scenario, dispatching on ``model_type``.

    The function only enforces constraints that protect the model from
    invalid configurations (positive prices, non-negative volatility, valid
    model name). Sensible defaults are NOT enforced here — they live in
    ``scenario_builder.build_default_price_model``.

    Args:
        price_cfg: The ``price`` sub-dict of a hydrated scenario.

    Returns:
        List of human-readable error messages. Empty if the block is valid.

    Notes:
        - When ``model_type`` is missing, the legacy ``escalating`` rules apply.
        - Unknown ``model_type`` values produce a single error pointing the
          user to the list of supported model identifiers.
    """
    errors: List[str] = []
    model_type = str(price_cfg.get("model_type", "escalating")).lower()

    if model_type not in _VALID_PRICE_MODEL_TYPES:
        errors.append(
            f"price.model_type='{model_type}' is not recognised. "
            f"Valid values: escalating, gbm, mean_reverting."
        )
        return errors

    # Common: base price must be > 0 when explicitly provided.
    if "base_price_eur_per_kwh" in price_cfg:
        bp = price_cfg["base_price_eur_per_kwh"]
        if not isinstance(bp, (int, float)):
            errors.append("price.base_price_eur_per_kwh must be a number")
        elif bp <= 0:
            errors.append("price.base_price_eur_per_kwh must be positive")

    if model_type in ("gbm", "random_walk"):
        vol = price_cfg.get("volatility_annual")
        if vol is not None and (not isinstance(vol, (int, float)) or vol < 0):
            errors.append("price.volatility_annual must be a non-negative number")

    if model_type in ("mean_reverting", "ou", "ornstein_uhlenbeck"):
        kappa = price_cfg.get("mean_reversion_speed_annual")
        if kappa is None:
            errors.append(
                "price.mean_reversion_speed_annual is required for "
                "model_type='mean_reverting' (must be > 0)"
            )
        elif not isinstance(kappa, (int, float)) or kappa <= 0:
            errors.append(
                "price.mean_reversion_speed_annual must be a strictly positive number"
            )
        vol = price_cfg.get("volatility_annual")
        if vol is not None and (not isinstance(vol, (int, float)) or vol < 0):
            errors.append("price.volatility_annual must be a non-negative number")
        ltp = price_cfg.get("long_term_price_eur_per_kwh")
        if ltp is not None and (not isinstance(ltp, (int, float)) or ltp <= 0):
            errors.append(
                "price.long_term_price_eur_per_kwh must be a strictly positive number"
            )

    return errors


def _validate_tax_bonus(raw: Any) -> List[str]:
    """
    Validate the ``economic.tax_bonus`` block (Phase 11).

    Enforces sane bounds on the fraction (0–1) and the duration (≥ 1).
    Does NOT enforce ``duration_years ≤ n_years`` because the simulator
    silently truncates oversized durations (documented behaviour).

    Args:
        raw: The ``tax_bonus`` sub-dict, or anything if the user typed
            the wrong shape.

    Returns:
        List of human-readable error messages.
    """
    errors: List[str] = []
    if not isinstance(raw, dict):
        return ["economic.tax_bonus must be a dict/object"]
    if "enabled" in raw and not isinstance(raw["enabled"], bool):
        errors.append("economic.tax_bonus.enabled must be a boolean")
    if "fraction_of_investment" in raw:
        f = raw["fraction_of_investment"]
        if not isinstance(f, (int, float)):
            errors.append(
                "economic.tax_bonus.fraction_of_investment must be a number"
            )
        elif not (0.0 <= float(f) <= 1.0):
            errors.append(
                "economic.tax_bonus.fraction_of_investment must be in [0, 1] "
                "(decimal fraction, e.g. 0.5 for 50%)"
            )
    if "duration_years" in raw:
        d = raw["duration_years"]
        if not isinstance(d, int) or isinstance(d, bool):
            errors.append("economic.tax_bonus.duration_years must be an integer")
        elif d < 1:
            errors.append(
                "economic.tax_bonus.duration_years must be >= 1"
            )
    return errors


_VALID_INFLATION_MODES = {"deterministic", "stochastic"}


def _validate_inflation(raw: Any) -> List[str]:
    """
    Validate the ``economic.inflation`` block (Phase 11).

    Enforces the mode literal, non-negative std, and a coherent clipping
    interval. Does NOT require that ``mean`` itself lies inside the clip
    interval — sampling will clip out-of-range draws, which is the
    expected behaviour for a Truncated Normal.
    """
    errors: List[str] = []
    if not isinstance(raw, dict):
        return ["economic.inflation must be a dict/object"]
    if "mode" in raw:
        if raw["mode"] not in _VALID_INFLATION_MODES:
            errors.append(
                f"economic.inflation.mode must be one of "
                f"{sorted(_VALID_INFLATION_MODES)}; got {raw['mode']!r}"
            )
    for key in ("mean", "std", "min_clip", "max_clip"):
        if key in raw and not isinstance(raw[key], (int, float)):
            errors.append(f"economic.inflation.{key} must be a number")
    if "std" in raw and isinstance(raw["std"], (int, float)) and raw["std"] < 0:
        errors.append("economic.inflation.std must be non-negative")
    if (
        "min_clip" in raw
        and "max_clip" in raw
        and isinstance(raw["min_clip"], (int, float))
        and isinstance(raw["max_clip"], (int, float))
        and raw["min_clip"] > raw["max_clip"]
    ):
        errors.append(
            "economic.inflation.min_clip must be <= economic.inflation.max_clip"
        )
    return errors


_VALID_ELECTRICAL_MODES = {"off", "disabled", "mppt_window"}
_REQUIRED_PANEL_ELECTRICAL = (
    "power_w",
    "v_oc_stc_v",
    "v_mpp_stc_v",
    "n_cells_series",
    "beta_voc_pct_per_c",
    "gamma_pmax_pct_per_c",
    "noct_c",
)
_REQUIRED_INVERTER_ELECTRICAL = (
    "v_dc_min_v",
    "v_dc_max_v",
    "v_mppt_min_v",
    "v_mppt_max_v",
)


def _validate_electrical(raw: Any, full_data: Dict[str, Any]) -> List[str]:
    """
    Validate the ``electrical`` block (Phase 16).

    Three regimes:

    1. ``mode`` missing or set to ``"off"`` → block silently accepted.
       Any extra fields are ignored (the simulator runs the legacy
       energy path).
    2. ``mode="mppt_window"`` → enforce the full datasheet schema:
       panel + inverter must contain every required field, ``pv_strings``
       (when present) must be a non-empty list of valid string entries,
       and the scenario must reference a Phase-15 ``climate_profile_id``
       so the simulator has hourly ``T_ambient`` to drive ``T_cell``.
    3. Any other ``mode`` value → single, explanatory error.

    The function NEVER raises — it accumulates messages in a list so
    the UI can show all problems at once.

    Args:
        raw: The ``electrical`` sub-dict.
        full_data: The whole scenario dict (needed for the cross-block
            ``climate_profile_id`` requirement).

    Returns:
        List of human-readable error messages. Empty list when the
        block is valid (or absent in legacy mode).
    """
    errors: List[str] = []
    if not isinstance(raw, dict):
        return ["electrical must be a dict/object"]
    mode = str(raw.get("mode", "off")).lower()
    if mode not in _VALID_ELECTRICAL_MODES:
        return [
            f"electrical.mode={raw.get('mode')!r} not recognised. "
            f"Valid values: {sorted(_VALID_ELECTRICAL_MODES)}."
        ]
    if mode in ("off", "disabled"):
        return errors
    # mode == "mppt_window" — enforce full schema.
    panel = raw.get("panel") or {}
    inverter = raw.get("inverter") or {}
    missing_panel = [k for k in _REQUIRED_PANEL_ELECTRICAL if panel.get(k) in (None, "")]
    missing_inverter = [
        k for k in _REQUIRED_INVERTER_ELECTRICAL if inverter.get(k) in (None, "")
    ]
    if missing_panel:
        errors.append(
            "electrical.mode='mppt_window' requires panel datasheet fields "
            f"to be set; missing: {', '.join(missing_panel)}"
        )
    if missing_inverter:
        errors.append(
            "electrical.mode='mppt_window' requires inverter datasheet fields "
            f"to be set; missing: {', '.join(missing_inverter)}"
        )
    if "climate_profile_id" not in full_data and "climate_profile_name" not in full_data:
        errors.append(
            "electrical.mode='mppt_window' requires a climate_profile_id "
            "(or climate_profile_name) at the scenario root so the "
            "simulator can derive hourly T_cell from a Phase-15 thermal "
            "model."
        )
    pv_strings = raw.get("pv_strings")
    if pv_strings is not None:
        if not isinstance(pv_strings, list) or not pv_strings:
            errors.append("electrical.pv_strings must be a non-empty list when present")
        else:
            for idx, entry in enumerate(pv_strings):
                if not isinstance(entry, dict):
                    errors.append(f"electrical.pv_strings[{idx}] must be a dict")
                    continue
                if "n_panels" not in entry:
                    errors.append(
                        f"electrical.pv_strings[{idx}] missing 'n_panels'"
                    )
                else:
                    n = entry["n_panels"]
                    if not isinstance(n, int) or n <= 0:
                        errors.append(
                            f"electrical.pv_strings[{idx}].n_panels must be a positive integer"
                        )
                mppt_id = entry.get("mppt_id", 0)
                if not isinstance(mppt_id, int) or mppt_id < 0:
                    errors.append(
                        f"electrical.pv_strings[{idx}].mppt_id must be a non-negative integer"
                    )
    k = raw.get("derating_exponent_k")
    if k is not None and (not isinstance(k, (int, float)) or k < 0):
        errors.append("electrical.derating_exponent_k must be a non-negative number")
    return errors


_VALID_INSULATION_PRESETS = {"poor", "standard", "good"}


def _validate_stochastic_load(raw: Any) -> List[str]:
    """
    Validate the Phase-17 ``load_profile.stochastic`` (or
    root ``stochastic_load``) block.

    Enforces ``sigma_log >= 0`` and ``|phi_intra_day| < 1``. Does not
    require ``enabled=True``: the user may legitimately store the
    parameters but keep the feature off.
    """
    errors: List[str] = []
    if not isinstance(raw, dict):
        return ["stochastic_load must be a dict/object"]
    if "enabled" in raw and not isinstance(raw["enabled"], bool):
        errors.append("stochastic_load.enabled must be a boolean")
    if "sigma_log" in raw:
        s = raw["sigma_log"]
        if not isinstance(s, (int, float)):
            errors.append("stochastic_load.sigma_log must be a number")
        elif s < 0:
            errors.append("stochastic_load.sigma_log must be >= 0")
    if "phi_intra_day" in raw:
        p = raw["phi_intra_day"]
        if not isinstance(p, (int, float)):
            errors.append("stochastic_load.phi_intra_day must be a number")
        elif not (-1.0 < float(p) < 1.0):
            errors.append("stochastic_load.phi_intra_day must be strictly within (-1, 1)")
    return errors


def _validate_thermal_load(raw: Any, full_data: Dict[str, Any]) -> List[str]:
    """
    Validate the Phase-17 ``thermal_load`` block.

    When ``enabled=False`` (or absent) the block is silently accepted.
    When ``enabled=True`` the function enforces:
        - presence of ``climate_profile_id`` (or ``_name``) at the
          scenario root (the HVAC controller depends on T_ambient);
        - ``house.insulation_preset`` ∈ ``{poor, standard, good}`` when
          no explicit ``ua_w_per_c_per_m2`` is supplied;
        - ``house.floor_area_m2 > 0``;
        - ``heat_pump.cop_heating > 0``, ``cop_cooling > 0``,
          ``p_elec_max_kw > 0``;
        - ``setpoint.t_setpoint_heating_c < t_setpoint_cooling_c``.
    """
    errors: List[str] = []
    if not isinstance(raw, dict):
        return ["thermal_load must be a dict/object"]
    if "enabled" in raw and not isinstance(raw["enabled"], bool):
        errors.append("thermal_load.enabled must be a boolean")
    enabled = bool(raw.get("enabled", False))
    if not enabled:
        return errors
    if (
        "climate_profile_id" not in full_data
        and "climate_profile_name" not in full_data
    ):
        errors.append(
            "thermal_load.enabled=true requires a climate_profile_id "
            "(or climate_profile_name) at the scenario root."
        )

    house = raw.get("house") or {}
    if not isinstance(house, dict):
        errors.append("thermal_load.house must be a dict/object")
    else:
        if "floor_area_m2" in house:
            fa = house["floor_area_m2"]
            if not isinstance(fa, (int, float)) or fa <= 0:
                errors.append("thermal_load.house.floor_area_m2 must be > 0")
        ua_user = house.get("ua_w_per_c_per_m2")
        if ua_user is None:
            preset = str(house.get("insulation_preset", "standard")).lower()
            if preset not in _VALID_INSULATION_PRESETS:
                errors.append(
                    "thermal_load.house.insulation_preset must be one of "
                    f"{sorted(_VALID_INSULATION_PRESETS)}; got {preset!r}"
                )
        elif not isinstance(ua_user, (int, float)) or ua_user < 0:
            errors.append("thermal_load.house.ua_w_per_c_per_m2 must be >= 0")

    hp = raw.get("heat_pump") or {}
    if not isinstance(hp, dict):
        errors.append("thermal_load.heat_pump must be a dict/object")
    else:
        for key in ("cop_heating", "cop_cooling", "p_elec_max_kw"):
            if key in hp:
                v = hp[key]
                if not isinstance(v, (int, float)) or v <= 0:
                    errors.append(f"thermal_load.heat_pump.{key} must be > 0")

    sp = raw.get("setpoint") or {}
    if not isinstance(sp, dict):
        errors.append("thermal_load.setpoint must be a dict/object")
    else:
        heat = sp.get("t_setpoint_heating_c")
        cool = sp.get("t_setpoint_cooling_c")
        if isinstance(heat, (int, float)) and isinstance(cool, (int, float)):
            if heat >= cool:
                errors.append(
                    "thermal_load.setpoint.t_setpoint_heating_c must be < "
                    "t_setpoint_cooling_c (dead-band)"
                )
        away = sp.get("t_setpoint_away_c")
        if away is not None and not isinstance(away, (int, float)):
            errors.append("thermal_load.setpoint.t_setpoint_away_c must be a number")
    return errors


def validate_optimization(data: Dict[str, Any]) -> List[str]:
    """
    Validate optimization configuration after hydration.

    Checks both the base scenario and optimization-specific fields.

    Args:
        data: Hydrated optimization configuration dict.

    Returns:
        List of error messages. Empty list if valid.

    Example:
        ```python
        errors = validate_optimization(hydrated_config)
        if errors:
            raise ValueError(f"Invalid configuration: {errors}")
        ```
    """
    errors = []

    # First validate as a scenario (optimization contains a base scenario)
    scenario_errors = validate_scenario(data)
    errors.extend(scenario_errors)

    # Check for optimization-specific fields
    if "optimization" not in data:
        errors.append("Missing required section: 'optimization'")
        return errors

    opt_config = data["optimization"]
    if not isinstance(opt_config, dict):
        errors.append("optimization must be a dict/object")
        return errors

    # Validate hardware options
    has_options = False

    if "inverter_options" in opt_config:
        if not isinstance(opt_config["inverter_options"], list):
            errors.append("optimization.inverter_options must be a list")
        elif len(opt_config["inverter_options"]) == 0:
            errors.append("optimization.inverter_options cannot be empty")
        else:
            has_options = True

    if "panel_options" in opt_config:
        if not isinstance(opt_config["panel_options"], list):
            errors.append("optimization.panel_options must be a list")
        elif len(opt_config["panel_options"]) == 0:
            errors.append("optimization.panel_options cannot be empty")
        else:
            has_options = True

    if "battery_options" in opt_config:
        if not isinstance(opt_config["battery_options"], list):
            errors.append("optimization.battery_options must be a list")
        elif len(opt_config["battery_options"]) == 0:
            # An empty battery_options list is only an error when no other
            # hardware option list is present (no inverters or panels to try).
            if not has_options:
                errors.append("optimization.battery_options cannot be empty")
        else:
            has_options = True

    if not has_options:
        errors.append("optimization must have at least one of: inverter_options, panel_options, or battery_options")

    return errors
