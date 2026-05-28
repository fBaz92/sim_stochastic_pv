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
