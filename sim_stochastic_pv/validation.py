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
