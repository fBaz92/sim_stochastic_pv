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
    if "load_profile" in data:
        if not isinstance(data["load_profile"], dict):
            errors.append("load_profile must be a dict/object")
        elif "type" not in data["load_profile"]:
            errors.append("load_profile.type is required")

    # Validate solar model
    if "solar" in data:
        if not isinstance(data["solar"], dict):
            errors.append("solar must be a dict/object")
        elif "type" not in data["solar"]:
            errors.append("solar.type is required")

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
        elif "type" not in data["price"]:
            errors.append("price.type is required")

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
            errors.append("optimization.battery_options cannot be empty")
        else:
            has_options = True

    if not has_options:
        errors.append("optimization must have at least one of: inverter_options, panel_options, or battery_options")

    return errors
