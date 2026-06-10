"""
Hydration functions for expanding ID references to full specifications.
"""

from __future__ import annotations

from typing import Any, Dict

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import (
    BatteryModel,
    ClimateProfileModel,
    InverterModel,
    LoadProfileModel,
    PanelModel,
    PlantDesignModel,
    PriceProfileModel,
    SolarProfileModel,
)


def apply_plant_design(
    scenario_data: Dict[str, Any],
    session: Session,
) -> Dict[str, Any]:
    """
    Expand a ``plant_design_id`` reference into the scenario fields it owns.

    A plant design ("Impianto") is the source of truth for the *system*
    and its *investment*: when a scenario references one, the design's
    values overwrite any same-named field in the scenario, while
    everything the design does not define (load, price, market coupling,
    MC settings, horizon) stays untouched. Mapping for the ``essential``
    level (a received offer):

    - ``energy.pv_kwp`` / ``solar.pv_kwp`` ← ``p_dc_kwp`` (or ``p_ac_kw``
      when the offer does not state a DC value);
    - ``energy.inverter_p_ac_max_kw`` ← ``p_ac_kw``;
    - ``energy.battery_specs.capacity_kwh`` ← ``storage_kwh`` (0/absent →
      no battery, ``n_batteries`` set accordingly);
    - ``economic.investment_eur`` ← ``total_cost_eur``;
    - ``economic.tax_bonus`` ← the design's incentive block, when present;
    - when the design is anchored to a location, the site's solar profile
      (and climate profile, when present) is inherited:
      ``solar.solar_profile_id`` / ``climate_profile_id``.

    Args:
        scenario_data: Scenario dict, possibly carrying ``plant_design_id``.
        session: Active SQLAlchemy session.

    Returns:
        A new scenario dict with the design expanded. The input is
        returned unchanged (same object) when no ``plant_design_id`` is
        present.

    Raises:
        ValueError: Unknown design id, unsupported ``design_level``, or a
            payload missing the required nameplate fields.
    """
    if "plant_design_id" not in scenario_data:
        return scenario_data

    design_id = scenario_data["plant_design_id"]
    design = session.get(PlantDesignModel, design_id)
    if design is None:
        raise ValueError(f"Plant design ID {design_id} not found")
    if design.design_level != "essential":
        raise ValueError(
            f"Plant design '{design.name}' has level '{design.design_level}', "
            "which this resolver does not support yet (only 'essential')."
        )

    payload = design.data or {}
    try:
        p_ac_kw = float(payload["p_ac_kw"])
        total_cost_eur = float(payload["total_cost_eur"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Plant design '{design.name}' is missing required nameplate "
            f"fields (p_ac_kw, total_cost_eur): {payload!r}"
        ) from exc
    p_dc_kwp = float(payload.get("p_dc_kwp") or p_ac_kw)
    storage_kwh = float(payload.get("storage_kwh") or 0.0)

    hydrated = dict(scenario_data)

    energy = dict(hydrated.get("energy") or {})
    energy["pv_kwp"] = p_dc_kwp
    energy["inverter_p_ac_max_kw"] = p_ac_kw
    energy["battery_specs"] = {"capacity_kwh": storage_kwh, "cycles_life": 6000}
    energy["n_batteries"] = 1 if storage_kwh > 0 else 0
    hydrated["energy"] = energy

    solar = dict(hydrated.get("solar") or {})
    solar["pv_kwp"] = p_dc_kwp

    economic = dict(hydrated.get("economic") or {})
    economic["investment_eur"] = total_cost_eur
    tax_bonus = payload.get("tax_bonus")
    if tax_bonus:
        economic["tax_bonus"] = dict(tax_bonus)
    hydrated["economic"] = economic

    # Site inheritance: the design's location decides where the plant is.
    if design.location_id is not None:
        solar_profile = (
            session.query(SolarProfileModel)
            .filter_by(location_id=design.location_id)
            .order_by(SolarProfileModel.name)
            .first()
        )
        if solar_profile is not None:
            solar["solar_profile_id"] = solar_profile.id
        climate_profile = (
            session.query(ClimateProfileModel)
            .filter_by(location_id=design.location_id)
            .order_by(ClimateProfileModel.name)
            .first()
        )
        if climate_profile is not None:
            hydrated["climate_profile_id"] = climate_profile.id

    hydrated["solar"] = solar
    return hydrated


def hydrate_scenario(
    scenario_data: Dict[str, Any],
    session: Session,
) -> Dict[str, Any]:
    """
    Hydrate a scenario configuration by replacing hardware/profile IDs with full specs.

    Handles single hardware selections:
    - plant_design_id → system + investment fields (see
      :func:`apply_plant_design`; runs first so a design-driven scenario
      gets its energy/economic blocks before the other expansions)
    - inverter_id → inverter spec dict
    - panel_id → panel spec dict
    - battery_id → battery spec dict
    - load_profile_id → load profile spec
    - price_profile_id → price profile spec

    Args:
        scenario_data: Partial scenario dict with IDs instead of full specs.
        session: Active SQLAlchemy session for database queries.

    Returns:
        Complete scenario dict with all hardware and profile data hydrated.

    Raises:
        ValueError: If a referenced ID is not found in the database.
    """
    scenario_data = apply_plant_design(scenario_data, session)
    hydrated = dict(scenario_data)

    # Hydrate inverter
    if "inverter_id" in scenario_data:
        inverter_id = scenario_data["inverter_id"]
        stmt = select(InverterModel).where(InverterModel.id == inverter_id)
        inverter = session.execute(stmt).scalar_one_or_none()
        if not inverter:
            raise ValueError(f"Inverter ID {inverter_id} not found")

        # Build energy section with inverter data
        if "energy" not in hydrated:
            hydrated["energy"] = {}
        # Get p_ac_max from specs or nominal_power_kw
        p_ac_max = (
            inverter.specs.get("p_ac_max_kw")
            if inverter.specs
            else inverter.nominal_power_kw
        )
        if p_ac_max:
            hydrated["energy"]["inverter_p_ac_max_kw"] = p_ac_max

    # Hydrate panel
    if "panel_id" in scenario_data:
        panel_id = scenario_data["panel_id"]
        stmt = select(PanelModel).where(PanelModel.id == panel_id)
        panel = session.execute(stmt).scalar_one_or_none()
        if not panel:
            raise ValueError(f"Panel ID {panel_id} not found")

        # Store panel power for later use in solar and energy configs
        if "solar" not in hydrated:
            hydrated["solar"] = {}
        if "energy" not in hydrated:
            hydrated["energy"] = {}

        # Panel specs might be used by optimization - store in metadata
        hydrated.setdefault("_hardware_metadata", {})["panel"] = {
            "id": panel.id,
            "name": panel.name,
            "power_w": panel.power_w,
            "price_eur": panel.specs.get("price_eur") if panel.specs else None,
        }

    # Hydrate battery
    if "battery_id" in scenario_data:
        battery_id = scenario_data["battery_id"]
        stmt = select(BatteryModel).where(BatteryModel.id == battery_id)
        battery = session.execute(stmt).scalar_one_or_none()
        if not battery:
            raise ValueError(f"Battery ID {battery_id} not found")

        if "energy" not in hydrated:
            hydrated["energy"] = {}

        hydrated["energy"]["battery_specs"] = {
            "capacity_kwh": battery.capacity_kwh or 0.0,
            # Use `or` to treat both missing key and explicit None as "use default".
            # specs.get("cycles_life", 0) would return None when the key exists
            # but its stored value is null (e.g. battery created without the field),
            # which later causes a TypeError in BatteryModel._update_soh.
            "cycles_life": (
                battery.specs.get("cycles_life") or 6000 if battery.specs else 6000
            ),
        }

        # Store battery metadata
        hydrated.setdefault("_hardware_metadata", {})["battery"] = {
            "id": battery.id,
            "name": battery.name,
            "capacity_kwh": battery.capacity_kwh,
            "price_eur": battery.specs.get("price_eur") if battery.specs else None,
        }

    # Hydrate load profile
    if "load_profile_id" in scenario_data:
        load_id = scenario_data["load_profile_id"]
        stmt = select(LoadProfileModel).where(LoadProfileModel.id == load_id)
        load_profile = session.execute(stmt).scalar_one_or_none()
        if not load_profile:
            raise ValueError(f"Load profile ID {load_id} not found")

        hydrated["load_profile"] = load_profile.data

    # Hydrate price profile
    if "price_profile_id" in scenario_data:
        price_id = scenario_data["price_profile_id"]
        stmt = select(PriceProfileModel).where(PriceProfileModel.id == price_id)
        price_profile = session.execute(stmt).scalar_one_or_none()
        if not price_profile:
            raise ValueError(f"Price profile ID {price_id} not found")

        hydrated["price"] = price_profile.data

    return hydrated


def hydrate_optimization(
    optimization_data: Dict[str, Any],
    session: Session,
) -> Dict[str, Any]:
    """
    Hydrate an optimization configuration by expanding hardware_selections.

    Handles multiple hardware selections:
    - hardware_selections.inverter_ids[] → list of inverter specs
    - hardware_selections.panel_ids[] → list of panel specs
    - hardware_selections.battery_ids[] → list of battery specs

    Also hydrates the base scenario using hydrate_scenario().

    Args:
        optimization_data: Partial optimization dict with hardware_selections containing IDs.
        session: Active SQLAlchemy session for database queries.

    Returns:
        Complete optimization dict with all hardware options expanded.

    Raises:
        ValueError: If a referenced ID is not found in the database.
    """
    # First hydrate the base scenario (single IDs)
    hydrated = hydrate_scenario(optimization_data, session)

    # Handle optimization hardware selections (multiple IDs for optimization)
    if "hardware_selections" in optimization_data:
        selections = optimization_data["hardware_selections"]

        # Hydrate inverter options for optimization
        if "inverter_ids" in selections and selections["inverter_ids"]:
            inverter_ids = selections["inverter_ids"]
            stmt = select(InverterModel).where(InverterModel.id.in_(inverter_ids))
            inverter_records = list(session.execute(stmt).scalars().all())

            inverter_options = []
            for inv in inverter_records:
                inv_dict = {
                    "id": inv.id,
                    "name": inv.name,
                    "p_ac_max_kw": inv.specs.get("p_ac_max_kw") if inv.specs else inv.nominal_power_kw,
                    "p_dc_max_kw": inv.specs.get("p_dc_max_kw") if inv.specs else None,
                    "price_eur": inv.specs.get("price_eur") if inv.specs else None,
                    "install_cost_eur": inv.specs.get("install_cost_eur") if inv.specs else None,
                }
                if inv.specs and "integrated_battery_specs" in inv.specs:
                    inv_dict["integrated_battery_specs"] = inv.specs["integrated_battery_specs"]
                    inv_dict["integrated_battery_price_eur"] = inv.specs.get("integrated_battery_price_eur")
                    inv_dict["integrated_battery_count_options"] = inv.specs.get("integrated_battery_count_options", [])
                inverter_options.append(inv_dict)

            if "optimization" not in hydrated:
                hydrated["optimization"] = {}
            hydrated["optimization"]["inverter_options"] = inverter_options

        # Hydrate panel options for optimization
        if "panel_ids" in selections and selections["panel_ids"]:
            panel_ids = selections["panel_ids"]
            stmt = select(PanelModel).where(PanelModel.id.in_(panel_ids))
            panel_records = list(session.execute(stmt).scalars().all())

            panel_options = []
            for panel in panel_records:
                panel_options.append({
                    "id": panel.id,
                    "name": panel.name,
                    "power_w": panel.power_w,
                    "price_eur": panel.specs.get("price_eur") if panel.specs else None,
                })

            if "optimization" not in hydrated:
                hydrated["optimization"] = {}
            hydrated["optimization"]["panel_options"] = panel_options

        # Hydrate battery options for optimization
        if "battery_ids" in selections and selections["battery_ids"]:
            battery_ids = selections["battery_ids"]
            stmt = select(BatteryModel).where(BatteryModel.id.in_(battery_ids))
            battery_records = list(session.execute(stmt).scalars().all())

            battery_options = []
            for bat in battery_records:
                battery_options.append({
                    "id": bat.id,
                    "name": bat.name,
                    "specs": {
                        "capacity_kwh": bat.capacity_kwh or 0.0,
                        "cycles_life": bat.specs.get("cycles_life") or 6000 if bat.specs else 6000,
                    },
                    "price_eur": bat.specs.get("price_eur") if bat.specs else None,
                    "manufacturer": bat.manufacturer,
                    "model_number": bat.model_number,
                    "datasheet": bat.datasheet,
                })

            if "optimization" not in hydrated:
                hydrated["optimization"] = {}
            hydrated["optimization"]["battery_options"] = battery_options

    return hydrated


def hydrate_scenario_from_ids(
    scenario_data: Dict[str, Any],
    session: Session,
) -> Dict[str, Any]:
    """
    Hydrate a scenario or optimization configuration (backward compatible).

    DEPRECATED: Use hydrate_scenario() or hydrate_optimization() instead.

    Automatically detects whether the data contains hardware_selections
    (optimization) or single hardware IDs (scenario) and calls the appropriate
    hydration function.

    Args:
        scenario_data: Partial scenario/optimization dict with IDs.
        session: Active SQLAlchemy session for database queries.

    Returns:
        Complete configuration dict with all data hydrated.

    Raises:
        ValueError: If a referenced ID is not found in the database.
    """
    # Auto-detect optimization vs scenario based on hardware_selections presence
    if "hardware_selections" in scenario_data:
        return hydrate_optimization(scenario_data, session)
    else:
        return hydrate_scenario(scenario_data, session)
