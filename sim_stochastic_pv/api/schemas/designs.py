"""
Pydantic schemas for plant designs (``/api/designs``).

A plant design ("Impianto") is the first-class description of one PV
system. The ``essential`` level captures a received commercial offer in
five numbers (AC power, optional DC power, optional storage, turn-key
cost, optional incentive); the ``detailed`` level is produced by the
electrical designer and carries the full layout in the same ``data``
payload.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaxBonusBlock(BaseModel):
    """
    Tax-incentive block carried by a design.

    Mirrors the scenario's ``economic.tax_bonus`` shape so the resolver
    can copy it verbatim.

    Attributes:
        enabled: Whether the incentive applies.
        fraction_of_investment: Refunded fraction of the investment
            (0–1, e.g. 0.5 for the Italian 50% detrazione).
        duration_years: Number of years the refund is spread over.
    """

    enabled: bool = True
    fraction_of_investment: Annotated[float, Field(gt=0.0, le=1.0)] = 0.5
    duration_years: Annotated[int, Field(ge=1, le=30)] = 10


class EssentialDesignData(BaseModel):
    """
    Nameplate payload of an ``essential`` design (a received offer).

    Attributes:
        p_ac_kw: Inverter AC nameplate power (kW), as stated by the offer.
        p_dc_kwp: DC peak power (kWp). ``None`` when the offer does not
            state it — the resolver assumes it equals ``p_ac_kw``.
        storage_kwh: Battery capacity (kWh). ``None``/0 = no storage.
        total_cost_eur: Turn-key cost of the offer (€).
        tax_bonus: Optional incentive block.
        notes: Free-text notes about the offer.
    """

    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "p_ac_kw": 6.0,
            "p_dc_kwp": 6.6,
            "storage_kwh": 10.0,
            "total_cost_eur": 14500.0,
            "tax_bonus": {
                "enabled": True,
                "fraction_of_investment": 0.5,
                "duration_years": 10,
            },
        }]
    })

    p_ac_kw: Annotated[float, Field(gt=0.0, le=1000.0)]
    p_dc_kwp: Annotated[float | None, Field(gt=0.0, le=2000.0)] = None
    storage_kwh: Annotated[float | None, Field(ge=0.0, le=1000.0)] = None
    total_cost_eur: Annotated[float, Field(gt=0.0)]
    tax_bonus: TaxBonusBlock | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _dc_not_absurdly_low(self) -> "EssentialDesignData":
        """A DC power below half the AC rating is almost surely a typo."""
        if self.p_dc_kwp is not None and self.p_dc_kwp < 0.5 * self.p_ac_kw:
            raise ValueError(
                f"p_dc_kwp={self.p_dc_kwp} is less than half of "
                f"p_ac_kw={self.p_ac_kw}: probable typo in the offer data"
            )
        return self


class PlantDesignCreate(BaseModel):
    """
    Create/upsert payload for a plant design.

    Attributes:
        name: Unique design identifier (upsert key).
        design_level: Currently only ``"essential"`` is accepted from the
            API — ``"detailed"`` designs are produced by the designer
            feature, which owns their payload validation.
        description: Optional free text.
        data: Validated :class:`EssentialDesignData` payload.
        location_id: Optional installation-site FK; when set, scenarios
            built from this design inherit the site's solar (and climate)
            profile automatically.
        inverter_id: Optional catalogue references, when the offer names
            the hardware.
        panel_id: See ``inverter_id``.
        battery_id: See ``inverter_id``.
    """

    name: Annotated[str, Field(min_length=1, max_length=255)]
    design_level: Literal["essential"] = "essential"
    description: str | None = None
    data: EssentialDesignData
    location_id: int | None = None
    inverter_id: int | None = None
    panel_id: int | None = None
    battery_id: int | None = None


class PlantDesignUpdate(BaseModel):
    """
    Partial-update payload (rename, edit data, move site).

    Only the fields explicitly present are written.
    """

    name: Annotated[str | None, Field(min_length=1, max_length=255)] = None
    description: str | None = None
    data: EssentialDesignData | None = None
    location_id: int | None = None
    inverter_id: int | None = None
    panel_id: int | None = None
    battery_id: int | None = None


class PlantDesignResponse(BaseModel):
    """
    Plant design record as returned by the API.

    ``data`` is returned as a plain dict (not re-validated) so future
    ``detailed`` payloads round-trip unchanged.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    design_level: str
    description: str | None = None
    data: dict
    location_id: int | None = None
    inverter_id: int | None = None
    panel_id: int | None = None
    battery_id: int | None = None
    updated_at: datetime | None = None
