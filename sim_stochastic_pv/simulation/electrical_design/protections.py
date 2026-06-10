"""
String-protection (DC fuse) sizing per CEI EN 62548.

The norm requires string overcurrent protection when 3 or more strings
are wired in parallel (a fault in one string can then sink the combined
reverse current of the others). The fuse nominal current must sit in
the window

    1.5 × I_sc(STC)  <=  I_n  <=  2.4 × I_sc(STC)

and must not exceed the module's "max series fuse rating" from the
datasheet (IEC 61730-2). The recommendation picks the first standard
gPV size above the lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..electrical import PanelElectricalSpecs
from .inputs import DesignRequirements
from .sizing import PlantSizing, _require


# Standard gPV fuse sizes commonly available for 10×38 mm holders (A).
STANDARD_GPV_FUSE_SIZES_A: tuple[float, ...] = (
    10.0, 12.0, 15.0, 16.0, 20.0, 25.0, 30.0, 32.0, 40.0, 50.0, 63.0,
)


@dataclass(frozen=True)
class ProtectionSizing:
    """
    String-fuse sizing result.

    Attributes:
        protection_required: ``n_strings >= 3`` (CEI EN 62548).
        i_fuse_min_a: Lower bound ``fuse_factor_min × I_sc(STC)`` (A).
        i_fuse_max_norm_a: Upper bound allowed by the norm (A).
        i_fuse_module_max_a: Max series fuse from the module datasheet
            (A), or ``None`` when the datasheet omits it.
        recommended_fuse_a: First standard gPV size ≥ the lower bound,
            or ``None`` when no standard size qualifies.
        fuse_within_module_limit: Recommendation ≤ module limit
            (``None`` when either side is unknown).
        fuse_within_norm_limit: Recommendation ≤ 2.4 × I_sc(STC)
            (``None`` when no recommendation).
    """

    protection_required: bool
    i_fuse_min_a: float
    i_fuse_max_norm_a: float
    i_fuse_module_max_a: float | None
    recommended_fuse_a: float | None
    fuse_within_module_limit: bool | None
    fuse_within_norm_limit: bool | None


def size_string_protection(
    panel: PanelElectricalSpecs,
    plant: PlantSizing,
    requirements: DesignRequirements,
    standard_sizes_a: tuple[float, ...] = STANDARD_GPV_FUSE_SIZES_A,
) -> ProtectionSizing:
    """
    Size the string fuse per CEI EN 62548.

    Args:
        panel: Module specs (requires ``i_sc_stc_a``; uses
            ``max_series_fuse_a`` when present).
        plant: Plant sizing (string count → protection requirement).
        requirements: Fuse window factors.
        standard_sizes_a: Catalogue of standard gPV sizes to pick from.

    Returns:
        :class:`ProtectionSizing`.

    Example:
        ```python
        # I_sc(STC) = 15.88 A → window [23.82, 38.11] A; first standard
        # size ≥ 23.82 is 25 A; module limit 30 A → all checks OK.
        ```
    """
    i_sc_stc = _require(panel.i_sc_stc_a, "i_sc_stc_a")
    i_min = requirements.fuse_factor_min * i_sc_stc
    i_max_norm = requirements.fuse_factor_max * i_sc_stc
    module_max = (
        float(panel.max_series_fuse_a)
        if panel.max_series_fuse_a is not None
        else None
    )

    recommended = next((s for s in sorted(standard_sizes_a) if s >= i_min), None)

    return ProtectionSizing(
        protection_required=plant.n_strings >= 3,
        i_fuse_min_a=i_min,
        i_fuse_max_norm_a=i_max_norm,
        i_fuse_module_max_a=module_max,
        recommended_fuse_a=recommended,
        fuse_within_module_limit=(
            recommended <= module_max
            if recommended is not None and module_max is not None
            else None
        ),
        fuse_within_norm_limit=(
            recommended <= i_max_norm if recommended is not None else None
        ),
    )
