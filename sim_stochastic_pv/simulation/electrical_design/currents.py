"""
Per-MPPT current checks: operating current, short-circuit current and
physical string inputs.

The inverter datasheets distinguish the maximum *operating* current the
tracker can exploit (exceeding it only wastes current, the MPP shifts)
from the maximum *short-circuit* current it tolerates (a hardware
limit). Both are checked at the hot-cell corner, where the module
currents peak. The 1.25 safety factor is reserved for cables and
protections (see :mod:`.cables` / :mod:`.protections`) — datasheet
inverter limits are compared against the bare hot-corner currents, as
the reference spreadsheet does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..electrical import InverterElectricalSpecs
from .sizing import PlantSizing, TemperatureCorrectedValues, _require


@dataclass(frozen=True)
class CurrentChecks:
    """
    Per-MPPT current verification for the worst-loaded tracker.

    Negative margins mean the check fails. All currents in A.

    Attributes:
        strings_per_mppt: Strings on the worst-loaded MPPT
            (``ceil(n_strings / n_mppt)``).
        inputs_ok: ``strings_per_mppt <= max_strings_per_mppt`` —
            enough physical inputs.
        i_operating_a: Hot-corner operating current into the tracker
            (``strings_per_mppt × I_mp(hot)``).
        i_operating_margin_a: ``i_dc_max_per_mppt − i_operating``.
        i_sc_a: Hot-corner short-circuit current into the tracker.
        i_sc_margin_a: ``i_sc_max_per_mppt − i_sc``.
    """

    strings_per_mppt: int
    inputs_ok: bool
    i_operating_a: float
    i_operating_margin_a: float
    i_sc_a: float
    i_sc_margin_a: float


def check_mppt_currents(
    inverter: InverterElectricalSpecs,
    corrected: TemperatureCorrectedValues,
    plant: PlantSizing,
) -> CurrentChecks:
    """
    Verify the worst-loaded MPPT tracker against the inverter limits.

    Args:
        inverter: Inverter specs (requires ``n_mppt_trackers``,
            ``i_dc_max_per_mppt_a``, ``i_sc_max_per_mppt_a``,
            ``max_strings_per_mppt``).
        corrected: Hot-corner module currents.
        plant: Plant sizing (string count).

    Returns:
        :class:`CurrentChecks`.

    Example:
        ```python
        # Reference case (2 strings on 2 MPPTs → 1 string each):
        # I_op = 15.235 A vs 12 A limit → FAIL (margin −3.2 A),
        # I_sc = 16.237 A vs 15 A limit → FAIL — the spreadsheet's
        # deliberately broken example, reproduced by the tests.
        ```
    """
    n_mppt = max(1, int(inverter.n_mppt_trackers or 1))
    strings_per_mppt = math.ceil(plant.n_strings / n_mppt)

    i_dc_max = _require(inverter.i_dc_max_per_mppt_a, "i_dc_max_per_mppt_a")
    i_sc_max = _require(inverter.i_sc_max_per_mppt_a, "i_sc_max_per_mppt_a")
    max_inputs = int(_require(inverter.max_strings_per_mppt, "max_strings_per_mppt"))

    i_operating = strings_per_mppt * corrected.i_mp_hot_a
    i_sc = strings_per_mppt * corrected.i_sc_hot_a
    return CurrentChecks(
        strings_per_mppt=strings_per_mppt,
        inputs_ok=strings_per_mppt <= max_inputs,
        i_operating_a=i_operating,
        i_operating_margin_a=i_dc_max - i_operating,
        i_sc_a=i_sc,
        i_sc_margin_a=i_sc_max - i_sc,
    )
