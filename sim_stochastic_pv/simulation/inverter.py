"""
Simple inverter dispatch logic for coupling PV, load, and batteries.
"""

from __future__ import annotations

from typing import Tuple

from .battery import BatteryBank


class InverterAC:
    """
    Simple AC inverter with a limit on AC output power.

    **What this model captures**:

    - Maximum AC output (``p_ac_max_kw``) — the inverter clips both
      PV-direct flow and battery discharge to this ceiling.
    - Optional DC cap (``p_dc_max_kw``) — when set, the array DC power
      is truncated before any further dispatch.

    **What this model does NOT capture** (see
    ``docs/electrical_simplifications.md`` for the full list):

    - String voltage and inverter MPPT window. There is no notion of
      ``v_oc``, ``v_mpp``, ``v_min_mppt`` or ``v_max_mppt``. Splitting a
      field into N strings vs one string therefore has **no effect on
      the energy output** — the array is treated as a single DC source.
    - Temperature derating, mismatch losses, soiling.
    - AC-coupled hybrid topologies with a second inverter for the
      battery (battery is always DC-side of the same inverter).

    These omissions are intentional: the project's focus is the
    *economic* uncertainty of a 20-year residential investment, not the
    physical precision of a single hourly kWh. A future Phase 9-bis will
    couple ``PVModelSingleDiode`` to this dispatcher when the project
    needs real datasheet-driven sizing.
    """

    def __init__(self, p_ac_max_kw: float, p_dc_max_kw: float | None = None) -> None:
        """
        Initialize an AC inverter with power limit.

        Args:
            p_ac_max_kw: Maximum AC output power in kW.
            p_dc_max_kw: Optional cap on DC input power in kW. When ``None``
                no DC-side cap is enforced (only ``p_ac_max_kw`` matters).
        """
        self.p_ac_max_kw = p_ac_max_kw
        self.p_dc_max_kw = p_dc_max_kw

    def dispatch(
        self,
        p_pv_dc_kw: float,
        p_load_kw: float,
        battery: BatteryBank,
    ) -> Tuple[float, float, float, float, float, float, float]:
        """Dispatch one hour of PV, load and battery, splitting the surplus.

        Runs the fixed self-consumption priority order for a single hourly
        step and resolves the fate of every kWh the array produces, so that
        no PV energy is silently lost: whatever is neither self-consumed nor
        stored is split into an *exportable* part (fed to the grid, to be
        valued under dedicated withdrawal / ritiro dedicato) and a *curtailed*
        part (physically clipped and worth nothing).

        Priority order:

        1. **PV → load** directly, up to ``min(load, p_ac_max)``.
        2. **PV → battery** with the DC remainder (DC-coupled charging, so it
           is not limited by the AC ceiling, only by the battery's own
           charge-power and capacity headroom).
        3. **Battery → load** for any residual load, within the remaining AC
           headroom.
        4. **Grid → load** for whatever load is still unmet.
        5. **PV → grid** for the surplus left after steps 1-2, bounded by the
           inverter's remaining AC headroom; the rest is **curtailed**.

        The exportable surplus must leave through the same inverter as the
        self-consumed and battery-discharge flows, so it competes with them
        for the AC rating: ``p_pv_direct + p_batt_discharge + p_export
        <= p_ac_max``. With a deliberately undersized inverter (a common
        residential choice) the export term is therefore capped and the
        balance of the surplus is curtailed.

        The model is lossless DC<->AC (1:1), consistent with the rest of this
        simple inverter, so the per-hour energy balance closes exactly:
        ``e_pv_prod == e_pv_direct + e_pv_to_batt + e_pv_to_grid
        + e_pv_curtailed``.

        Args:
            p_pv_dc_kw: PV array DC power for this hour in kW (>= 0). Capped
                to ``p_dc_max_kw`` first when that limit is set.
            p_load_kw: Household load for this hour in kW (>= 0).
            battery: The shared :class:`BatteryBank`; mutated in place by the
                charge/discharge calls (SOC and SOH evolve).

        Returns:
            Tuple of seven floats, all energies in kWh over the one-hour step:
            ``(e_pv_prod, e_pv_direct, e_batt_discharge_to_load,
            e_grid_to_load, e_pv_to_batt, e_pv_to_grid, e_pv_curtailed)``.

        Notes:
            ``e_pv_to_grid`` and ``e_pv_curtailed`` are both >= 0 and sum to
            the surplus ``e_pv_prod - e_pv_direct - e_pv_to_batt``. When the
            inverter has ample AC headroom the curtailed term is zero; when
            the ceiling is fully occupied it absorbs the whole surplus.
        """
        dt = 1.0
        p_ac_max = self.p_ac_max_kw

        if self.p_dc_max_kw is not None:
            p_pv_dc_kw = min(p_pv_dc_kw, self.p_dc_max_kw)

        e_pv_prod = p_pv_dc_kw * dt

        p_pv_ac = min(p_pv_dc_kw, p_load_kw, p_ac_max)
        e_pv_direct = p_pv_ac * dt

        p_pv_remaining_dc = max(0.0, p_pv_dc_kw - p_pv_ac)
        e_pv_remaining_dc = p_pv_remaining_dc * dt
        e_pv_to_batt = battery.charge(e_pv_remaining_dc)

        p_load_residual = max(0.0, p_load_kw - p_pv_ac)

        p_margin_for_batt = max(0.0, p_ac_max - p_pv_ac)
        p_batt_discharge_req = min(p_load_residual, p_margin_for_batt)
        e_batt_discharge_req = p_batt_discharge_req * dt
        e_batt_discharge_to_load = battery.discharge(e_batt_discharge_req)

        e_grid_to_load = (p_load_residual * dt) - e_batt_discharge_to_load

        # Surplus = PV neither self-consumed nor stored. It leaves through the
        # inverter, so the exportable part is bounded by the AC headroom left
        # after PV-direct and any battery discharge; the rest is curtailed.
        e_pv_surplus = max(0.0, e_pv_remaining_dc - e_pv_to_batt)
        p_ac_used = p_pv_ac + e_batt_discharge_to_load / dt
        e_ac_headroom = max(0.0, p_ac_max - p_ac_used) * dt
        e_pv_to_grid = min(e_pv_surplus, e_ac_headroom)
        e_pv_curtailed = e_pv_surplus - e_pv_to_grid

        return (
            e_pv_prod,
            e_pv_direct,
            e_batt_discharge_to_load,
            e_grid_to_load,
            e_pv_to_batt,
            e_pv_to_grid,
            e_pv_curtailed,
        )
