from __future__ import annotations

from typing import Tuple

from .battery import BatteryBank


class InverterAC:
    """
    Simple AC inverter with a limit on AC output power.
    """

    def __init__(self, p_ac_max_kw: float, p_dc_max_kw: float | None = None) -> None:
        """
        Initialize an AC inverter with power limit.
        
        Args:
            p_ac_max_kw: Maximum AC output power in kW.
        """
        self.p_ac_max_kw = p_ac_max_kw
        self.p_dc_max_kw = p_dc_max_kw

    def dispatch(
        self,
        p_pv_dc_kw: float,
        p_load_kw: float,
        battery: BatteryBank,
    ) -> Tuple[float, float, float, float, float]:
        """
        One-hour time step.
        Returns (all in kWh over the hour):
          - e_pv_prod
          - e_pv_direct
          - e_batt_discharge_to_load
          - e_grid_to_load
          - e_pv_to_batt
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

        return (
            e_pv_prod,
            e_pv_direct,
            e_batt_discharge_to_load,
            e_grid_to_load,
            e_pv_to_batt,
        )
