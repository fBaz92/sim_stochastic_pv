from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatterySpecs:
    """
    Battery specifications for capacity and cycle life.
    
    Attributes:
        capacity_kwh: Nominal capacity of a single battery in kWh.
        cycles_life: Expected number of charge/discharge cycles before
            reaching end of life (SoH = 0).
    """
    capacity_kwh: float
    cycles_life: int = 6000


class BatteryBank:
    """
    Banco di n_batteries identiche.

    Conteggio cicli:
      cycles_per_batt = (throughput_discharge_per_batt_kwh / capacity_single_kwh)
      SoH = max(0, 1 - cycles_per_batt / cycles_life)
    """

    def __init__(
        self,
        specs: BatterySpecs,
        n_batteries: int,
        soc_init: float = 0.5,
        eta_charge: float = 0.95,
        eta_discharge: float = 0.95,
        max_charge_kw: float | None = None,
        max_discharge_kw: float | None = None,
        dt_hours: float = 1.0,
    ) -> None:
        """
        Initialize a battery bank with multiple identical batteries.
        
        Args:
            specs: Battery specifications (capacity and cycle life).
            n_batteries: Number of batteries in the bank.
            soc_init: Initial state of charge as fraction (0-1).
            eta_charge: Charging efficiency (0-1).
            eta_discharge: Discharging efficiency (0-1).
            max_charge_kw: Maximum charging power per battery in kW (None for unlimited).
            max_discharge_kw: Maximum discharging power per battery in kW (None for unlimited).
            dt_hours: Time step duration in hours.
        """
        self.specs = specs
        self.n_batteries = n_batteries
        self.capacity_single_kwh = specs.capacity_kwh
        self.capacity_bank_kwh = n_batteries * specs.capacity_kwh

        self.eta_charge = eta_charge
        self.eta_discharge = eta_discharge
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.dt_hours = dt_hours

        self.soc_kwh = soc_init * self.capacity_bank_kwh
        self.throughput_discharge_bank_kwh = 0.0
        self.soh = 1.0

    def reset(self, soc_init: float = 0.5) -> None:
        """
        Reset the battery bank to initial state.
        
        Args:
            soc_init: Initial state of charge as fraction (0-1).
        """
        self.soc_kwh = soc_init * self.capacity_bank_kwh
        self.throughput_discharge_bank_kwh = 0.0
        self.soh = 1.0

    def _update_soh(self, discharge_dc_kwh: float) -> None:
        """
        Update battery state of health based on discharge throughput.
        
        Args:
            discharge_dc_kwh: DC energy discharged in this time step (kWh).
        """
        if discharge_dc_kwh <= 0.0:
            return

        self.throughput_discharge_bank_kwh += discharge_dc_kwh
        throughput_per_batt_kwh = self.throughput_discharge_bank_kwh / self.n_batteries
        cycles_per_batt = throughput_per_batt_kwh / self.capacity_single_kwh
        self.soh = max(0.0, 1.0 - cycles_per_batt / self.specs.cycles_life)

    def charge(self, energy_dc_kwh: float) -> float:
        """
        Charge the battery bank with DC energy.
        
        Args:
            energy_dc_kwh: Available DC energy for charging (kWh).
        
        Returns:
            DC energy actually used for charging (kWh), accounting for
            efficiency losses and capacity limits.
        """
        if energy_dc_kwh <= 0.0:
            return 0.0

        if self.max_charge_kw is not None:
            max_energy = self.max_charge_kw * self.dt_hours
            energy_dc_kwh = min(energy_dc_kwh, max_energy)

        energy_storable_ac = energy_dc_kwh * self.eta_charge
        available_capacity = self.capacity_bank_kwh - self.soc_kwh
        stored_energy_ac = min(energy_storable_ac, available_capacity)
        energy_used_dc = stored_energy_ac / self.eta_charge

        self.soc_kwh += stored_energy_ac
        return energy_used_dc

    def discharge(self, energy_ac_requested_kwh: float) -> float:
        """
        Discharge the battery bank to supply AC energy.
        
        Args:
            energy_ac_requested_kwh: Requested AC energy (kWh).
        
        Returns:
            AC energy actually supplied (kWh), accounting for efficiency
            losses, capacity limits, and power constraints.
        """
        if energy_ac_requested_kwh <= 0.0:
            return 0.0

        if self.max_discharge_kw is not None:
            max_energy = self.max_discharge_kw * self.dt_hours
            energy_ac_requested_kwh = min(energy_ac_requested_kwh, max_energy)

        energy_available_ac = self.soc_kwh * self.eta_discharge
        energy_served_ac = min(energy_ac_requested_kwh, energy_available_ac)

        energy_dc = energy_served_ac / self.eta_discharge
        self.soc_kwh -= energy_dc
        self._update_soh(discharge_dc_kwh=energy_dc)
        return energy_served_ac

    def soc_fraction(self) -> float:
        """
        Get the current state of charge as a fraction.
        
        Returns:
            State of charge as fraction (0-1), where 0 is empty and 1 is full.
        """
        if self.capacity_bank_kwh <= 0.0:
            return 0.0
        return self.soc_kwh / self.capacity_bank_kwh
