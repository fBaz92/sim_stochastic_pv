"""
Battery specifications and degradation-aware storage simulation.

Contains the :class:`BatterySpecs` dataclass describing individual modules
and the :class:`BatteryBank` aggregate that manages SoC/SoH while applying
power limits and efficiency losses.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatterySpecs:
    """
    Battery specifications for capacity and cycle life degradation modeling.

    Encapsulates the key parameters needed to model battery energy storage
    and degradation over the system lifetime. Used to initialize BatteryBank
    instances and determine State of Health (SoH) degradation.

    The degradation model assumes linear capacity fade based on cumulative
    discharge throughput (in full cycle equivalents). State of Health reaches
    0 when cumulative cycles equal cycles_life and linearly reduces both the
    usable capacity (down to capacity_eol_fraction of nominal) and the
    round-trip efficiencies (down to efficiency_eol_factor of their nominal
    values).

    Attributes:
        capacity_kwh: Nominal usable capacity of a single battery module (kWh).
            This is the rated energy capacity at beginning of life (SoH = 1.0),
            typically the manufacturer's specified usable capacity (not total
            cell capacity). Common values: 5-15 kWh for residential systems.
        cycles_life: Expected cycle life before reaching end of life (integer).
            Number of full charge/discharge cycles before battery reaches 0% SoH
            (complete degradation). Typical values:
            - LiFePO4 (LFP): 6000-10000 cycles
            - Li-ion NMC/NCA: 3000-6000 cycles
            - Lead-acid: 500-2000 cycles
            Defaults to 6000 (conservative Li-ion assumption).

    Example:
        ```python
        # Tesla Powerwall 2 specifications
        powerwall_specs = BatterySpecs(
            capacity_kwh=13.5,  # Usable capacity
            cycles_life=5000    # Conservative estimate
        )

        # Budget LFP battery
        lfp_specs = BatterySpecs(
            capacity_kwh=5.12,
            cycles_life=8000
        )
        ```

    Notes:
        - capacity_kwh should be usable capacity, not nominal cell capacity
        - Degradation model is simplified linear fade (real batteries degrade non-linearly)
        - cycles_life significantly affects economic viability of battery systems
        - Higher cycles_life reduces amortized cost per kWh cycled
    """
    capacity_kwh: float
    cycles_life: int = 6000


class BatteryBank:
    """
    Battery energy storage system with degradation tracking.

    Models a bank of n_batteries identical battery modules operating as a single
    energy storage system. Tracks state of charge (SoC), state of health (SoH),
    and cumulative discharge throughput for degradation modeling.

    The battery bank simulates realistic energy storage behavior including:
    - Round-trip efficiency losses (separate charge/discharge efficiencies)
    - Power constraints (maximum charge/discharge rates)
    - Capacity limits (cannot overcharge or over-discharge)
    - Linear degradation based on cumulative cycling

    Degradation Model:
        Cycle counting:
            cycles_per_battery = (total_discharge_kwh / n_batteries) / capacity_single_kwh
        State of Health:
            SoH = max(0, 1 - cycles_per_battery / cycles_life)

        SoH represents remaining capacity fraction (1.0 = new, 0.0 = end of life).
        Currently SoH is tracked but not applied to capacity (future enhancement).

    Energy Flow:
        Charging:  DC_in → [losses] → stored AC energy → battery SoC increases
        Discharging: battery SoC decreases → [losses] → AC energy out

    Attributes:
        specs: Battery specifications (capacity and cycle life).
        n_batteries: Number of identical battery modules in the bank.
        capacity_single_kwh: Usable capacity of one battery module (kWh).
        capacity_bank_kwh: Total usable capacity of the entire bank (kWh).
        eta_charge: Charging efficiency as fraction (0-1). Energy lost during charge.
        eta_discharge: Discharging efficiency as fraction (0-1). Energy lost during discharge.
        max_charge_kw: Maximum charging power limit (kW), or None for unlimited.
        max_discharge_kw: Maximum discharging power limit (kW), or None for unlimited.
        dt_hours: Simulation time step duration (hours). Used for power limit calculations.
        soc_kwh: Current state of charge in kWh (0 to capacity_bank_kwh).
        throughput_discharge_bank_kwh: Cumulative DC energy discharged across all batteries (kWh).
        soh: Current state of health as fraction (0-1). 1.0 = new, 0.0 = degraded.

    Example:
        ```python
        # Create battery bank with 2x Powerwall batteries
        specs = BatterySpecs(capacity_kwh=13.5, cycles_life=5000)
        bank = BatteryBank(
            specs=specs,
            n_batteries=2,
            soc_init=0.5,          # Start at 50% charge
            eta_charge=0.95,       # 95% charging efficiency
            eta_discharge=0.95,    # 95% discharging efficiency
            max_charge_kw=5.0,     # 5 kW max charge rate per battery
            max_discharge_kw=5.0,  # 5 kW max discharge rate per battery
            dt_hours=1.0           # 1-hour time steps
        )

        # Total capacity: 2 × 13.5 = 27 kWh
        # Initial SoC: 0.5 × 27 = 13.5 kWh stored

        # Charge with excess PV (10 kWh available)
        used = bank.charge(10.0)  # Returns DC energy used
        # Stored: min(10 × 0.95, remaining capacity)

        # Discharge to meet load (5 kWh needed)
        supplied = bank.discharge(5.0)  # Returns AC energy supplied
        # Updates SoC and degradation (SoH)
        ```

    Notes:
        - All energy values are in kWh, power values in kW
        - Charge/discharge efficiencies are separate (not round-trip)
        - Round-trip efficiency = eta_charge × eta_discharge (typically ~0.90)
        - Power limits apply per battery, not bank total
        - SoH tracking is currently informational (doesn't reduce capacity yet)
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
        capacity_eol_fraction: float = 0.6,
        efficiency_eol_factor: float = 0.9,
    ) -> None:
        """
        Initialize a battery bank with multiple identical battery modules.

        Creates a battery energy storage system by combining n_batteries identical
        modules. The bank operates as a single unit with combined capacity and
        shared state tracking.

        Args:
            specs: Battery specifications defining capacity and cycle life.
                See BatterySpecs for details.
            n_batteries: Number of identical battery modules (must be >= 1).
                Total capacity = n_batteries × specs.capacity_kwh.
            soc_init: Initial state of charge as fraction (0-1).
                0.0 = completely empty, 1.0 = completely full, 0.5 = half charged.
                Defaults to 0.5 for balanced initial state.
            eta_charge: Charging efficiency (0-1). Fraction of input DC energy
                actually stored. Typical values: 0.92-0.98 for Li-ion.
                Defaults to 0.95 (5% charging losses).
            eta_discharge: Discharging efficiency (0-1). Fraction of stored energy
                delivered as AC output. Typical values: 0.92-0.98 for Li-ion.
                Defaults to 0.95 (5% discharging losses).
            max_charge_kw: Maximum charging power per battery module (kW).
                None means unlimited. Constrains charge rate to prevent damage.
                Typical values: 0.5C to 1.0C rate (e.g., 5-10 kW for 10 kWh battery).
            max_discharge_kw: Maximum discharging power per battery module (kW).
                None means unlimited. Constrains discharge rate.
                Typical values: 0.5C to 2.0C rate.
            dt_hours: Simulation time step duration (hours). Used to convert
                power limits (kW) to energy limits (kWh) for each step.
                Typical values: 1.0 (hourly) or 0.25 (15-minute).
            capacity_eol_fraction: Fraction of nominal capacity still usable
                when SoH reaches zero. Defaults to 0.6 (60% of new capacity).
            efficiency_eol_factor: Multiplier applied to eta_charge and
                eta_discharge when SoH reaches zero (linear interpolation with
                SoH). Defaults to 0.9 (i.e., 10% additional losses at end of life).

        Example:
            ```python
            # Typical residential battery system
            specs = BatterySpecs(capacity_kwh=10.0, cycles_life=6000)
            bank = BatteryBank(
                specs=specs,
                n_batteries=1,
                soc_init=0.2,           # Start nearly empty (morning)
                eta_charge=0.96,        # High-quality battery
                eta_discharge=0.96,
                max_charge_kw=5.0,      # 0.5C charge rate
                max_discharge_kw=10.0,  # 1.0C discharge rate
                dt_hours=1.0
            )
            ```

        Notes:
            - Total bank capacity = n_batteries × specs.capacity_kwh
            - Initial energy stored = soc_init × capacity_bank_kwh
            - Efficiency losses are one-way (separate for charge/discharge) and
              decrease further as SoH degrades
            - Power limits prevent battery damage and follow manufacturer specs
            - dt_hours should match simulation time step for accurate power limiting
        """
        if not 0.0 < capacity_eol_fraction <= 1.0:
            raise ValueError("capacity_eol_fraction must be within (0, 1]")
        if not 0.0 < efficiency_eol_factor <= 1.0:
            raise ValueError("efficiency_eol_factor must be within (0, 1]")

        self.specs = specs
        self.n_batteries = n_batteries
        self.capacity_single_kwh = specs.capacity_kwh
        self.capacity_nominal_kwh = n_batteries * specs.capacity_kwh
        self.capacity_bank_kwh = self.capacity_nominal_kwh

        self._eta_charge_base = eta_charge
        self._eta_discharge_base = eta_discharge
        self.capacity_eol_fraction = capacity_eol_fraction
        self.efficiency_eol_factor = efficiency_eol_factor
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.dt_hours = dt_hours

        self.soc_kwh = soc_init * self.capacity_bank_kwh
        self.throughput_discharge_bank_kwh = 0.0
        self.soh = 1.0
        self._recompute_capacity_after_degradation()

    def reset(self, soc_init: float = 0.5) -> None:
        """
        Reset the battery bank to initial state for a new simulation run.

        Clears all accumulated state (charge level, throughput, degradation)
        and reinitializes the battery as if brand new. Used between Monte Carlo
        simulation paths to ensure independent runs.

        Args:
            soc_init: Initial state of charge as fraction (0-1).
                0.0 = empty, 1.0 = full. Defaults to 0.5 (half charged).
                This represents the assumed initial charge level at the
                start of the simulation period.

        Example:
            ```python
            bank = BatteryBank(specs, n_batteries=2)

            # Run simulation path 1
            for step in range(8760):  # Yearly simulation
                # ... charge/discharge operations ...
                pass

            # Bank is now degraded with SoH < 1.0

            # Reset for next Monte Carlo path
            bank.reset(soc_init=0.5)
            # Now: SoH = 1.0, throughput = 0, SoC = 50%
            ```

        Notes:
            - Resets SoC to soc_init × capacity_bank_kwh
            - Clears cumulative discharge throughput (back to 0)
            - Resets State of Health to 1.0 (brand new battery)
            - Does NOT change battery specifications or configuration
            - Essential for Monte Carlo simulations with independent paths
        """
        self.capacity_bank_kwh = self.capacity_nominal_kwh
        self.soc_kwh = soc_init * self.capacity_bank_kwh
        self.throughput_discharge_bank_kwh = 0.0
        self.soh = 1.0
        self._recompute_capacity_after_degradation()

    def _update_soh(self, discharge_dc_kwh: float) -> None:
        """
        Update battery state of health based on discharge throughput.

        Internal method called automatically after each discharge operation.
        Tracks cumulative discharge energy and calculates equivalent full cycles
        to determine battery degradation (State of Health).

        The degradation model assumes:
        - Only discharge cycles contribute to wear (charging doesn't degrade)
        - Linear capacity fade with cumulative throughput
        - Throughput is shared equally across all batteries in the bank
        - SoH reaches 0 when cumulative cycles equal specs.cycles_life

        Args:
            discharge_dc_kwh: DC energy discharged in this time step (kWh).
                This is the energy removed from battery storage (before
                discharge efficiency losses). Automatically provided by
                the discharge() method.

        Example:
            ```python
            # This method is called internally - users don't call it directly
            bank = BatteryBank(specs, n_batteries=2)

            # discharge() automatically calls _update_soh()
            bank.discharge(5.0)  # Internally: _update_soh(discharge_dc_kwh=...)

            # Check degradation
            print(f"SoH: {bank.soh:.3f}")  # e.g., 0.998 after one cycle
            ```

        Notes:
            - Only discharge operations update SoH (charge doesn't)
            - SoH is floored at 0.0 (cannot go negative)
            - Throughput is cumulative across battery lifetime
            - SoH currently informational (doesn't reduce capacity yet)
            - Future enhancement: apply SoH multiplier to effective capacity
        """
        if discharge_dc_kwh <= 0.0:
            return

        self.throughput_discharge_bank_kwh += discharge_dc_kwh
        throughput_per_batt_kwh = self.throughput_discharge_bank_kwh / self.n_batteries
        cycles_per_batt = throughput_per_batt_kwh / self.capacity_single_kwh
        self.soh = max(0.0, 1.0 - cycles_per_batt / self.specs.cycles_life)
        self._recompute_capacity_after_degradation()

    def _capacity_scale(self) -> float:
        """
        Calculate capacity scaling factor based on current State of Health.

        Computes the multiplier applied to nominal capacity to account for
        degradation. Uses linear interpolation between end-of-life capacity
        fraction and full capacity based on SoH.

        Returns:
            float: Capacity scaling factor (capacity_eol_fraction to 1.0).
                - SoH = 1.0 (new): returns 1.0 (full nominal capacity)
                - SoH = 0.0 (degraded): returns capacity_eol_fraction (e.g., 0.6)
                - SoH = 0.5 (half-life): returns midpoint interpolation

        Formula:
            scale = capacity_eol_fraction + (1 - capacity_eol_fraction) × SoH

        Example:
            With capacity_eol_fraction=0.6:
            - SoH=1.0 → scale=1.0 (100% capacity)
            - SoH=0.5 → scale=0.8 (80% capacity)
            - SoH=0.0 → scale=0.6 (60% capacity)

        Notes:
            - Internal helper for _recompute_capacity_after_degradation()
            - Linear degradation model (simplified vs real battery behavior)
            - Called automatically after SoH updates
        """
        return self.capacity_eol_fraction + (1.0 - self.capacity_eol_fraction) * self.soh

    def _efficiency_scale(self) -> float:
        """
        Calculate efficiency scaling factor based on current State of Health.

        Computes the multiplier applied to base charging/discharging efficiencies
        to account for degradation. Uses linear interpolation between end-of-life
        efficiency factor and full efficiency based on SoH.

        Returns:
            float: Efficiency scaling factor (efficiency_eol_factor to 1.0).
                - SoH = 1.0 (new): returns 1.0 (full base efficiency)
                - SoH = 0.0 (degraded): returns efficiency_eol_factor (e.g., 0.9)
                - SoH = 0.5 (half-life): returns midpoint interpolation

        Formula:
            scale = efficiency_eol_factor + (1 - efficiency_eol_factor) × SoH

        Example:
            With efficiency_eol_factor=0.9, base eta_charge=0.95:
            - SoH=1.0 → scale=1.0 → actual_eta=0.95×1.0=0.950 (95.0%)
            - SoH=0.5 → scale=0.95 → actual_eta=0.95×0.95=0.9025 (90.25%)
            - SoH=0.0 → scale=0.9 → actual_eta=0.95×0.9=0.855 (85.5%)

        Notes:
            - Internal helper for eta_charge and eta_discharge properties
            - Models increasing resistance and losses as battery ages
            - Applied to both charge and discharge efficiencies
            - Called on every efficiency property access
        """
        return self.efficiency_eol_factor + (1.0 - self.efficiency_eol_factor) * self.soh

    def _recompute_capacity_after_degradation(self) -> None:
        """
        Update effective battery capacity based on current degradation.

        Recalculates the usable bank capacity by applying the SoH-based
        capacity scaling factor to the nominal capacity. Also ensures the
        current SoC doesn't exceed the reduced capacity (clipping if needed).

        This method is called automatically whenever SoH changes (after
        discharge operations or reset). It implements the capacity fade
        effect of battery degradation.

        Side Effects:
            - Updates capacity_bank_kwh to degraded value
            - Clips soc_kwh to new capacity if needed (prevents overflow)

        Example:
            ```python
            bank = BatteryBank(
                specs=BatterySpecs(capacity_kwh=10.0, cycles_life=6000),
                n_batteries=1,
                soc_init=1.0,  # Start full: 10 kWh
                capacity_eol_fraction=0.6
            )

            # Initial: capacity_bank_kwh = 10.0, SoH = 1.0

            # After heavy cycling...
            # (discharge many cycles, SoH degrades to 0.5)

            # Internal call: _recompute_capacity_after_degradation()
            # capacity_scale = 0.6 + (1-0.6)×0.5 = 0.8
            # capacity_bank_kwh = 10.0 × 0.8 = 8.0 kWh
            # soc_kwh = min(10.0, 8.0) = 8.0 kWh (clipped)
            ```

        Notes:
            - Internal method (not part of public API)
            - Called after _update_soh() and reset()
            - SoC clipping prevents impossible states (SoC > capacity)
            - Linear degradation model (simplified)
            - Capacity reduction reflects real-world battery aging
        """
        self.capacity_bank_kwh = self.capacity_nominal_kwh * self._capacity_scale()
        self.soc_kwh = min(self.soc_kwh, self.capacity_bank_kwh)

    @property
    def eta_charge(self) -> float:
        """
        Current charging efficiency accounting for degradation.

        Returns the effective charging efficiency that includes both the
        base efficiency and degradation effects. As the battery ages (SoH
        decreases), charging efficiency reduces due to increased internal
        resistance and losses.

        Returns:
            float: Effective charging efficiency (0-1).
                - New battery (SoH=1.0): returns base efficiency (e.g., 0.95)
                - Degraded battery (SoH=0.0): returns reduced efficiency
                  (e.g., 0.95 × 0.9 = 0.855 with efficiency_eol_factor=0.9)
                - Mid-life (SoH=0.5): returns interpolated value

        Formula:
            eta_charge = base_eta_charge × efficiency_scale(SoH)

        Example:
            ```python
            bank = BatteryBank(
                specs=BatterySpecs(capacity_kwh=10.0, cycles_life=6000),
                n_batteries=1,
                eta_charge=0.96,
                efficiency_eol_factor=0.90
            )

            # New battery
            print(f"Eta charge: {bank.eta_charge:.3f}")  # 0.960

            # Simulate heavy use (many discharge cycles)
            # ... SoH degrades to 0.5 ...
            print(f"Eta charge: {bank.eta_charge:.3f}")  # ~0.931

            # End of life (SoH=0.0)
            # ... SoH degrades to 0.0 ...
            print(f"Eta charge: {bank.eta_charge:.3f}")  # 0.864
            ```

        Notes:
            - Dynamic property (recalculated on every access)
            - Reflects aging through reduced efficiency
            - Used by charge() method to calculate stored energy
            - Round-trip efficiency = eta_charge × eta_discharge
            - Both efficiencies degrade together
        """
        return self._eta_charge_base * self._efficiency_scale()

    @property
    def eta_discharge(self) -> float:
        """
        Current discharging efficiency accounting for degradation.

        Returns the effective discharging efficiency that includes both the
        base efficiency and degradation effects. As the battery ages (SoH
        decreases), discharging efficiency reduces due to increased internal
        resistance and losses.

        Returns:
            float: Effective discharging efficiency (0-1).
                - New battery (SoH=1.0): returns base efficiency (e.g., 0.95)
                - Degraded battery (SoH=0.0): returns reduced efficiency
                  (e.g., 0.95 × 0.9 = 0.855 with efficiency_eol_factor=0.9)
                - Mid-life (SoH=0.5): returns interpolated value

        Formula:
            eta_discharge = base_eta_discharge × efficiency_scale(SoH)

        Example:
            ```python
            bank = BatteryBank(
                specs=BatterySpecs(capacity_kwh=10.0, cycles_life=6000),
                n_batteries=1,
                eta_discharge=0.96,
                efficiency_eol_factor=0.90
            )

            # New battery
            print(f"Eta discharge: {bank.eta_discharge:.3f}")  # 0.960

            # Simulate heavy use (many discharge cycles)
            # ... SoH degrades to 0.5 ...
            print(f"Eta discharge: {bank.eta_discharge:.3f}")  # ~0.931

            # End of life (SoH=0.0)
            # ... SoH degrades to 0.0 ...
            print(f"Eta discharge: {bank.eta_discharge:.3f}")  # 0.864

            # Impact on round-trip efficiency
            rte_new = 0.96 * 0.96  # ~0.922 (92.2%)
            rte_eol = 0.864 * 0.864  # ~0.746 (74.6%)
            print(f"Round-trip degradation: {rte_new:.1%} → {rte_eol:.1%}")
            ```

        Notes:
            - Dynamic property (recalculated on every access)
            - Reflects aging through reduced efficiency
            - Used by discharge() method to calculate deliverable energy
            - Round-trip efficiency = eta_charge × eta_discharge
            - Both efficiencies degrade symmetrically
        """
        return self._eta_discharge_base * self._efficiency_scale()

    def charge(self, energy_dc_kwh: float) -> float:
        """
        Charge the battery bank with available DC energy from PV or grid.

        Attempts to store the provided DC energy in the battery, subject to:
        - Charging efficiency losses (eta_charge)
        - Remaining capacity (cannot overcharge)
        - Power rate limits (max_charge_kw)

        The charging process:
        1. Apply power limit if max_charge_kw is set
        2. Calculate storable energy after efficiency losses
        3. Limit to available capacity (full battery stops accepting charge)
        4. Update battery SoC with stored energy
        5. Return the DC energy actually consumed from source

        Args:
            energy_dc_kwh: Available DC energy for charging (kWh).
                This is typically excess PV production after meeting load.
                Must be non-negative. Zero or negative values are no-ops.

        Returns:
            float: DC energy actually consumed for charging (kWh).
                This is the amount of input energy used (before losses).
                Always <= energy_dc_kwh (cannot use more than available).
                Can be less than requested due to:
                - Battery approaching full (limited remaining capacity)
                - Power rate limits (max_charge_kw constraint)
                Returned value helps track unused excess energy (curtailment).

        Example:
            ```python
            bank = BatteryBank(
                specs=BatterySpecs(capacity_kwh=10.0, cycles_life=6000),
                n_batteries=1,
                soc_init=0.3,  # 3 kWh currently stored
                eta_charge=0.95,
                max_charge_kw=5.0,
                dt_hours=1.0
            )

            # Excess PV: 8 kWh available
            used = bank.charge(8.0)

            # Charging calculation:
            # - Max power limit: 5.0 kW × 1.0 h = 5.0 kWh (limits input)
            # - Storable after losses: 5.0 × 0.95 = 4.75 kWh
            # - Available capacity: 10.0 - 3.0 = 7.0 kWh (plenty of room)
            # - Stored: 4.75 kWh
            # - DC used: 4.75 / 0.95 = 5.0 kWh
            # - Unused excess: 8.0 - 5.0 = 3.0 kWh (curtailed or exported)

            print(f"Used: {used:.2f} kWh")  # 5.00 kWh
            print(f"SoC: {bank.soc_kwh:.2f} kWh")  # 7.75 kWh
            ```

        Notes:
            - Input is DC energy (pre-conversion), output is also DC
            - Efficiency losses mean stored energy < input energy
            - With eta_charge=0.95, 5% energy lost as heat during charging
            - Power limits convert to energy limits using dt_hours
            - Battery SoC cannot exceed capacity_bank_kwh
            - Does NOT update SoH (charging doesn't degrade battery)
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
        Discharge the battery bank to supply AC energy to meet load demand.

        Attempts to provide the requested AC energy from battery storage,
        subject to:
        - Discharging efficiency losses (eta_discharge)
        - Available stored energy (cannot over-discharge)
        - Power rate limits (max_discharge_kw)

        The discharging process:
        1. Apply power limit if max_discharge_kw is set
        2. Calculate available AC energy after efficiency losses
        3. Limit to requested amount (don't supply more than needed)
        4. Update battery SoC by removing DC energy
        5. Update State of Health based on discharge throughput
        6. Return the AC energy actually supplied to load

        Args:
            energy_ac_requested_kwh: Requested AC energy to supply (kWh).
                This is typically unmet load demand after PV production.
                Must be non-negative. Zero or negative values are no-ops.

        Returns:
            float: AC energy actually supplied to load (kWh).
                This is the usable output energy (after discharge losses).
                Always <= energy_ac_requested_kwh (cannot supply more than requested).
                Can be less than requested due to:
                - Battery depleted (limited stored energy)
                - Power rate limits (max_discharge_kw constraint)
                - Discharge efficiency losses
                Returned value < requested means load is not fully met.

        Example:
            ```python
            bank = BatteryBank(
                specs=BatterySpecs(capacity_kwh=10.0, cycles_life=6000),
                n_batteries=1,
                soc_init=0.7,  # 7 kWh currently stored
                eta_discharge=0.95,
                max_discharge_kw=5.0,
                dt_hours=1.0
            )

            # Load demand: 4 kWh needed
            supplied = bank.discharge(4.0)

            # Discharging calculation:
            # - Max power limit: 5.0 kW × 1.0 h = 5.0 kWh AC (not limiting)
            # - Available AC: 7.0 × 0.95 = 6.65 kWh
            # - Requested: 4.0 kWh (less than available)
            # - Supplied AC: 4.0 kWh
            # - DC removed: 4.0 / 0.95 = 4.21 kWh
            # - New SoC: 7.0 - 4.21 = 2.79 kWh
            # - SoH updated based on 4.21 kWh discharge

            print(f"Supplied: {supplied:.2f} kWh")  # 4.00 kWh
            print(f"SoC: {bank.soc_kwh:.2f} kWh")  # 2.79 kWh
            print(f"SoH: {bank.soh:.4f}")  # Slightly degraded
            ```

        Example (battery depleted):
            ```python
            bank = BatteryBank(
                specs=BatterySpecs(capacity_kwh=10.0, cycles_life=6000),
                n_batteries=1,
                soc_init=0.1,  # Only 1 kWh stored (nearly empty)
                eta_discharge=0.95
            )

            # Load needs 5 kWh but battery nearly empty
            supplied = bank.discharge(5.0)

            # Available AC: 1.0 × 0.95 = 0.95 kWh
            # Can only supply 0.95 kWh (not enough!)
            # Unmet load: 5.0 - 0.95 = 4.05 kWh (must be drawn from grid)

            print(f"Supplied: {supplied:.2f} kWh")  # 0.95 kWh
            print(f"Unmet: {5.0 - supplied:.2f} kWh")  # 4.05 kWh
            ```

        Notes:
            - Input request is AC energy, output is also AC
            - Efficiency losses mean more DC removed from storage than AC supplied
            - With eta_discharge=0.95, 5% energy lost during discharge
            - Automatically updates SoH (discharge degrades battery)
            - Battery SoC cannot go negative (floor at 0)
            - Power limits convert to energy limits using dt_hours
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
        Get the current state of charge as a normalized fraction.

        Converts the absolute state of charge (in kWh) to a fractional
        representation between 0 and 1, useful for monitoring and control logic.

        Returns:
            float: State of charge as fraction (0.0 to 1.0).
                - 0.0 = completely empty (no energy stored)
                - 0.5 = half charged
                - 1.0 = completely full (at capacity)
                Returns 0.0 if battery bank has zero capacity (safety check).

        Example:
            ```python
            bank = BatteryBank(
                specs=BatterySpecs(capacity_kwh=10.0, cycles_life=6000),
                n_batteries=2,  # Total: 20 kWh
                soc_init=0.75   # Start at 75% = 15 kWh
            )

            print(f"SoC: {bank.soc_fraction():.1%}")  # 75.0%

            # After some discharge
            bank.discharge(5.0)  # Remove ~5.3 kWh DC
            print(f"SoC: {bank.soc_fraction():.1%}")  # ~48.5%

            # Check if battery needs recharging
            if bank.soc_fraction() < 0.2:
                print("Battery low! Consider charging.")
            ```

        Notes:
            - Returns current SoC / total capacity
            - Useful for battery management and visualization
            - Typically want to avoid deep discharge (< 20%) for longevity
            - Values outside [0, 1] indicate programming errors
            - Zero capacity bank returns 0.0 (safety check)
        """
        if self.capacity_bank_kwh <= 0.0:
            return 0.0
        return self.soc_kwh / self.capacity_bank_kwh
