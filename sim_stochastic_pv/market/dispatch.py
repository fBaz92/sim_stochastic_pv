"""
Vectorized merit-order dispatch engine with inertia and storage constraints.

Implements a four-phase dispatch algorithm:

1. **Merit-order dispatch** (vectorized): sorts generators by SRMC, stacks
   them in order of increasing cost, and dispatches to meet load. The marginal
   price is set by the most expensive dispatched generator at each timestep.
   Interconnection imports, if supplied, enter the merit order as virtual
   generators with time-varying SRMC and availability (NTC paths) — they
   clear naturally alongside domestic units with no special-casing.

2. **Inertia fix** (iterative): checks system inertia against the minimum
   threshold and forces the cheapest offline synchronous generator online if
   inertia is too low. Curtails non-synchronous generation if this causes
   oversupply. Imports do not contribute inertia and therefore do not help
   satisfy ``H_MIN_SECONDS`` — a realistic effect for HVDC links.

3. **Export adjustment** (per-interconnection, per-timestep): for each
   timestep where the domestic marginal price is below a link's export
   floor (``foreign_price - τ``) and the link has available export NTC,
   additional generation is dispatched (up to NTC) from the cheapest
   unused headroom with ``SRMC ≤ export_floor``. The marginal price is
   updated to the SRMC of the last-called unit.

4. **Storage dispatch** (sequential, stateful): a rolling-percentile
   arbitrage strategy over Phase 1–3 baseline prices. When the current
   price is in the cheapest quartile, the battery charges (adding load);
   when in the most expensive quartile, it discharges (displacing the
   marginal unit). SOC evolves sequentially. The marginal price is
   re-evaluated per timestep to reflect the changed load/generation
   balance. BESS units also contribute synthetic inertia when their SOC
   is sufficiently far from the operational bounds; this contribution is
   folded into ``h_system`` at the end.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from sim_stochastic_pv.market.config import (
    H_MIN_SECONDS, P_PEAK_GW, P_BASE, QUARTERS_PER_HOUR,
    STORAGE_PERCENTILE_WINDOW_QH, STORAGE_CHARGE_PERCENTILE,
    STORAGE_DISCHARGE_PERCENTILE,
)
from sim_stochastic_pv.market.generators import Generator
from sim_stochastic_pv.market.interconnections import (
    InterconnectionRealization, VirtualImportGenerator,
)
from sim_stochastic_pv.market.storage import StorageUnit


@dataclass
class InterconnectionMetrics:
    """Per-link cross-border metrics derived from a completed dispatch.

    Uses the **congestion-rent** approximation for the economic benefit:
    every infra-marginal import clears at the domestic marginal price and
    is assumed not to perturb that price when removed. Under this
    first-order assumption the economic benefit is the volume times the
    gap between the domestic clearing price and the import SRMC
    (symmetrically for exports), and is ≥ 0 by construction of the merit
    order. This matches the standard TSO definition of cross-border
    welfare and avoids the 2× cost of re-dispatching the closed system.

    The CO₂ benefit uses the **load-weighted emission intensity of the
    domestic fleet** at each quarter-hour as the counterfactual intensity
    that each imported/exported MWh would have replaced. A positive
    value means the link reduced system emissions (importing from a
    cleaner grid or exporting to a dirtier one); a negative value means
    the opposite. Integrating over the year gives a consumption-based
    carbon budget adjustment.

    Attributes:
        import_hours (np.ndarray): Hours per year in net-import state
            (net_flow > 0), shape ``(n_interconnections,)``.
        export_hours (np.ndarray): Hours per year in net-export state,
            shape ``(n_interconnections,)``.
        import_energy_mwh (np.ndarray): Gross annual import volume per
            link in MWh, shape ``(n_interconnections,)``.
        export_energy_mwh (np.ndarray): Gross annual export volume per
            link in MWh, shape ``(n_interconnections,)``.
        economic_benefit_eur_qh (np.ndarray): Per-link, per-quarter-hour
            economic benefit in EUR, shape ``(n_interconnections, 35040)``.
            Sum of import and export contributions; each contribution is
            ≥ 0 under the congestion-rent convention.
        co2_benefit_tons_qh (np.ndarray): Per-link, per-quarter-hour
            CO₂ benefit in tonnes, shape ``(n_interconnections, 35040)``.
            Can be negative on quarter-hours where we import from an
            area dirtier than the domestic marginal mix.
        total_economic_benefit_eur (np.ndarray): Annual sum of
            :attr:`economic_benefit_eur_qh` per link, shape
            ``(n_interconnections,)``.
        total_co2_benefit_tons (np.ndarray): Annual sum of
            :attr:`co2_benefit_tons_qh` per link, shape
            ``(n_interconnections,)``.
        domestic_marginal_ci_g_per_kwh (np.ndarray): Load-weighted
            emission intensity of the domestic fleet per quarter-hour,
            shape ``(35040,)``, in gCO₂ per kWh_e. Zero on quarter-hours
            where no domestic unit is online (e.g. total unserved energy
            scenarios).
    """

    import_hours: np.ndarray
    export_hours: np.ndarray
    import_energy_mwh: np.ndarray
    export_energy_mwh: np.ndarray
    economic_benefit_eur_qh: np.ndarray
    co2_benefit_tons_qh: np.ndarray
    total_economic_benefit_eur: np.ndarray
    total_co2_benefit_tons: np.ndarray
    domestic_marginal_ci_g_per_kwh: np.ndarray


@dataclass
class DispatchResult:
    """Results of a full-year merit-order dispatch.

    All power arrays are in per-unit of system base. All prices are in
    EUR/MWh (electrical). All emissions are in tonnes CO₂ per
    quarter-hour (territorial) or per link (consumption-based).

    Attributes:
        power (np.ndarray): Dispatched power matrix of shape
            ``(n_units, 35040)`` in per-unit. Rows follow :attr:`gen_names`;
            interconnection imports (if any) appear as additional rows
            with ``gen_type == 'import'``.
        marginal_price (np.ndarray): System marginal price array of shape
            ``(35040,)`` in EUR/MWh. Reflects both Phase 1 clearing, any
            inertia-fix adjustments, and the Phase 3 export re-dispatch.
        curtailment (np.ndarray): Curtailed energy array of shape ``(35040,)``
            in per-unit (energy curtailed due to inertia constraint).
        h_system (np.ndarray): System inertia constant array of shape
            ``(35040,)`` in seconds. Imports are not included in the
            weighted average (they do not provide synchronous inertia).
        unserved (np.ndarray): Unserved energy array of shape ``(35040,)``
            in per-unit (residual load after all dispatchable units and
            imports are exhausted).
        gen_names (list[str]): Names of generators and imports in the
            same order as the rows of :attr:`power`.
        gen_types (list[str]): Matching ``gen_type`` labels (``'gas'``,
            ``'solar'``, ``'import'``, …). Used by downstream aggregators
            to separate territorial from consumption-based accounting.
        emissions (np.ndarray): Territorial CO₂ emissions, shape
            ``(n_units, 35040)``, in tonnes CO₂ per quarter-hour.
            Dimensional analysis: power_pu · P_BASE(GW) · 0.25(h) · 1000
            (MW/GW) · ef(tCO₂/MWh_th) / η = tCO₂. Rows for imports are
            zero under the IPCC territorial convention.
        net_import_pu (np.ndarray): Signed cross-border flow per
            interconnection, shape ``(n_interconnections, 35040)``.
            Positive values mean energy flowing *into* the domestic
            system; negative values mean export out. Zero-length array
            when no interconnections are supplied.
        interconnection_names (list[str]): Names matching the rows of
            :attr:`net_import_pu`. Empty when no interconnections.
        foreign_prices (np.ndarray): Foreign day-ahead price path per
            interconnection, shape ``(n_interconnections, 35040)`` in
            EUR/MWh. Useful for convergence analysis. Same ordering as
            :attr:`interconnection_names`.
        emissions_imported_tons (np.ndarray): Consumption-based emissions
            embedded in net imports, shape ``(n_interconnections, 35040)``
            in tonnes CO₂ per quarter-hour. Computed as
            ``max(net_import, 0) * P_BASE_GW * 0.25 * CI_g/kWh``, which
            collapses pu·GWh · (g/kWh) = pu·10⁶·g = pu·tons (the factor
            10⁶ kWh/GWh cancels the factor 10⁻⁶ ton/g). The export
            portion is not credited as a negative footprint.
        storage_power_pu (np.ndarray): Battery AC-side power per unit per
            timestep, shape ``(n_storage, 35040)``. Sign convention:
            positive = discharge (battery → grid), negative = charge
            (grid → battery). Zero when no storage units are supplied.
        storage_soc (np.ndarray): State of charge per unit per timestep,
            shape ``(n_storage, 35040)``, as a fraction in
            ``[soc_min_frac, soc_max_frac]``.
        storage_names (list[str]): Names matching the rows of
            :attr:`storage_power_pu`.
        price_setter_idx (np.ndarray): Index into :attr:`gen_names` of the
            unit that set the system marginal price at each timestep,
            shape ``(35040,)``, dtype ``int16``. Sentinel ``-1`` when the
            marginal price is zero (all load unserved or no unit dispatched).
            Derived post-hoc from the dispatch outcome at each phase; no
            new random draws are introduced, so reproducibility is preserved.
    """

    power: np.ndarray
    marginal_price: np.ndarray
    curtailment: np.ndarray
    h_system: np.ndarray
    unserved: np.ndarray
    gen_names: list[str]
    emissions: np.ndarray
    gen_types: list[str] = field(default_factory=list)
    net_import_pu: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0)))
    interconnection_names: list[str] = field(default_factory=list)
    foreign_prices: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0)))
    emissions_imported_tons: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0)))
    ic_metrics: 'InterconnectionMetrics | None' = None
    storage_power_pu: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0)))
    storage_soc: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0)))
    storage_names: list[str] = field(default_factory=list)
    price_setter_idx: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int16))


def _redispatch_timestep(
    t: int,
    effective_load: float,
    n_units: int,
    srmc_all: np.ndarray,
    avail_all: np.ndarray,
    power: np.ndarray,
    marginal_price: np.ndarray,
    price_setter_idx: np.ndarray | None = None,
) -> None:
    """Re-run the merit-order stack at a single timestep.

    Updates ``power[:, t]`` and ``marginal_price[t]`` **in place** to
    reflect the changed ``effective_load``. This is the single-timestep
    equivalent of Phase 1 — sort by SRMC, stack, clip, find the marginal
    unit — but operates on pre-computed arrays to avoid re-allocating.

    Called by Phase 4 (storage dispatch) whenever a charge or discharge
    decision changes the effective load at a given timestep. The merit
    order is re-evaluated from scratch (not incrementally) because the
    changed load may shift which units clear and which sit below their
    minimum stable output. This is O(n_units·log(n_units)) per call,
    which is negligible for the ~5–10 units in the typical Italian mix.

    Args:
        t: Timestep index (0 to 35039).
        effective_load: Net load (original load ± storage power) in p.u.
        n_units: Number of units in the merit-order stack.
        srmc_all: SRMC array, shape ``(n_units, n_t)``.
        avail_all: Available power array, shape ``(n_units, n_t)``.
        power: Dispatched power matrix, shape ``(n_units, n_t)``.
            **Modified in place**.
        marginal_price: Marginal price array, shape ``(n_t,)``.
            **Modified in place**.
        price_setter_idx: Optional array of shape ``(n_t,)``. When provided,
            the element ``price_setter_idx[t]`` is updated in place with the
            index of the price-setting unit (or ``-1`` if nothing dispatched).
    """
    srmc_t = srmc_all[:, t]
    avail_t = avail_all[:, t]
    order = np.argsort(srmc_t)
    remaining = effective_load
    for idx in order:
        if remaining <= 0:
            power[idx, t] = 0.0
        else:
            take = min(avail_t[idx], remaining)
            power[idx, t] = take
            remaining -= take
    dispatched_mask = power[:, t] > 0
    if dispatched_mask.any():
        marginal_price[t] = srmc_t[dispatched_mask].max()
        if price_setter_idx is not None:
            dispatched_indices = np.where(dispatched_mask)[0]
            price_setter_idx[t] = dispatched_indices[
                np.argmax(srmc_t[dispatched_indices])]
    else:
        marginal_price[t] = 0.0
        if price_setter_idx is not None:
            price_setter_idx[t] = -1


def dispatch_year(
    generators: list[Generator],
    load: np.ndarray,
    interconnection_realizations: list[InterconnectionRealization] | None = None,
    storage_units: list[StorageUnit] | None = None,
) -> DispatchResult:
    """Run merit-order dispatch for one simulated year.

    The dispatch proceeds in four phases described in the module
    docstring. Optional phases are skipped when their inputs are not
    supplied — preserving full backward compatibility with existing
    callers.

    Args:
        generators: List of :class:`~sim_stochastic_pv.market.generators.Generator` objects
            with :meth:`~sim_stochastic_pv.market.generators.Generator.prepare_run`
            already called.
        load: Load profile array of shape ``(35040,)`` in per-unit.
        interconnection_realizations: Optional list of realized
            interconnections. Each is wrapped in a
            :class:`~sim_stochastic_pv.market.interconnections.VirtualImportGenerator`
            and appended to the merit-order stack; its export path is
            consumed by Phase 3.
        storage_units: Optional list of :class:`StorageUnit` objects. When
            supplied, Phase 4 is executed: a rolling-percentile arbitrage
            strategy charges/discharges the batteries sequentially,
            updating the generation stack and marginal price per timestep.

    Returns:
        DispatchResult: Aggregated dispatch results for the year,
            including cross-border flows and storage arrays when their
            respective inputs are supplied.
    """
    interconnection_realizations = interconnection_realizations or []
    n_domestic = len(generators)
    n_ic = len(interconnection_realizations)

    # Build the full merit-order stack: domestic first, then virtual imports.
    # The ordering is purely a convention — the merit-order algorithm is
    # invariant to the initial ordering (it sorts by SRMC).
    virtual_imports = [r.as_virtual_import_generator()
                       for r in interconnection_realizations]
    units: list = list(generators) + list(virtual_imports)

    n_units = len(units)
    n_t = len(load)

    srmc_all = np.array([u.srmc() for u in units])
    avail_all = np.array([u.available_power_pu() for u in units])
    h_values = np.array([u.h_inertia for u in units])
    is_sync = np.array([u.is_synchronous for u in units])
    min_stable = np.array([u.min_stable_power_pu() for u in units])
    capacity_pu = np.array([u.capacity_pu for u in units])

    # Phase 1: vectorized merit order
    order = np.argsort(srmc_all, axis=0)
    avail_sorted = np.take_along_axis(avail_all, order, axis=0)
    cum_before = np.vstack([np.zeros((1, n_t)), np.cumsum(avail_sorted, axis=0)[:-1]])
    remaining = np.maximum(load[np.newaxis, :] - cum_before, 0)
    dispatched_sorted = np.minimum(avail_sorted, remaining)
    inv_order = np.argsort(order, axis=0)
    power = np.take_along_axis(dispatched_sorted, inv_order, axis=0)

    unserved = np.maximum(load - power.sum(axis=0), 0)

    srmc_dispatched = np.where(power > 0, srmc_all, -np.inf)
    marginal_price = np.maximum(srmc_dispatched.max(axis=0), 0)

    # Track which unit sets the marginal price at each timestep. The
    # price-setter is the dispatched unit with the highest SRMC. Sentinel
    # -1 marks timesteps where the marginal price is zero (nothing
    # dispatched or total unserved).
    price_setter_idx = np.argmax(srmc_dispatched, axis=0).astype(np.int16)
    price_setter_idx[marginal_price <= 0] = -1

    # Phase 2: inertia fix. Imports contribute no synchronous inertia and
    # cannot be used to satisfy H_MIN — the existing algorithm already
    # respects this because it only considers is_sync == True units.
    sync_online = (power > 0) & is_sync[:, np.newaxis]
    wh = (h_values[:, np.newaxis] * capacity_pu[:, np.newaxis]) * sync_online
    tc = np.maximum((capacity_pu[:, np.newaxis] * sync_online).sum(axis=0), 1e-10)
    h_system = wh.sum(axis=0) / tc
    h_system[tc < 1e-9] = 0

    curtailment = np.zeros(n_t)
    violation_mask = h_system < H_MIN_SECONDS

    if violation_mask.any():
        sync_indices = np.where(is_sync)[0]
        if len(sync_indices) > 0:
            avg_srmc_sync = srmc_all[sync_indices].mean(axis=1)
            sync_sorted = sync_indices[np.argsort(avg_srmc_sync)]

            viol_indices = np.where(violation_mask)[0]
            for t in viol_indices:
                for si in sync_sorted:
                    if power[si, t] > 0 or avail_all[si, t] <= 0:
                        continue
                    power[si, t] = min(min_stable[si], avail_all[si, t])
                    if power[si, t] <= 0:
                        power[si, t] = avail_all[si, t] * 0.4
                    sm = (power[:, t] > 0) & is_sync
                    if sm.any():
                        h_system[t] = ((h_values[sm] * capacity_pu[sm]).sum()
                                       / capacity_pu[sm].sum())
                    if h_system[t] >= H_MIN_SECONDS:
                        break

                excess = power[:, t].sum() - load[t]
                if excess > 0:
                    nonsync = np.where(~is_sync & (power[:, t] > 0))[0]
                    if len(nonsync) > 0:
                        for ni in nonsync[np.argsort(-srmc_all[nonsync, t])]:
                            cut = min(power[ni, t], excess)
                            power[ni, t] -= cut
                            curtailment[t] += cut
                            excess -= cut
                            if excess <= 0:
                                break

                dm = power[:, t] > 0
                if dm.any():
                    marginal_price[t] = srmc_all[dm, t].max()
                    dm_indices = np.where(dm)[0]
                    price_setter_idx[t] = dm_indices[
                        np.argmax(srmc_all[dm_indices, t])]
                else:
                    price_setter_idx[t] = -1

    # ── Phase 3: export adjustment ──────────────────────────────────────
    #
    # For each interconnection, at each timestep where the current
    # marginal price is below the link's export floor and export NTC is
    # available, dispatch additional headroom from units with
    # SRMC <= export_floor until the floor is reached or NTC is saturated.
    # Interconnections are processed in order of decreasing export floor
    # so that the most lucrative destination is served first. Domestic
    # units are called upon greedily in merit order; virtual imports do
    # NOT serve export (they are a separate commercial flow on the same
    # physical link's opposite direction — modelling them as an export
    # source would be economically incoherent). Imports already
    # dispatched in Phase 1 are unaffected.
    export_power = np.zeros((n_ic, n_t))

    if n_ic > 0:
        # Domestic-only slice for headroom accounting
        domestic_mask = np.zeros(n_units, dtype=bool)
        domestic_mask[:n_domestic] = True

        # Build export metadata arrays
        export_floor = np.array(
            [r.export_floor_path for r in interconnection_realizations])   # (n_ic, T)
        ntc_export_pu = np.array(
            [r.ntc_export_pu_path for r in interconnection_realizations])  # (n_ic, T)

        # For each timestep, find links with profitable export opportunity
        # and available NTC. Vectorized candidate mask:
        candidate_mask = (ntc_export_pu > 1e-12) & (
            export_floor > marginal_price[np.newaxis, :])

        active_t = np.where(candidate_mask.any(axis=0))[0]

        for t in active_t:
            # Links active at this timestep, sorted by floor descending
            ic_at_t = np.where(candidate_mask[:, t])[0]
            if ic_at_t.size == 0:
                continue
            ic_at_t = ic_at_t[np.argsort(-export_floor[ic_at_t, t])]

            # Current per-unit headroom, by unit, domestic only
            headroom = np.where(
                domestic_mask,
                avail_all[:, t] - power[:, t],
                0.0,
            )
            headroom = np.maximum(headroom, 0.0)

            for k in ic_at_t:
                floor_k = float(export_floor[k, t])
                ntc_k = float(ntc_export_pu[k, t])

                # Merit-order over domestic units with SRMC <= floor_k
                eligible = np.where(
                    domestic_mask & (srmc_all[:, t] <= floor_k) & (headroom > 0)
                )[0]
                if eligible.size == 0:
                    continue
                # Call units by increasing SRMC
                eligible = eligible[np.argsort(srmc_all[eligible, t])]

                remaining_demand = ntc_k
                last_srmc_called = marginal_price[t]
                last_ui_called = -1
                for ui in eligible:
                    if remaining_demand <= 1e-12:
                        break
                    take = min(headroom[ui], remaining_demand)
                    if take <= 0:
                        continue
                    power[ui, t] += take
                    headroom[ui] -= take
                    export_power[k, t] += take
                    remaining_demand -= take
                    last_srmc_called = float(srmc_all[ui, t])
                    last_ui_called = int(ui)

                # Marginal price rises to the SRMC of the last unit called
                # (if any was called for this link). When it does, that unit
                # becomes the price-setter for this timestep.
                if export_power[k, t] > 0:
                    if last_srmc_called > marginal_price[t]:
                        marginal_price[t] = last_srmc_called
                        if last_ui_called >= 0:
                            price_setter_idx[t] = last_ui_called

    # Net import = import_dispatched - export_power (per interconnection).
    # Import-dispatched is the dispatched power of each virtual import
    # generator, which lives in rows [n_domestic : n_units] of `power`.
    if n_ic > 0:
        import_power = power[n_domestic:n_units, :]           # shape (n_ic, T)
        net_import_pu = import_power - export_power
        foreign_prices = np.array(
            [r.foreign_price for r in interconnection_realizations])
        ci_g_per_kwh = np.array(
            [r.carbon_intensity_g_per_kwh
             for r in interconnection_realizations])
        # Consumption-based emissions: only when net flow is into IT (import).
        # Dimensional analysis: pos_net[pu] · P_BASE[GW] · 0.25[h] gives
        # energy in pu·GWh; multiplying by CI[g/kWh] yields
        # pu · GWh · g/kWh = pu · 10⁶ kWh · g/kWh = pu · 10⁶ g = pu · tons.
        # So the product below is already in tonnes of CO₂ per quarter-hour —
        # no further scaling needed.
        pos_net = np.maximum(net_import_pu, 0.0)
        emissions_imported_tons = (
            pos_net * P_BASE * 0.25 * ci_g_per_kwh[:, np.newaxis])
    else:
        net_import_pu = np.zeros((0, n_t))
        foreign_prices = np.zeros((0, n_t))
        emissions_imported_tons = np.zeros((0, n_t))

    # ── Phase 4: storage dispatch (sequential, stateful) ─────────────
    #
    # Uses the Phase 1–3 marginal price as the baseline for a rolling-
    # percentile arbitrage strategy. The battery charges (adds load) when
    # the current price is in the cheapest quartile of a trailing window
    # and discharges (displaces the marginal unit) when the price is in
    # the most expensive quartile. SOC evolves sequentially.
    #
    # For each timestep affected by the battery, the generation stack and
    # marginal price are updated to reflect the changed load/generation
    # balance:
    # - Charge: re-dispatches the timestep with load + charge_power.
    # - Discharge: reduces generation from the most expensive dispatched
    #   unit(s), and updates the marginal price to the new marginal unit.
    #
    # BESS synthetic inertia is folded into ``h_system`` at the end.
    storage_units = storage_units or []
    n_storage = len(storage_units)

    if n_storage > 0:
        storage_power_arr = np.zeros((n_storage, n_t))  # + = discharge
        storage_soc_arr = np.zeros((n_storage, n_t))
        baseline_price = marginal_price.copy()

        # Pre-compute per-unit quantities to avoid repeated property calls.
        s_pwr_pu = np.array([s.power_capacity_pu for s in storage_units])
        s_ecap_puh = np.array([s.energy_capacity_pu_h for s in storage_units])
        s_eta_c = np.array([s.eta_charge for s in storage_units])
        s_eta_d = np.array([s.eta_discharge for s in storage_units])
        s_sd_qh = np.array([s.self_discharge_per_qh for s in storage_units])
        s_soc_min = np.array([s.soc_min_frac for s in storage_units])
        s_soc_max = np.array([s.soc_max_frac for s in storage_units])

        soc = np.array([s.initial_soc_frac for s in storage_units])  # mutable

        window = STORAGE_PERCENTILE_WINDOW_QH
        p_charge_pct = STORAGE_CHARGE_PERCENTILE
        p_discharge_pct = STORAGE_DISCHARGE_PERCENTILE

        for t in range(n_t):
            # Rolling-percentile thresholds from the baseline price.
            w_start = max(0, t - window)
            price_window = baseline_price[w_start:t + 1]
            th_charge = np.percentile(price_window, p_charge_pct)
            th_discharge = np.percentile(price_window, p_discharge_pct)

            current_price = marginal_price[t]

            for si in range(n_storage):
                # Apply self-discharge first.
                soc[si] *= (1.0 - s_sd_qh[si])

                if current_price <= th_charge and soc[si] < s_soc_max[si]:
                    # ── Charge ─────────────────────────────────────────
                    # How much can we store?  Limited by power and by the
                    # remaining headroom in the SOC band (AC-side).
                    soc_room_pu_h = (
                        (s_soc_max[si] - soc[si])
                        * s_ecap_puh[si] / s_eta_c[si])
                    max_ac_pu_h = min(s_pwr_pu[si] * 0.25, soc_room_pu_h)
                    charge_ac_pu = max_ac_pu_h / 0.25  # avg power this qh

                    if charge_ac_pu > 1e-12:
                        # Re-dispatch this timestep with augmented load.
                        eff_load_t = load[t] + charge_ac_pu
                        _redispatch_timestep(
                            t, eff_load_t, n_units, srmc_all, avail_all,
                            power, marginal_price, price_setter_idx)
                        current_price = marginal_price[t]

                        soc[si] += (charge_ac_pu * 0.25
                                    * s_eta_c[si] / s_ecap_puh[si])
                        soc[si] = min(soc[si], s_soc_max[si])
                        storage_power_arr[si, t] = -charge_ac_pu

                elif current_price >= th_discharge and soc[si] > s_soc_min[si]:
                    # ── Discharge ──────────────────────────────────────
                    soc_avail_pu_h = (
                        (soc[si] - s_soc_min[si])
                        * s_ecap_puh[si] * s_eta_d[si])
                    max_ac_pu_h = min(s_pwr_pu[si] * 0.25, soc_avail_pu_h)
                    discharge_ac_pu = max_ac_pu_h / 0.25

                    if discharge_ac_pu > 1e-12:
                        # Reduce load for the merit-order re-dispatch.
                        eff_load_t = max(load[t] - discharge_ac_pu, 0.0)
                        _redispatch_timestep(
                            t, eff_load_t, n_units, srmc_all, avail_all,
                            power, marginal_price, price_setter_idx)
                        current_price = marginal_price[t]

                        soc[si] -= (discharge_ac_pu * 0.25
                                    / s_eta_d[si] / s_ecap_puh[si])
                        soc[si] = max(soc[si], s_soc_min[si])
                        storage_power_arr[si, t] = discharge_ac_pu

                storage_soc_arr[si, t] = soc[si]

        # Fold BESS synthetic inertia into h_system for timesteps where
        # the SOC is inside the operational band.
        for t in range(n_t):
            wh_extra = 0.0
            w_extra = 0.0
            for si in range(n_storage):
                wh, w = storage_units[si].inertia_contribution(
                    storage_soc_arr[si, t])
                wh_extra += wh
                w_extra += w
            if w_extra > 0:
                sync_online_t = (power[:, t] > 0) & is_sync
                wh_sync = (h_values[sync_online_t]
                           * capacity_pu[sync_online_t]).sum()
                w_sync = capacity_pu[sync_online_t].sum()
                total_wh = wh_sync + wh_extra
                total_w = w_sync + w_extra
                if total_w > 1e-12:
                    h_system[t] = total_wh / total_w
    else:
        storage_power_arr = np.zeros((0, n_t))
        storage_soc_arr = np.zeros((0, n_t))

    # Territorial CO₂ emissions: kg per quarter-hour per unit.
    # power_pu * P_BASE(GW) * 0.25(h) = energy in GWh
    # GWh * 1000 = MWh_e; MWh_e / efficiency = MWh_th; MWh_th * emission_factor(tCO₂/MWh_th) = tCO₂
    # tCO₂ * 1000 = kgCO₂
    emission_factors = np.array([u.emission_factor for u in units])
    efficiencies = np.array([u.efficiency for u in units])
    safe_eff = np.where(efficiencies > 0, efficiencies, 1.0)
    emissions = (power * P_BASE * 0.25 * 1000
                 * emission_factors[:, np.newaxis] / safe_eff[:, np.newaxis])

    gen_names = [u.name for u in units]
    gen_types = [getattr(u, 'gen_type', 'unknown') for u in units]

    # ── Interconnection metrics (congestion-rent method) ──────────────
    # Computed only when at least one interconnection is active. See
    # :class:`InterconnectionMetrics` for the methodology.
    if n_ic > 0:
        # Load-weighted domestic marginal CI in gCO₂/kWh_e.
        # ef[tCO₂/MWh_th] / η[MWh_e/MWh_th] = tCO₂/MWh_e = kg/kWh_e,
        # so multiplying by 1000 yields g/kWh_e.
        dom_power = power[:n_domestic, :]                # (n_domestic, T)
        ef_g_per_kwh_dom = (
            emission_factors[:n_domestic] / safe_eff[:n_domestic] * 1000.0)
        total_dom_power = dom_power.sum(axis=0)          # (T,)
        weighted_g_per_kwh = (
            dom_power * ef_g_per_kwh_dom[:, np.newaxis]).sum(axis=0)  # (T,)
        marg_ci = np.where(
            total_dom_power > 1e-12,
            weighted_g_per_kwh / np.maximum(total_dom_power, 1e-12),
            0.0,
        )

        # Economic benefit per qh per link in EUR.
        # pu × P_BASE(GW) × 0.25(h) × 1000(MW/GW) × Δprice(€/MWh) = €
        srmc_import_path = np.array(
            [r.import_srmc_path for r in interconnection_realizations])
        floor_export_path = np.array(
            [r.export_floor_path for r in interconnection_realizations])

        econ_import_eur = (import_power * P_BASE * 0.25 * 1000
                           * (marginal_price - srmc_import_path))
        econ_export_eur = (export_power * P_BASE * 0.25 * 1000
                           * (floor_export_path - marginal_price))
        # Clamp tiny negatives from floating-point noise on borderline qh.
        econ_import_eur = np.maximum(econ_import_eur, 0.0)
        econ_export_eur = np.maximum(econ_export_eur, 0.0)
        economic_benefit_eur_qh = econ_import_eur + econ_export_eur

        # CO₂ benefit per qh per link in tonnes.
        # pu × P_BASE(GW) × 0.25(h) × CI(g/kWh) → pu·tons (see
        # emissions_imported_tons docstring for the full derivation).
        # Import: benefit is avoided domestic emissions - incurred foreign.
        # Export: benefit is avoided foreign emissions - incurred domestic.
        co2_import_tons = (import_power * P_BASE * 0.25
                           * (marg_ci - ci_g_per_kwh[:, np.newaxis]))
        co2_export_tons = (export_power * P_BASE * 0.25
                           * (ci_g_per_kwh[:, np.newaxis] - marg_ci))
        co2_benefit_tons_qh = co2_import_tons + co2_export_tons

        # Hours and energy aggregates
        dt_h = 0.25
        eps = 1e-9
        import_hours = (import_power > eps).sum(axis=1) * dt_h
        export_hours = (export_power > eps).sum(axis=1) * dt_h
        import_energy_mwh = import_power.sum(axis=1) * P_BASE * 1000 * dt_h
        export_energy_mwh = export_power.sum(axis=1) * P_BASE * 1000 * dt_h

        ic_metrics = InterconnectionMetrics(
            import_hours=import_hours,
            export_hours=export_hours,
            import_energy_mwh=import_energy_mwh,
            export_energy_mwh=export_energy_mwh,
            economic_benefit_eur_qh=economic_benefit_eur_qh,
            co2_benefit_tons_qh=co2_benefit_tons_qh,
            total_economic_benefit_eur=economic_benefit_eur_qh.sum(axis=1),
            total_co2_benefit_tons=co2_benefit_tons_qh.sum(axis=1),
            domestic_marginal_ci_g_per_kwh=marg_ci,
        )
    else:
        ic_metrics = None

    return DispatchResult(
        power=power,
        marginal_price=marginal_price,
        curtailment=curtailment,
        h_system=h_system,
        unserved=unserved,
        gen_names=gen_names,
        gen_types=gen_types,
        emissions=emissions,
        net_import_pu=net_import_pu,
        interconnection_names=[r.name for r in interconnection_realizations],
        foreign_prices=foreign_prices,
        emissions_imported_tons=emissions_imported_tons,
        ic_metrics=ic_metrics,
        storage_power_pu=storage_power_arr,
        storage_soc=storage_soc_arr,
        storage_names=[s.name for s in storage_units],
        price_setter_idx=price_setter_idx,
    )
