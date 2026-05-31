"""
Cross-border electricity exchanges.

This module ties together price areas (stochastic foreign markets) and
reliability models (stochastic asset availability) into a transmission-link
abstraction used by the dispatch engine.

Two layers:

- :class:`Interconnection` — static definition of a commercial border:
  nominal NTC capacities (import/export, possibly asymmetric), transport
  cost, optional monthly seasonal profile, a reference to a
  :class:`~sim_stochastic_pv.market.price_areas.PriceArea`, and a
  :class:`~sim_stochastic_pv.market.reliability.ReliabilityModel`.
- :class:`InterconnectionRealization` — a per-Monte-Carlo-run realization:
  the resolved foreign price path, the time-varying NTC paths (after
  applying seasonal factors and availability multipliers), and the
  derived economic paths (import SRMC, export floor).

Integration with the dispatch engine happens through
:class:`VirtualImportGenerator`, a Generator-compatible duck-typed object
produced from a realization. It exposes the same interface expected by
:func:`~sim_stochastic_pv.market.dispatch.dispatch_year` (``srmc()``,
``available_power_pu()``, ``h_inertia``, ``is_synchronous``, ``capacity_pu``,
``emission_factor``, ``efficiency``, ``name``, ``gen_type``,
``min_stable_power_pu()``) so that imports clear through the merit order
alongside domestic generators without any special-casing.

Exports are handled as a post-dispatch adjustment (see
:mod:`~sim_stochastic_pv.market.dispatch`) using the realization's export NTC and
export-floor paths directly, since they represent a price-responsive
demand rather than a supply unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from sim_stochastic_pv.market.config import P_PEAK_GW
from sim_stochastic_pv.market.grid import TimeGrid
from sim_stochastic_pv.market.price_areas import PriceAreaCoupling, build_price_areas_from_config
from sim_stochastic_pv.market.reliability import (
    PerfectReliability, ReliabilityModel, build_reliability_model,
)


# ── Static definition ────────────────────────────────────────────────────


@dataclass
class Interconnection:
    """Static definition of a cross-border transmission link.

    One :class:`Interconnection` corresponds to one commercial border with
    a neighbouring price area. The same price area can be shared among
    multiple interconnections (useful in multi-country generic models).

    The nominal NTC is modulated at simulation time by two multiplicative
    factors, both in ``[0, 1]``:

    1. A deterministic monthly seasonal factor (optional).
    2. A stochastic availability path from the reliability model
       (forced outages and partial derates).

    The transport cost is applied symmetrically (paid on import and
    "owed" on export) and accounts for wheeling fees plus average
    transmission losses.

    Attributes:
        name (str): Unique identifier (e.g. ``'IT-FR'``).
        price_area_name (str): Key of the linked :class:`PriceArea`.
        ntc_import_gw (float): Nominal maximum flow into the domestic
            system (GW).
        ntc_export_gw (float): Nominal maximum flow out of the domestic
            system (GW).
        transport_cost_eur_mwh (float): Wheeling fee + loss compensation.
        reliability_model (ReliabilityModel): Stochastic availability
            model. Use :class:`~sim_stochastic_pv.market.reliability.PerfectReliability`
            to disable faults for this link.
        seasonal_ntc_factor_monthly (np.ndarray | None): Optional 12-element
            array of monthly multipliers (index 0 = January). ``None``
            means constant NTC year-round.
    """

    name: str
    price_area_name: str
    ntc_import_gw: float
    ntc_export_gw: float
    transport_cost_eur_mwh: float
    reliability_model: ReliabilityModel = field(default_factory=PerfectReliability)
    seasonal_ntc_factor_monthly: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Validate NTC values and seasonal profile shape."""
        if self.ntc_import_gw < 0:
            raise ValueError(f"ntc_import_gw must be >= 0, got {self.ntc_import_gw}")
        if self.ntc_export_gw < 0:
            raise ValueError(f"ntc_export_gw must be >= 0, got {self.ntc_export_gw}")
        if self.transport_cost_eur_mwh < 0:
            raise ValueError(
                f"transport_cost_eur_mwh must be >= 0, "
                f"got {self.transport_cost_eur_mwh}")
        if self.seasonal_ntc_factor_monthly is not None:
            arr = np.asarray(self.seasonal_ntc_factor_monthly, dtype=float)
            if arr.shape != (12,):
                raise ValueError(
                    "seasonal_ntc_factor_monthly must have length 12, "
                    f"got shape {arr.shape}")
            if np.any(arr < 0):
                raise ValueError(
                    "seasonal_ntc_factor_monthly entries must be >= 0")
            self.seasonal_ntc_factor_monthly = arr

    @property
    def ntc_import_pu(self) -> float:
        """Nominal import NTC in per-unit of system base."""
        return self.ntc_import_gw / P_PEAK_GW

    @property
    def ntc_export_pu(self) -> float:
        """Nominal export NTC in per-unit of system base."""
        return self.ntc_export_gw / P_PEAK_GW

    def realize(self, foreign_price: np.ndarray,
                carbon_intensity_g_per_kwh: float,
                time_grid: TimeGrid,
                rng: np.random.Generator) -> "InterconnectionRealization":
        """Materialize a per-MC-run realization for this interconnection.

        Combines the supplied foreign price path (already realized by
        the :class:`~sim_stochastic_pv.market.price_areas.PriceAreaCoupling`) with a
        freshly-sampled availability path and the deterministic seasonal
        profile. The outputs are the time-varying NTC paths and the
        economic paths (import SRMC, export floor) ready for consumption
        by the dispatch engine.

        Args:
            foreign_price: Foreign day-ahead price path (EUR/MWh), shape
                ``(time_grid.n,)``. Typically one of the values returned
                by :meth:`PriceAreaCoupling.realize`.
            carbon_intensity_g_per_kwh: Average emission intensity of the
                neighbour's generation mix. Copied from the associated
                :class:`~sim_stochastic_pv.market.price_areas.PriceArea`.
            time_grid: Temporal backbone (provides the month array for
                seasonal modulation).
            rng: NumPy random generator for the availability sampling.
                Distinct from the one used by the price coupling, to keep
                price and reliability stochastic sources independent.

        Returns:
            InterconnectionRealization: Frozen container of paths ready
                for dispatch.
        """
        T = time_grid.n

        # Availability multiplier from the reliability model
        availability = self.reliability_model.sample_availability_path(
            time_grid, rng)

        # Seasonal multiplier — expand monthly factors to quarter-hour resolution
        if self.seasonal_ntc_factor_monthly is not None:
            seasonal = self.seasonal_ntc_factor_monthly[time_grid.month - 1]
        else:
            seasonal = np.ones(T)

        # Effective NTC paths in per-unit
        ntc_import_pu_path = self.ntc_import_pu * seasonal * availability
        ntc_export_pu_path = self.ntc_export_pu * seasonal * availability

        return InterconnectionRealization(
            name=self.name,
            price_area_name=self.price_area_name,
            foreign_price=np.asarray(foreign_price, dtype=float),
            transport_cost=float(self.transport_cost_eur_mwh),
            carbon_intensity_g_per_kwh=float(carbon_intensity_g_per_kwh),
            ntc_import_pu_path=ntc_import_pu_path,
            ntc_export_pu_path=ntc_export_pu_path,
            availability=availability,
            seasonal=seasonal,
        )


# ── Per-run realization ──────────────────────────────────────────────────


@dataclass
class InterconnectionRealization:
    """Frozen per-MC-run realization of an interconnection.

    Contains all time-dependent paths needed by the dispatch engine.
    Import and export share the same foreign price path (they represent
    the same neighbouring market at the same instant) — it would be a
    bug to generate two independent stochastic paths for the two
    directions.

    Attributes:
        name (str): Link identifier, copied from the parent
            :class:`Interconnection`.
        price_area_name (str): Linked price area, for lookup and display.
        foreign_price (np.ndarray): Foreign day-ahead price path (EUR/MWh),
            shape ``(T,)``.
        transport_cost (float): Wheeling + loss cost (EUR/MWh).
        carbon_intensity_g_per_kwh (float): Emission intensity of the
            neighbour (gCO₂/kWh), for consumption-based accounting.
        ntc_import_pu_path (np.ndarray): Effective import NTC (per-unit),
            shape ``(T,)``.
        ntc_export_pu_path (np.ndarray): Effective export NTC (per-unit),
            shape ``(T,)``.
        availability (np.ndarray): Raw reliability multiplier (diagnostic),
            shape ``(T,)``.
        seasonal (np.ndarray): Raw seasonal multiplier (diagnostic),
            shape ``(T,)``.
    """

    name: str
    price_area_name: str
    foreign_price: np.ndarray
    transport_cost: float
    carbon_intensity_g_per_kwh: float
    ntc_import_pu_path: np.ndarray
    ntc_export_pu_path: np.ndarray
    availability: np.ndarray
    seasonal: np.ndarray

    @property
    def import_srmc_path(self) -> np.ndarray:
        """SRMC of the import virtual generator: ``foreign_price + τ``.

        This is what the link "costs" per MWh of energy imported into
        the domestic system. It enters the merit order directly.
        """
        return self.foreign_price + self.transport_cost

    @property
    def export_floor_path(self) -> np.ndarray:
        """Price floor below which export is profitable: ``foreign_price - τ``.

        If the domestic marginal price is below this value and export NTC
        is available, it is economically rational to send energy abroad.
        Used by the export-adjustment pass in the dispatch engine.
        """
        return self.foreign_price - self.transport_cost

    def as_virtual_import_generator(self) -> "VirtualImportGenerator":
        """Adapter: expose this realization as a Generator-compatible unit.

        The returned object is duck-typed to :class:`Generator` and can be
        added to the list consumed by
        :func:`~sim_stochastic_pv.market.dispatch.dispatch_year`. This is the mechanism
        that injects imports into the merit order.

        Returns:
            VirtualImportGenerator: Lightweight adapter over ``self``.
        """
        return VirtualImportGenerator(realization=self)


# ── Generator adapter for the dispatch engine ────────────────────────────


@dataclass
class VirtualImportGenerator:
    """Generator-interface adapter over an :class:`InterconnectionRealization`.

    This class exposes exactly the attributes and methods that
    :func:`~sim_stochastic_pv.market.dispatch.dispatch_year` reads from a
    :class:`~sim_stochastic_pv.market.generators.Generator`, so that imports can be
    mixed with domestic units in the merit-order stack without any
    special-casing in the dispatch loop.

    Semantic mapping of generator parameters:

    * Inertia ``h_inertia = 0``, ``is_synchronous = False`` — imports do
      not contribute rotational inertia to the domestic grid. HVDC links
      are decoupled by nature; for AC links this is a mild simplification.
    * ``emission_factor = 0`` and ``efficiency = 1`` — territorial
      emissions accounting (IPCC convention); consumption-based
      emissions are tracked separately in the dispatch result.
    * ``min_stable = 0`` — NTC can modulate freely from 0 to the current
      availability-limited maximum.
    * ``capacity_pu`` — nominal import NTC; the *available* power at each
      timestep reflects seasonal + availability derates via
      :meth:`available_power_pu`.

    Attributes:
        realization (InterconnectionRealization): Underlying realization.
    """

    realization: InterconnectionRealization

    # ── Attributes expected by dispatch_year ──

    @property
    def name(self) -> str:
        """Dispatch row label, prefixed to disambiguate from domestic units."""
        return f"import_{self.realization.name}"

    @property
    def gen_type(self) -> str:
        """Fixed type tag used by downstream aggregation code."""
        return 'import'

    @property
    def h_inertia(self) -> float:
        """Inertia constant. Always zero for imports (see class docstring)."""
        return 0.0

    @property
    def is_synchronous(self) -> bool:
        """Whether the unit provides rotational inertia. Always ``False``."""
        return False

    @property
    def capacity_pu(self) -> float:
        """Nominal import capacity in per-unit (before seasonal/availability)."""
        return float(self.realization.ntc_import_pu_path.max()) \
            if self.realization.ntc_import_pu_path.size > 0 else 0.0

    @property
    def emission_factor(self) -> float:
        """Territorial emission factor. Always zero for imports."""
        return 0.0

    @property
    def efficiency(self) -> float:
        """Fictitious thermal efficiency (unused; imports have no fuel)."""
        return 1.0

    # ── Methods expected by dispatch_year ──

    def srmc(self) -> np.ndarray:
        """Time-varying short-run marginal cost of import (EUR/MWh)."""
        return self.realization.import_srmc_path

    def available_power_pu(self) -> np.ndarray:
        """Effective import NTC path (per-unit) for this run."""
        return self.realization.ntc_import_pu_path

    def min_stable_power_pu(self) -> float:
        """Minimum stable generation. Always zero for imports."""
        return 0.0


# ── Factory ──────────────────────────────────────────────────────────────


def build_interconnections_from_config(
    interconnections_cfg: dict[str, dict],
    enable_faults: bool = True,
) -> list[Interconnection]:
    """Build a list of :class:`Interconnection` from the config dict format.

    Resolves the ``reliability`` sub-dict through
    :func:`~sim_stochastic_pv.market.reliability.build_reliability_model`. When
    ``enable_faults`` is ``False``, every interconnection is forced to
    :class:`~sim_stochastic_pv.market.reliability.PerfectReliability` regardless of its
    config entry — this implements the
    :data:`~sim_stochastic_pv.market.config.ENABLE_NTC_FAULTS` master switch.

    Args:
        interconnections_cfg: Mapping ``name -> {price_area, ntc_import_gw,
            ntc_export_gw, transport_cost_eur_mwh,
            seasonal_ntc_factor_monthly, reliability}``.
        enable_faults: If ``False``, use :class:`PerfectReliability` for
            all links.

    Returns:
        list[Interconnection]: One instance per entry, in insertion order.
    """
    out = []
    for name, cfg in interconnections_cfg.items():
        if enable_faults:
            rel_model = build_reliability_model(dict(cfg['reliability']))
        else:
            rel_model = PerfectReliability()

        seasonal = cfg.get('seasonal_ntc_factor_monthly')
        if seasonal is not None:
            seasonal = np.asarray(seasonal, dtype=float)

        out.append(Interconnection(
            name=name,
            price_area_name=cfg['price_area'],
            ntc_import_gw=cfg['ntc_import_gw'],
            ntc_export_gw=cfg['ntc_export_gw'],
            transport_cost_eur_mwh=cfg['transport_cost_eur_mwh'],
            reliability_model=rel_model,
            seasonal_ntc_factor_monthly=seasonal,
        ))
    return out


def realize_interconnections(
    interconnections: Iterable[Interconnection],
    coupling: PriceAreaCoupling,
    time_grid: TimeGrid,
    rng_prices: np.random.Generator,
    rng_faults: np.random.Generator,
) -> list[InterconnectionRealization]:
    """Produce per-run realizations for a full set of interconnections.

    Handles the two-stage stochastic resolution in the correct order:

    1. Generate jointly-correlated price paths for all referenced price
       areas in a single call to :meth:`PriceAreaCoupling.realize`.
       Multiple interconnections referencing the same area share the
       same realized path (correct physical semantics).
    2. For each link, sample an independent availability path from its
       reliability model.

    Two separate random generators are used to keep the price stochastic
    source independent of the reliability stochastic source. This matters
    for sensitivity analyses that want to isolate the contribution of
    each noise source to the final price distribution.

    Args:
        interconnections: Iterable of :class:`Interconnection` objects.
        coupling: :class:`PriceAreaCoupling` covering all referenced price
            areas. Its :meth:`realize` is called once per MC run.
        time_grid: Temporal backbone.
        rng_prices: Random generator for the price coupling.
        rng_faults: Random generator for reliability sampling.

    Returns:
        list[InterconnectionRealization]: One realization per input link,
            preserving input order.

    Raises:
        KeyError: If an interconnection references a price area not
            present in the supplied coupling.
    """
    price_paths = coupling.realize(time_grid, rng_prices)
    area_by_name = {a.name: a for a in coupling.areas}

    realizations = []
    for link in interconnections:
        if link.price_area_name not in price_paths:
            raise KeyError(
                f"Interconnection '{link.name}' references unknown "
                f"price area '{link.price_area_name}'. "
                f"Known: {sorted(price_paths.keys())}")
        area = area_by_name[link.price_area_name]
        realizations.append(link.realize(
            foreign_price=price_paths[link.price_area_name],
            carbon_intensity_g_per_kwh=area.carbon_intensity_g_per_kwh,
            time_grid=time_grid,
            rng=rng_faults,
        ))
    return realizations


def build_coupling_for_interconnections(
    interconnections: Iterable[Interconnection],
    price_areas_cfg: dict[str, dict],
    correlations_cfg: dict[tuple[str, str], float] | None = None,
    correlated: bool = True,
) -> PriceAreaCoupling:
    """Build a :class:`PriceAreaCoupling` covering exactly the areas referenced.

    Convenience wrapper that inspects the interconnection list, collects
    the set of referenced price-area names, and delegates to
    :func:`~sim_stochastic_pv.market.price_areas.build_price_areas_from_config` with the
    appropriate subset. Avoids simulating unused areas.

    Args:
        interconnections: Iterable of :class:`Interconnection` objects.
        price_areas_cfg: See :data:`~sim_stochastic_pv.market.config.PRICE_AREAS`.
        correlations_cfg: See :data:`~sim_stochastic_pv.market.config.PRICE_AREA_CORRELATIONS`.
        correlated: Forwarded to the coupling.

    Returns:
        PriceAreaCoupling: Configured coupling.
    """
    needed = sorted({link.price_area_name for link in interconnections})
    return build_price_areas_from_config(
        price_areas_cfg=price_areas_cfg,
        correlations_cfg=correlations_cfg,
        correlated=correlated,
        subset=needed,
    )
