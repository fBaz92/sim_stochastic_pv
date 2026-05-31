"""
Foreign electricity price areas and correlated stochastic coupling.

This module provides the abstraction layer between neighbouring electricity
markets (modelled as exogenous stochastic price processes) and the
domestic system. It is deliberately independent of any specific topology
(Italian borders or otherwise) so that the same machinery can model any
price-taker configuration in a generic API-style workflow.

Key concepts:

- :class:`PriceArea` — a single external market with its own O-U price
  process and carbon intensity. Knows nothing about interconnections.
- :class:`PriceAreaCoupling` — a set of :class:`PriceArea` instances plus a
  pairwise correlation matrix. Generates jointly-correlated price paths in
  a single call using Cholesky decomposition of the correlation matrix
  applied to the O-U shocks.

Correlation lives in the *shocks* (the Brownian increments), not the levels.
Different μ/σ/θ between areas are preserved; only the random driving noise
is coupled. This is the standard approach for correlated O-U processes in
multi-commodity / multi-market simulations.

The master switch :data:`~sim_stochastic_pv.market.config.PRICE_AREAS_CORRELATED` allows
falling back to fully-independent simulations without changing calling code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from sim_stochastic_pv.market.generators import FuelPriceModel
from sim_stochastic_pv.market.grid import TimeGrid


# ── Price area definition ────────────────────────────────────────────────


class _PricePathGenerator(Protocol):
    """Structural interface for anything that can produce a price path.

    Matches :class:`~sim_stochastic_pv.market.generators.FuelPriceModel` and
    :class:`~sim_stochastic_pv.market.generators.ConstantFuelPrice`. Only the two methods
    used by the coupling are required.
    """

    def generate_path(self, n_steps: int,
                      rng: np.random.Generator) -> np.ndarray: ...
    def generate_path_from_shocks(self, shocks: np.ndarray) -> np.ndarray: ...


@dataclass
class PriceArea:
    """An external electricity market modelled as an exogenous price area.

    A :class:`PriceArea` represents one price zone (e.g. ``'FR'``) with its
    own stochastic price process and carbon intensity. It does not know
    about the domestic system or any particular interconnection — multiple
    :class:`~sim_stochastic_pv.market.interconnections.Interconnection` instances can
    reference the same price area (useful in multi-country models).

    Attributes:
        name (str): Unique identifier for this area (used as key when
            looking up paths from the coupling).
        price_model: Object implementing the price-path protocol
            (typically :class:`~sim_stochastic_pv.market.generators.FuelPriceModel`).
        carbon_intensity_g_per_kwh (float): Average emission intensity
            of the area's generation mix, used for consumption-based
            accounting of imports. Does not affect the domestic dispatch.
    """

    name: str
    price_model: _PricePathGenerator
    carbon_intensity_g_per_kwh: float = 0.0


# ── Coupling of multiple price areas ─────────────────────────────────────


@dataclass
class PriceAreaCoupling:
    """Joint stochastic simulator for a set of correlated price areas.

    Generates one price path per area in a single call. When ``correlated``
    is ``True``, shocks are coupled via Cholesky decomposition of the
    correlation matrix assembled from ``correlations`` (symmetric, diagonal
    forced to 1). When ``False``, each area's shocks are iid and independent.

    The fallback to independent shocks uses exactly the same numerical
    integrator as the correlated case, so disabling correlation is a pure
    correlation-structure change — marginal distributions are unaffected.

    Attributes:
        areas (list[PriceArea]): Ordered list of price areas. Order defines
            the row/column ordering of the assembled correlation matrix.
        correlations (dict[tuple[str, str], float]): Sparse specification of
            pairwise correlations. Keys are unordered name pairs; order in
            the tuple is irrelevant (symmetry enforced). Missing pairs
            default to zero correlation.
        correlated (bool): If ``False``, all off-diagonal entries are
            forced to zero regardless of ``correlations``.
        jitter (float): Diagonal regularization added before Cholesky to
            mitigate numerical rank deficiency. Defaults to 1e-10.
    """

    areas: list[PriceArea]
    correlations: dict[tuple[str, str], float] = field(default_factory=dict)
    correlated: bool = True
    jitter: float = 1e-10

    def __post_init__(self) -> None:
        """Validate area names are unique."""
        names = [a.name for a in self.areas]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate area names: {names}")

    # ── public API ──

    def build_correlation_matrix(self) -> np.ndarray:
        """Assemble the K×K correlation matrix from the sparse dict.

        Returns:
            np.ndarray: Symmetric matrix with unit diagonal, shape
                ``(K, K)`` where ``K = len(self.areas)``. If
                :attr:`correlated` is ``False``, returns the identity.

        Raises:
            ValueError: If a correlation value is outside ``[-1, 1]``, or
                if the resulting matrix is not positive semidefinite.
            KeyError: If a correlation key references an unknown area name.
        """
        K = len(self.areas)
        idx = {a.name: i for i, a in enumerate(self.areas)}

        if not self.correlated:
            return np.eye(K)

        rho = np.eye(K)
        for (n1, n2), r in self.correlations.items():
            if not -1.0 <= r <= 1.0:
                raise ValueError(
                    f"Correlation {n1}-{n2} = {r} outside [-1, 1]")
            if n1 not in idx:
                raise KeyError(f"Unknown price area in correlation: '{n1}'")
            if n2 not in idx:
                raise KeyError(f"Unknown price area in correlation: '{n2}'")
            i, j = idx[n1], idx[n2]
            rho[i, j] = r
            rho[j, i] = r

        # Validate positive semidefiniteness. Allow a small negative
        # eigenvalue tolerance for pure numerical noise, but reject any
        # user-supplied matrix that is materially indefinite.
        w = np.linalg.eigvalsh(rho)
        if w.min() < -1e-8:
            raise ValueError(
                f"Correlation matrix is not positive semidefinite "
                f"(min eigenvalue = {w.min():.3e}). "
                f"Revise the entries in `correlations`.")

        return rho

    def realize(self, time_grid: TimeGrid,
                rng: np.random.Generator) -> dict[str, np.ndarray]:
        """Generate one jointly-correlated price path per area.

        Produces correlated Gaussian shocks via Cholesky rotation, then
        feeds each row into the corresponding area's price model using the
        ``generate_path_from_shocks`` entry point. The result is one full
        year (``time_grid.n`` quarter-hours) of prices per area.

        Args:
            time_grid: Temporal backbone for the simulated year.
            rng: NumPy random generator. A single call consumes
                ``K * time_grid.n`` standard normal draws.

        Returns:
            dict[str, np.ndarray]: Mapping from area name to price path
                of shape ``(time_grid.n,)`` in EUR/MWh.
        """
        K = len(self.areas)
        T = time_grid.n

        rho = self.build_correlation_matrix()
        # Regularize then Cholesky. For the identity the regularization is
        # harmless and the factor equals (1+jitter) * I.
        L = np.linalg.cholesky(rho + self.jitter * np.eye(K))

        eta = rng.standard_normal((K, T))            # iid shocks
        Z = L @ eta                                   # correlated shocks

        return {
            area.name: area.price_model.generate_path_from_shocks(Z[i])
            for i, area in enumerate(self.areas)
        }


# ── Factory from config dicts ────────────────────────────────────────────


def build_price_areas_from_config(
    price_areas_cfg: dict[str, dict],
    correlations_cfg: dict[tuple[str, str], float] | None = None,
    correlated: bool = True,
    subset: list[str] | None = None,
) -> PriceAreaCoupling:
    """Build a :class:`PriceAreaCoupling` from the plain-dict config format.

    This is the glue that consumes :data:`~sim_stochastic_pv.market.config.PRICE_AREAS`
    and :data:`~sim_stochastic_pv.market.config.PRICE_AREA_CORRELATIONS` and returns a
    ready-to-use coupling. Applications that want to restrict the set of
    areas (e.g. only the subset referenced by enabled interconnections)
    can pass ``subset``; the returned coupling then contains only those
    areas and only the correlations among them.

    Args:
        price_areas_cfg: Mapping ``name -> {mu, sigma, theta,
            carbon_intensity_g_per_kwh}``.
        correlations_cfg: Pairwise correlations. Entries referencing areas
            not in ``subset`` are silently dropped. Defaults to empty.
        correlated: Forwarded to :class:`PriceAreaCoupling`.
        subset: Optional whitelist of area names. ``None`` means use all.

    Returns:
        PriceAreaCoupling: Ready-to-use coupling instance.

    Raises:
        KeyError: If ``subset`` contains a name missing from
            ``price_areas_cfg``.
    """
    if subset is None:
        selected = list(price_areas_cfg.keys())
    else:
        missing = [n for n in subset if n not in price_areas_cfg]
        if missing:
            raise KeyError(f"Unknown price areas in subset: {missing}")
        selected = list(subset)

    areas = []
    for name in selected:
        params = price_areas_cfg[name]
        price_model = FuelPriceModel(
            mu=params['mu'], sigma=params['sigma'], theta=params['theta'])
        areas.append(PriceArea(
            name=name,
            price_model=price_model,
            carbon_intensity_g_per_kwh=params.get(
                'carbon_intensity_g_per_kwh', 0.0),
        ))

    # Filter correlations to the selected subset
    selected_set = set(selected)
    correlations = {
        (a, b): r
        for (a, b), r in (correlations_cfg or {}).items()
        if a in selected_set and b in selected_set
    }

    return PriceAreaCoupling(
        areas=areas,
        correlations=correlations,
        correlated=correlated,
    )
