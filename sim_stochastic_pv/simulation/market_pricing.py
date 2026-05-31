"""
Valuation of PV grid flows against a pre-computed wholesale price surface.

This module bridges the standalone electricity-market engine
(:mod:`sim_stochastic_pv.market`) and the economic post-processing of the PV
Monte Carlo. The market engine produces a :class:`~sim_stochastic_pv.market.PriceSurface`
— a bank of wholesale-price trajectories indexed by
``(trajectory, year, month, hour)`` — which is expensive to build and therefore
computed once and cached on a saved market profile. The PV Monte Carlo then
only performs cheap array lookups against that surface.

The central object here is :class:`MarketPriceProvider`. It answers two
questions for one Monte Carlo path:

* **How much is exported PV energy worth?** Under the Italian *ritiro dedicato*
  scheme the grid operator pays the producer ``max(wholesale, PMG)`` for each
  exported kWh, where ``PMG`` ("prezzo minimo garantito") is a guaranteed floor
  indexed to inflation: ``PMG(year) = PMG_base · (1 + inflation)^year``.

* **What would the household pay to buy that same energy?** (optional) A retail
  tariff derived from the wholesale price as
  ``retail = wholesale · (1 + markup) + fixed_components``, used by callers that
  let the simulated market drive the purchase price too.

The provider is deliberately stateless apart from its configuration: the
per-path inflation trajectory and the chosen trajectory index are passed in by
the caller (the Monte Carlo orchestrator), so the same provider instance can be
reused across all paths.

Notes:
    - All grids are EUR/kWh and share the surface's ``(n_years, 12, 24)`` layout
      with **0-based** months (January = index 0), matching the export-by-hour
      array produced by the energy simulator.
    - "Nominal vs real": the wholesale prices already embed their own escalation
      via the market engine's fuel/CO2 drift, while the PMG floor escalates with
      inflation. Both are nominal cash amounts; the Monte Carlo discounts them to
      real terms with the same inflation factors it uses for energy savings, so
      the treatment is internally consistent.
    - Horizon mismatch between the cached surface and the simulation is handled
      by clamping (see :meth:`_aligned_wholesale_grid`), so a surface built for a
      slightly different horizon never crashes a run.
"""

from __future__ import annotations

import numpy as np

from ..market import PriceSurface


class MarketPriceProvider:
    """
    Value PV grid export (and optionally retail purchase) against a wholesale
    price surface, with an inflation-indexed guaranteed minimum price (PMG).

    A :class:`MarketPriceProvider` wraps a pre-computed
    :class:`~sim_stochastic_pv.market.PriceSurface` and exposes vectorised
    methods that turn an hourly export-energy grid into a monthly revenue
    stream. It implements the *ritiro dedicato* valuation: each exported kWh in
    bucket ``(year, month, hour)`` is paid at::

        export_price(year, month, hour) = max(
            wholesale(trajectory, year, month, hour),
            PMG_base · inflation_factor(year),
        )

    where ``inflation_factor(year)`` is the cumulative inflation multiplier of
    that year (year 0 → 1.0), supplied by the caller so the PMG floor escalates
    consistently with the rest of the cash flow.

    Optionally, when a retail markup is configured, the provider can also derive
    a retail tariff grid (``wholesale · (1 + markup) + fixed_components``) used
    by callers that want the simulated market to drive the purchase price.

    The provider is **inert when not attached to a simulation**: building one has
    no side effects, and a Monte Carlo run that is given no provider behaves
    byte-identically to one that predates this feature.

    Attributes:
        price_surface: The cached wholesale :class:`PriceSurface`
            (``(n_trajectories, n_years, 12, 24)`` EUR/kWh).
        pmg_base_eur_per_kwh: Guaranteed minimum price floor at year 0
            (EUR/kWh), before inflation indexing. ``0.0`` disables the floor
            (export is then always paid the raw wholesale price).
        retail_markup_fraction: Multiplicative markup applied to the wholesale
            price to obtain the retail tariff (decimal, e.g. ``0.8`` = +80%).
            ``None`` (the default) means retail derivation is not configured and
            :meth:`retail_price_grid` raises.
        retail_fixed_components_eur_per_kwh: Flat per-kWh component added to the
            marked-up wholesale price (taxes, grid fees, system charges
            aggregated), EUR/kWh. Only consulted when a markup is configured.

    Example:
        ```python
        import numpy as np
        from sim_stochastic_pv.market import PriceSurface
        from sim_stochastic_pv.simulation.market_pricing import MarketPriceProvider

        # A 1-trajectory, 2-year flat 0.05 EUR/kWh wholesale surface.
        grid = np.full((1, 2, 12, 24), 0.05)
        surface = PriceSurface(price_eur_per_kwh=grid, n_trajectories=1, n_years=2)

        provider = MarketPriceProvider(surface, pmg_base_eur_per_kwh=0.07)

        # 1 kWh exported in every (year, month, hour) bucket.
        export = np.ones((2, 12, 24))
        infl = np.array([1.0, 1.025])  # year 0 and year 1 inflation factors
        eur, kwh = provider.value_export_grid(
            export, trajectory_index=0, inflation_factor_by_year=infl
        )
        # Year 0: PMG 0.07 > wholesale 0.05 → 0.07 EUR/kWh paid.
        # Year 1: PMG 0.07*1.025 = 0.07175 > 0.05 → that price paid.
        assert abs(eur[0] - 24 * 0.07) < 1e-9      # January of year 0
        assert kwh[0] == 24.0
        ```

    Notes:
        - Months are **0-based** throughout this class (to match the energy
          simulator's export-by-hour array), whereas
          :meth:`PriceSurface.price_at` uses 1-based months. Internally the
          provider works with the 0-based ``trajectory_grid`` view.
        - All public methods are pure functions of their arguments and the
          immutable configuration; the provider holds no per-path state.
    """

    def __init__(
        self,
        price_surface: PriceSurface,
        *,
        pmg_base_eur_per_kwh: float = 0.0,
        retail_markup_fraction: float | None = None,
        retail_fixed_components_eur_per_kwh: float = 0.0,
    ) -> None:
        """
        Initialise the provider from a cached wholesale surface and PMG config.

        Validates the configuration at this boundary (CLAUDE.md §2.4): the
        surface must be a well-formed :class:`PriceSurface`, the PMG floor must
        be non-negative, and the retail components (when a markup is given) must
        be sensible. Downstream methods then trust their inputs.

        Args:
            price_surface: Pre-computed wholesale :class:`PriceSurface`. Its
                ``price_eur_per_kwh`` must be a 4-D array of shape
                ``(n_trajectories, n_years, 12, 24)`` in EUR/kWh.
            pmg_base_eur_per_kwh: Guaranteed minimum export price at year 0
                (EUR/kWh, ≥ 0). Default ``0.0`` (no floor). Typical Italian
                small-plant *ritiro dedicato* values are ~0.03–0.06 EUR/kWh.
            retail_markup_fraction: Optional multiplicative markup on the
                wholesale price for the retail tariff (decimal ≥ -1). ``None``
                disables retail derivation. Default ``None``.
            retail_fixed_components_eur_per_kwh: Flat per-kWh add-on for the
                retail tariff (EUR/kWh, ≥ 0). Ignored when no markup is set.
                Default ``0.0``.

        Raises:
            TypeError: If ``price_surface`` is not a :class:`PriceSurface`.
            ValueError: If the surface array is not 4-D with a trailing
                ``(12, 24)`` shape, if ``pmg_base_eur_per_kwh`` is negative, if
                ``retail_markup_fraction`` is below ``-1`` (would make the
                tariff negative before fixed components), or if
                ``retail_fixed_components_eur_per_kwh`` is negative.

        Notes:
            - Construction is cheap and side-effect free.
            - The surface is stored by reference (not copied); callers must not
              mutate it after handing it over.
        """
        if not isinstance(price_surface, PriceSurface):
            raise TypeError(
                "price_surface must be a PriceSurface, got "
                f"{type(price_surface).__name__}"
            )
        arr = price_surface.price_eur_per_kwh
        if arr.ndim != 4 or arr.shape[2:] != (12, 24):
            raise ValueError(
                "price_surface.price_eur_per_kwh must have shape "
                f"(n_trajectories, n_years, 12, 24); got {arr.shape}"
            )
        if pmg_base_eur_per_kwh < 0.0:
            raise ValueError(
                f"pmg_base_eur_per_kwh must be >= 0, got {pmg_base_eur_per_kwh}"
            )
        if retail_markup_fraction is not None and retail_markup_fraction < -1.0:
            raise ValueError(
                "retail_markup_fraction must be >= -1 (else the tariff would go "
                f"negative); got {retail_markup_fraction}"
            )
        if retail_fixed_components_eur_per_kwh < 0.0:
            raise ValueError(
                "retail_fixed_components_eur_per_kwh must be >= 0, got "
                f"{retail_fixed_components_eur_per_kwh}"
            )

        self.price_surface = price_surface
        self.pmg_base_eur_per_kwh = float(pmg_base_eur_per_kwh)
        self.retail_markup_fraction = (
            None if retail_markup_fraction is None else float(retail_markup_fraction)
        )
        self.retail_fixed_components_eur_per_kwh = float(
            retail_fixed_components_eur_per_kwh
        )

    def _aligned_wholesale_grid(
        self, trajectory_index: int, n_years: int
    ) -> np.ndarray:
        """
        Return the wholesale grid for one trajectory, aligned to ``n_years``.

        The cached surface may have been built for a slightly different horizon
        than the current simulation. Rather than crash, the grid is clamped:
        if the surface is longer it is truncated; if it is shorter its last
        available year is repeated to fill the requested horizon. This keeps a
        reusable saved profile usable across simulations whose horizon was
        tweaked, at the cost of a flat tail in the (rare) shorter case.

        Args:
            trajectory_index: Trajectory to read in ``[0, n_trajectories)``.
            n_years: Number of years the caller needs.

        Returns:
            np.ndarray: Shape ``(n_years, 12, 24)`` in EUR/kWh.

        Notes:
            - Returns a fresh array (never a view onto the surface) whenever
              clamping happens, so callers may safely operate in place.
        """
        traj = trajectory_index % self.price_surface.n_trajectories
        grid = self.price_surface.trajectory_grid(traj)  # (S, 12, 24)
        surface_years = grid.shape[0]
        if surface_years == n_years:
            return grid
        if surface_years > n_years:
            return grid[:n_years]
        pad = np.repeat(grid[-1:], n_years - surface_years, axis=0)
        return np.concatenate([grid, pad], axis=0)

    def export_price_grid(
        self,
        *,
        trajectory_index: int,
        inflation_factor_by_year: np.ndarray,
    ) -> np.ndarray:
        """
        Build the *ritiro dedicato* export price grid ``max(wholesale, PMG)``.

        For each ``(year, month, hour)`` bucket the export is paid the larger of
        the wholesale price for the chosen trajectory and the inflation-indexed
        PMG floor of that year. The floor is constant across months and hours of
        a given year.

        Args:
            trajectory_index: Wholesale trajectory to use; taken modulo the
                surface's ``n_trajectories``, so any non-negative integer works.
            inflation_factor_by_year: Cumulative inflation factor per year,
                shape ``(n_years,)`` with ``[0] = 1.0`` for year 0. The PMG floor
                of year ``y`` is ``pmg_base_eur_per_kwh · factor[y]``.

        Returns:
            np.ndarray: Shape ``(n_years, 12, 24)`` EUR/kWh export prices.

        Raises:
            ValueError: If ``inflation_factor_by_year`` is not 1-D.

        Example:
            ```python
            import numpy as np
            grid = provider.export_price_grid(
                trajectory_index=0,
                inflation_factor_by_year=np.array([1.0, 1.02, 1.04]),
            )
            assert grid.shape == (3, 12, 24)
            ```

        Notes:
            - With ``pmg_base_eur_per_kwh == 0`` this reduces to the raw
              wholesale grid (the floor never binds).
        """
        infl = np.asarray(inflation_factor_by_year, dtype=float)
        if infl.ndim != 1:
            raise ValueError(
                f"inflation_factor_by_year must be 1-D (n_years,), got shape {infl.shape}"
            )
        n_years = int(infl.size)
        wholesale = self._aligned_wholesale_grid(trajectory_index, n_years)
        pmg_by_year = self.pmg_base_eur_per_kwh * infl  # (n_years,)
        return np.maximum(wholesale, pmg_by_year[:, None, None])

    def value_export_grid(
        self,
        export_kwh_by_year_month_hour: np.ndarray,
        *,
        trajectory_index: int,
        inflation_factor_by_year: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Turn an hourly export-energy grid into a monthly revenue stream.

        Multiplies the export energy of each ``(year, month, hour)`` bucket by
        its *ritiro dedicato* price (:meth:`export_price_grid`) and sums over the
        hour axis to obtain monthly totals. Energy is also aggregated to monthly
        kWh for diagnostics. Both outputs are flattened to a single month axis of
        length ``n_years * 12`` ordered ``[y0m0, y0m1, ..., y0m11, y1m0, ...]`` —
        the same month ordering the Monte Carlo uses for its cash-flow vectors.

        Args:
            export_kwh_by_year_month_hour: Exported energy per bucket, shape
                ``(n_years, 12, 24)`` in kWh, as surfaced by the energy
                simulator (``last_export_kwh_by_year_month_hour``).
            trajectory_index: Wholesale trajectory to use (taken modulo
                ``n_trajectories``).
            inflation_factor_by_year: Cumulative inflation factor per year,
                shape ``(n_years,)``. Must match the export grid's year count.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(monthly_export_eur,
                monthly_export_kwh)``, each of shape ``(n_years * 12,)``.
                ``monthly_export_eur`` is nominal EUR revenue per month;
                ``monthly_export_kwh`` is the exported energy per month.

        Raises:
            ValueError: If ``export_kwh_by_year_month_hour`` is not
                ``(n_years, 12, 24)``, or if its year count disagrees with
                ``inflation_factor_by_year``.

        Example:
            ```python
            import numpy as np
            export = np.zeros((2, 12, 24))
            export[0, 0, 12] = 3.0  # 3 kWh exported in y0 Jan at 12:00
            eur, kwh = provider.value_export_grid(
                export, trajectory_index=0,
                inflation_factor_by_year=np.array([1.0, 1.0]),
            )
            assert kwh[0] == 3.0  # January of year 0
            ```

        Notes:
            - Revenue is **nominal**; the caller discounts it to real terms with
              the same inflation factors used for energy savings.
            - Buckets with zero export contribute zero revenue regardless of
              price, so curtailed/never-exported hours are naturally ignored.
        """
        export = np.asarray(export_kwh_by_year_month_hour, dtype=float)
        if export.ndim != 3 or export.shape[1:] != (12, 24):
            raise ValueError(
                "export_kwh_by_year_month_hour must have shape (n_years, 12, 24); "
                f"got {export.shape}"
            )
        infl = np.asarray(inflation_factor_by_year, dtype=float)
        if infl.ndim != 1 or infl.size != export.shape[0]:
            raise ValueError(
                "inflation_factor_by_year must be 1-D with one entry per export "
                f"year; got shape {infl.shape} for {export.shape[0]} years"
            )

        price_grid = self.export_price_grid(
            trajectory_index=trajectory_index,
            inflation_factor_by_year=infl,
        )
        eur_by_year_month = (export * price_grid).sum(axis=2)  # (n_years, 12)
        kwh_by_year_month = export.sum(axis=2)  # (n_years, 12)
        return eur_by_year_month.reshape(-1), kwh_by_year_month.reshape(-1)

    def retail_price_grid(
        self,
        *,
        trajectory_index: int,
        inflation_factor_by_year: np.ndarray,
    ) -> np.ndarray:
        """
        Build the market-derived retail tariff grid (optional feature).

        Returns ``wholesale · (1 + markup) + fixed_components`` for each
        ``(year, month, hour)`` bucket, the price a household would pay to buy
        energy when the simulated market is allowed to drive the purchase price.
        The fixed components are a flat per-kWh add-on (taxes, grid fees and
        system charges aggregated) and are **not** inflation-indexed here.

        Args:
            trajectory_index: Wholesale trajectory to use (taken modulo
                ``n_trajectories``).
            inflation_factor_by_year: Cumulative inflation factor per year,
                shape ``(n_years,)``. Used only to size the horizon (the retail
                formula itself is not inflation-indexed); the wholesale escalation
                already lives inside the surface.

        Returns:
            np.ndarray: Shape ``(n_years, 12, 24)`` EUR/kWh retail prices.

        Raises:
            ValueError: If retail derivation was not configured
                (``retail_markup_fraction is None``) or
                ``inflation_factor_by_year`` is not 1-D.

        Notes:
            - This is the building block for the "market drives the purchase
              price too" option; wiring it into the savings computation is the
              caller's responsibility.
        """
        if self.retail_markup_fraction is None:
            raise ValueError(
                "retail pricing is not configured: pass retail_markup_fraction "
                "to MarketPriceProvider to enable retail_price_grid()"
            )
        infl = np.asarray(inflation_factor_by_year, dtype=float)
        if infl.ndim != 1:
            raise ValueError(
                f"inflation_factor_by_year must be 1-D (n_years,), got shape {infl.shape}"
            )
        n_years = int(infl.size)
        wholesale = self._aligned_wholesale_grid(trajectory_index, n_years)
        return (
            wholesale * (1.0 + self.retail_markup_fraction)
            + self.retail_fixed_components_eur_per_kwh
        )

    # ----------------------------------------------------------- serialization

    def to_config_dict(
        self,
        *,
        build_config: dict | None = None,
        price_decimals: int = 6,
    ) -> dict:
        """
        Serialise the provider to a JSON-friendly configuration dict.

        Captures everything needed to reconstruct an identical provider via
        :meth:`from_config_dict`: the cached wholesale surface (flattened to a
        plain list of floats plus its shape) and the *ritiro dedicato* / retail
        parameters. A ``version`` field is stamped so future schema changes can
        be detected, and an opaque ``build_config`` block can carry provenance
        (mix name, gas scenario, seed, …) for the editing UI — it is ignored on
        reload.

        The surface prices are rounded to ``price_decimals`` decimal places to
        keep the JSON tight; at 6 decimals the rounding error on an EUR/kWh
        price is ≤ 1e-6, far below any economically meaningful resolution.

        Args:
            build_config: Optional provenance metadata to embed verbatim under
                the ``"build_config"`` key. Not consumed on reload. Default
                ``None`` (stored as an empty dict).
            price_decimals: Decimal places to round the surface prices to before
                serialisation (≥ 0). Default 6.

        Returns:
            dict: A JSON-serialisable mapping with keys ``version``,
                ``pmg_base_eur_per_kwh``, ``retail_markup_fraction`` (possibly
                ``None``), ``retail_fixed_components_eur_per_kwh``, ``surface``
                (``{n_trajectories, n_years, shape, price_eur_per_kwh}``) and
                ``build_config``.

        Example:
            ```python
            blob = provider.to_config_dict(build_config={"mix": "italian"})
            clone = MarketPriceProvider.from_config_dict(blob)
            ```

        Notes:
            - Round-trips through :meth:`from_config_dict` up to the rounding
              of the surface prices.
        """
        surface = self.price_surface
        flat = surface.price_eur_per_kwh.reshape(-1)
        return {
            "version": 1,
            "pmg_base_eur_per_kwh": self.pmg_base_eur_per_kwh,
            "retail_markup_fraction": self.retail_markup_fraction,
            "retail_fixed_components_eur_per_kwh": (
                self.retail_fixed_components_eur_per_kwh
            ),
            "surface": {
                "n_trajectories": int(surface.n_trajectories),
                "n_years": int(surface.n_years),
                "shape": [int(s) for s in surface.price_eur_per_kwh.shape],
                "price_eur_per_kwh": [
                    round(float(x), price_decimals) for x in flat
                ],
            },
            "build_config": dict(build_config or {}),
        }

    @classmethod
    def from_config_dict(cls, blob: dict) -> "MarketPriceProvider":
        """
        Reconstruct a provider from a dict produced by :meth:`to_config_dict`.

        Rebuilds the :class:`PriceSurface` from the flattened price list and its
        shape, then re-applies the PMG / retail parameters through the normal
        constructor (so the same boundary validation runs).

        Args:
            blob: Mapping previously produced by :meth:`to_config_dict`. Must
                contain a ``"surface"`` sub-dict with ``shape`` and
                ``price_eur_per_kwh``; the PMG / retail keys are optional and
                default to a no-floor, retail-disabled configuration.

        Returns:
            MarketPriceProvider: A provider equivalent to the serialised one
                (up to the surface rounding applied at serialisation time).

        Raises:
            KeyError: If the ``"surface"`` block or its required keys are
                missing.
            ValueError: Propagated from the constructor / reshape when the
                stored shape and price list are inconsistent.

        Notes:
            - The ``build_config`` block, if present, is ignored here; it exists
              only for the editing UI.
        """
        surface_blob = blob["surface"]
        shape = tuple(int(s) for s in surface_blob["shape"])
        prices = np.asarray(
            surface_blob["price_eur_per_kwh"], dtype=float
        ).reshape(shape)
        surface = PriceSurface(
            price_eur_per_kwh=prices,
            n_trajectories=int(surface_blob["n_trajectories"]),
            n_years=int(surface_blob["n_years"]),
        )
        return cls(
            surface,
            pmg_base_eur_per_kwh=float(blob.get("pmg_base_eur_per_kwh", 0.0)),
            retail_markup_fraction=blob.get("retail_markup_fraction"),
            retail_fixed_components_eur_per_kwh=float(
                blob.get("retail_fixed_components_eur_per_kwh", 0.0)
            ),
        )
