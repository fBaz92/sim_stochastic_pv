"""
Single-diode photovoltaic module parameter estimation utilities.

Provides :class:`PVModelSingleDiode`, a lightweight helper that can
reconstruct the five unknown parameters of the classical single-diode model
starting from datasheet information only. The model focuses on electrical
calibration and is intentionally decoupled from the rest of the simulation
stack so that it can be integrated later with richer data models or storage.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, elementary_charge
from scipy.optimize import least_squares


class PVModelSingleDiode:
    """
    Single-diode photovoltaic module model that fits datasheet parameters.

    The single-diode equation describes the IV curve of a PV module with
    five unknowns (photocurrent, saturation current, diode ideality factor,
    series resistance, and shunt resistance). The class estimates those
    unknowns so that the equation passes exactly through the three points
    usually published on datasheets: short-circuit (Isc), open-circuit (Voc),
    and maximum power point (Imp, Vmp).

    Attributes:
        Isc: Short-circuit current in amperes.
        Voc: Open-circuit voltage in volts.
        Imp: Current at maximum power point in amperes.
        Vmp: Voltage at maximum power point in volts.
        Ns: Number of cells in series within the module.
        Tcell: Cell temperature in degrees Celsius (STC defaults to 25°C).
        Vt: Thermal voltage for a single cell at the given temperature.
        Vt_mod: Thermal voltage scaled by the number of cells.
        Iph: Photocurrent estimate (ampere) after calling :meth:`solve`.
        Is: Saturation current estimate (ampere) after calling :meth:`solve`.
        n: Diode ideality factor after calling :meth:`solve`.
        Rs: Series resistance (ohm) after calling :meth:`solve`.
        Rsh: Shunt resistance (ohm) after calling :meth:`solve`.
    """

    def __init__(
        self,
        Isc: float,
        Voc: float,
        Imp: float,
        Vmp: float,
        Ns: int,
        Tcell: float = 25.0,
    ) -> None:
        """
        Initialize the single-diode model from datasheet values.

        Args:
            Isc: Short-circuit current at STC (A).
            Voc: Open-circuit voltage at STC (V).
            Imp: Current at the maximum power point (A).
            Vmp: Voltage at the maximum power point (V).
            Ns: Number of cells connected in series inside the module.
            Tcell: Cell temperature in °C. Defaults to 25 °C for STC.
        """
        self.Isc = Isc
        self.Voc = Voc
        self.Imp = Imp
        self.Vmp = Vmp
        self.Ns = Ns
        self.Tcell = Tcell

        # Thermal voltages (per cell and per module)
        self.Vt = (Boltzmann * (Tcell + 273.15)) / elementary_charge
        self.Vt_mod = self.Vt * Ns

        # Unknown parameters solved later
        self.Iph: float | None = None
        self.Is: float | None = None
        self.n: float | None = None
        self.Rs: float | None = None
        self.Rsh: float | None = None

    def single_diode_equation(
        self,
        V: float,
        I: float,
        Iph: float,
        Is: float,
        n: float,
        Rs: float,
        Rsh: float,
    ) -> float:
        """
        Evaluate the residual of the implicit single-diode equation.

        Args:
            V: Terminal voltage (V).
            I: Terminal current (A).
            Iph: Photocurrent candidate (A).
            Is: Saturation current candidate (A).
            n: Diode ideality factor candidate.
            Rs: Series resistance candidate (Ω).
            Rsh: Shunt resistance candidate (Ω).

        Returns:
            float: Residual of the current equation. Zero indicates that
            the provided (V, I) pair lies exactly on the IV curve for the
            supplied parameters.
        """
        return (
            Iph
            - Is * (np.exp((V + I * Rs) / (n * self.Vt_mod)) - 1.0)
            - (V + I * Rs) / Rsh
            - I
        )

    def _di_dv(
        self,
        V: float,
        I: float,
        Is: float,
        n: float,
        Rs: float,
        Rsh: float,
    ) -> float:
        """
        Slope dI/dV of the implicit IV curve at a given operating point.

        From the implicit form ``F(V, I) = 0``, the slope is
        ``dI/dV = −F_V / F_I`` with the partial derivatives of the
        single-diode equation.

        Args:
            V: Terminal voltage (V).
            I: Terminal current (A).
            Is: Saturation current (A).
            n: Diode ideality factor.
            Rs: Series resistance (Ω).
            Rsh: Shunt resistance (Ω).

        Returns:
            float: dI/dV (A/V), negative on a physical IV curve.
        """
        exp_term = Is * np.exp((V + I * Rs) / (n * self.Vt_mod)) / (n * self.Vt_mod)
        f_v = -exp_term - 1.0 / Rsh
        f_i = -exp_term * Rs - Rs / Rsh - 1.0
        return -f_v / f_i

    def residuals(self, params: np.ndarray) -> list[float]:
        """
        Build the five residual equations used by the solver.

        Three datasheet points alone leave the five-parameter problem
        underdetermined (the optimizer can stall on a bound with ~0.3 A
        of error on modern half-cut modules). Two classical derivative
        conditions close the system:

        1–3. The curve passes through (0, Isc), (Voc, 0) and (Vmp, Imp).
        4. ``dP/dV = 0`` at the maximum power point
           (``Imp + Vmp · dI/dV = 0``).
        5. The short-circuit slope equals the shunt conductance:
           ``dI/dV |_(0, Isc) = −1/Rsh``.

        Args:
            params: Iterable of [Iph, Is, n, Rs, Rsh].

        Returns:
            list[float]: Five residuals that should all go to zero.
        """
        Iph, Is, n, Rs, Rsh = params

        r1 = self.single_diode_equation(0.0, self.Isc, Iph, Is, n, Rs, Rsh)
        r2 = self.single_diode_equation(self.Voc, 0.0, Iph, Is, n, Rs, Rsh)
        r3 = self.single_diode_equation(self.Vmp, self.Imp, Iph, Is, n, Rs, Rsh)
        # 4. Maximum-power condition at the MPP.
        r4 = self.Imp + self.Vmp * self._di_dv(self.Vmp, self.Imp, Is, n, Rs, Rsh)
        # 5. Shunt-dominated slope at short circuit.
        r5 = self._di_dv(0.0, self.Isc, Is, n, Rs, Rsh) + 1.0 / Rsh

        return [r1, r2, r3, r4, r5]

    def solve(self) -> tuple[float, float, float, float, float]:
        """
        Estimate the five single-diode parameters through non-linear least squares.

        The optimization runs in a **log-space parametrization** for the
        saturation current and the shunt resistance
        (``[Iph, log10(Is), n, Rs, log10(Rsh)]``): the raw variables span
        ten orders of magnitude (Is ~ 1e-7 A vs Rsh ~ 1e4 Ω) and a linear
        parametrization makes the trust-region optimizer stall on the
        bounds with ~0.3 A of residual error on modern half-cut modules.
        In log space the five conditions (three datasheet points + the
        two derivative constraints of :meth:`residuals`) converge to
        residuals below 1e-6 from a single standard initial guess.

        Returns:
            tuple: Estimated (Iph, Is, n, Rs, Rsh).

        Raises:
            RuntimeError: If the optimization does not converge or the
                residuals stay large (non-physical datasheet input).
        """

        def residuals_log(q: np.ndarray) -> list[float]:
            iph, log_is, n, rs, log_rsh = q
            return self.residuals([iph, 10.0**log_is, n, rs, 10.0**log_rsh])

        x0 = [self.Isc * 1.02, -9.0, 1.1, 0.15, 2.5]
        # Iph cannot exceed ~2× the datasheet Isc; the ideality-factor
        # window is wider than the textbook [1, 2] because the effective
        # module-level n of half-cut/multi-busbar designs can dip below 1.
        bounds = (
            [0.0, -14.0, 0.7, 0.0, 0.5],
            [2.0 * self.Isc, -3.0, 2.5, 2.0, 6.0],
        )
        result = least_squares(
            residuals_log,
            x0,
            bounds=bounds,
            xtol=1e-14,
            ftol=1e-14,
            gtol=1e-14,
        )

        # Convergence is judged on the residuals themselves: with the very
        # tight xtol/ftol the optimizer may stop on the iteration budget
        # (success=False) while the fit is already exact to 1e-7.
        max_residual = float(np.abs(result.fun).max())
        if max_residual > 1e-3:
            raise RuntimeError(
                "Single-diode parameter estimation did not converge "
                f"(max residual {max_residual:.2e}). Check the datasheet "
                "inputs (Isc, Voc, Imp, Vmp, Ns) for typos."
            )

        iph, log_is, n, rs, log_rsh = result.x
        self.Iph, self.Is, self.n, self.Rs, self.Rsh = (
            float(iph), float(10.0**log_is), float(n), float(rs), float(10.0**log_rsh),
        )
        return (self.Iph, self.Is, self.n, self.Rs, self.Rsh)

    def i_of_v(self, V: np.ndarray | float) -> np.ndarray:
        """
        Solve the implicit IV equation for the current at given voltages.

        Args:
            V: Scalar or array of voltages (V) where the IV curve should be evaluated.

        Returns:
            np.ndarray: Current values (A) corresponding to the provided voltages.

        Raises:
            RuntimeError: If :meth:`solve` has not been executed yet.
        """
        if None in (self.Iph, self.Is, self.n, self.Rs, self.Rsh):
            raise RuntimeError("Call solve() before evaluating the IV curve")

        Ivals = []
        for v in np.atleast_1d(V):
            # Use short-circuit current as the initial guess for the solver.
            fun = lambda i: self.single_diode_equation(  # noqa: E731
                v, i, self.Iph, self.Is, self.n, self.Rs, self.Rsh
            )
            sol = least_squares(fun, 0.9 * self.Isc)
            Ivals.append(sol.x[0])
        return np.array(Ivals)
