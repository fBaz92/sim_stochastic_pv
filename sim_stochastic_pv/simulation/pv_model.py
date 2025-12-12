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

    def residuals(self, params: np.ndarray) -> list[float]:
        """
        Build the three residual equations used by the solver.

        The optimizer enforces that the single-diode curve passes through
        the short-circuit point, the open-circuit point, and the maximum
        power point supplied at initialization time.

        Args:
            params: Iterable of [Iph, Is, n, Rs, Rsh].

        Returns:
            list[float]: Residuals for (Isc, Voc, MPP) that should go to zero.
        """
        Iph, Is, n, Rs, Rsh = params

        r1 = self.single_diode_equation(0.0, self.Isc, Iph, Is, n, Rs, Rsh)
        r2 = self.single_diode_equation(self.Voc, 0.0, Iph, Is, n, Rs, Rsh)
        r3 = self.single_diode_equation(self.Vmp, self.Imp, Iph, Is, n, Rs, Rsh)

        return [r1, r2, r3]

    def solve(self) -> tuple[float, float, float, float, float]:
        """
        Estimate the five single-diode parameters through non-linear least squares.

        Uses SciPy's :func:`least_squares` with tight tolerances and physical
        bounds to obtain a robust solution starting from reasonable guesses.

        Returns:
            tuple: Estimated (Iph, Is, n, Rs, Rsh).

        Raises:
            RuntimeError: If the optimization does not converge.
        """
        Iph0 = self.Isc * 1.02
        Is0 = 1e-10
        n0 = 1.3
        Rs0 = 0.2
        Rsh0 = 1000.0
        x0 = [Iph0, Is0, n0, Rs0, Rsh0]

        result = least_squares(
            self.residuals,
            x0,
            bounds=([0.0, 0.0, 1.0, 0.0, 1.0], [10.0, 1e-3, 2.0, 10.0, 1e5]),
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )

        if not result.success:
            raise RuntimeError("Single-diode parameter estimation did not converge")

        self.Iph, self.Is, self.n, self.Rs, self.Rsh = result.x
        return tuple(float(v) for v in result.x)

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
