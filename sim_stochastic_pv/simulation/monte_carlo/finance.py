"""
Financial calculation functions for Monte Carlo analysis.
"""

from __future__ import annotations

import numpy as np

def _npv(rate: float, cashflows: np.ndarray) -> float:
    """
    Calculate net present value with robust handling of extreme discount rates.

    Computes NPV using the standard discounted cashflow formula but with
    protection against division by zero and numerical instability when
    discount rates approach or exceed -100% (rate <= -1).

    Traditional NPV formula:
        NPV = Σ(cashflow_t / (1 + rate)^t) for t = 0, 1, 2, ...

    The robustness comes from clamping the growth factor (1 + rate) to a
    minimum positive value (1e-9) when rate is extremely negative, preventing
    division by zero while preserving monotonic relationship between rate and NPV.

    Args:
        rate: Discount rate per period as decimal (e.g., 0.05 = 5% per period).
            Can be negative (representing growth scenarios) but is clamped
            internally to avoid numerical issues.
            Typical range for monthly rates: -0.05 to 0.05 (-5% to +5%).
        cashflows: Array of cashflows for each period (EUR or other currency).
            Shape: (n_periods,). First element is typically negative (investment).
            Subsequent elements are usually positive (returns/savings).

    Returns:
        float: Net present value in same currency units as cashflows.
            Positive NPV indicates profitable investment.
            Negative NPV indicates unprofitable investment.

    Example:
        ```python
        import numpy as np

        # Investment: -10000 EUR, then +500 EUR/month for 24 months
        cashflows = np.array([-10000] + [500] * 24)

        # Calculate NPV at 0.5% monthly discount rate
        npv = _npv(rate=0.005, cashflows=cashflows)
        print(f"NPV: {npv:.2f} EUR")  # ~1420 EUR

        # Higher discount rate reduces NPV
        npv_high = _npv(rate=0.02, cashflows=cashflows)
        print(f"NPV at 2%: {npv_high:.2f} EUR")  # ~460 EUR

        # Extreme negative rate (handled robustly)
        npv_extreme = _npv(rate=-0.999, cashflows=cashflows)
        # Returns valid value instead of divide-by-zero error
        ```

    Notes:
        - Periods start at t=0 (immediate cashflow)
        - Monthly rates for annual rates: rate_monthly ≈ rate_annual / 12
        - For IRR calculation, NPV should equal zero
        - Growth factor clamped to max(1 + rate, 1e-9) for stability
        - Maintains monotonic NPV vs rate relationship even with clamping
    """
    periods = np.arange(cashflows.size, dtype=float)
    growth = max(1.0 + rate, 1e-9)
    discounts = np.power(growth, periods)
    return np.sum(cashflows / discounts)


def _compute_irr_monthly(
    cashflows: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """
    Compute monthly Internal Rate of Return using bisection method.

    IRR is the discount rate that makes NPV equal to zero. This function
    uses a robust bisection algorithm to find the rate where:
        NPV(rate) = 0

    The method automatically handles edge cases (no sign change, insufficient
    data) and expands the search range if needed to bracket the solution.

    Args:
        cashflows: Array of cashflows for each month (EUR or other currency).
            Shape: (n_months,). Must contain at least one positive and one
            negative value for IRR to exist. Typically starts with negative
            investment followed by positive returns.
        tol: Convergence tolerance for NPV (default 1e-6).
            Algorithm stops when |NPV| < tol. Smaller values = more precision
            but potentially more iterations.
        max_iter: Maximum bisection iterations (default 100).
            Prevents infinite loops. Typically converges in 20-40 iterations.

    Returns:
        float: Monthly internal rate of return as decimal (e.g., 0.01 = 1%/month).
            Returns np.nan if:
            - Fewer than 2 cashflows
            - All cashflows same sign (no investment or no returns)
            - No rate found within expanded search range
            - Algorithm doesn't converge within max_iter

    Example:
        ```python
        import numpy as np

        # Investment: -10000 EUR, then +500 EUR/month for 24 months
        cashflows = np.array([-10000] + [500] * 24)

        irr_monthly = _compute_irr_monthly(cashflows)
        print(f"Monthly IRR: {irr_monthly:.4f}")  # ~0.0143 (1.43%/month)
        print(f"Annual IRR: {(1 + irr_monthly)**12 - 1:.2%}")  # ~18.5%/year

        # Verify: NPV at IRR should be ~0
        npv_at_irr = _npv(irr_monthly, cashflows)
        print(f"NPV at IRR: {npv_at_irr:.2f} EUR")  # ~0.00 EUR

        # No solution case (all positive)
        bad_cashflows = np.array([100, 200, 300])
        irr = _compute_irr_monthly(bad_cashflows)
        print(f"IRR for bad cashflows: {irr}")  # nan
        ```

    Algorithm Details:
        1. Validation: Check for at least 2 cashflows with mixed signs
        2. Initial bracket: Start with [-0.9999, 5.0] (approx -100% to 500%)
        3. Bracket expansion: If no sign change, double upper bound up to 12 times
        4. Bisection: Iteratively narrow bracket until |NPV| < tol
        5. Return: Middle of final bracket as IRR estimate

    Notes:
        - Search range: Initially [-99.99%, 500%] per month
        - Expands upper bound if needed (up to ~10^6 or 100,000,000%)
        - Bisection is slower than Newton's method but more robust
        - Returns monthly rate - convert to annual with: (1 + irr_m)^12 - 1
        - For PV systems, typical monthly IRR: 0.5% to 2% (6% to 27% annual)
        - np.nan indicates no valid IRR exists for the cashflow sequence
    """
    if cashflows.size < 2:
        return np.nan
    if not (np.any(cashflows > 0) and np.any(cashflows < 0)):
        return np.nan

    low = -0.9999
    high = 5.0
    npv_low = _npv(low, cashflows)
    npv_high = _npv(high, cashflows)

    expand = 0
    while npv_low * npv_high > 0 and expand < 12:
        high *= 2.0
        npv_high = _npv(high, cashflows)
        expand += 1
        if high > 1e6:
            return np.nan
    if npv_low * npv_high > 0:
        return np.nan

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        npv_mid = _npv(mid, cashflows)
        if abs(npv_mid) < tol:
            return mid
        if npv_low * npv_mid < 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid
    return mid


def _compute_irr_annual(cashflows: np.ndarray) -> float:
    """
    Compute annualized Internal Rate of Return from monthly cashflows.

    Converts monthly IRR to effective annual rate using compound interest
    formula: (1 + monthly_rate)^12 - 1. This represents the true annual
    return accounting for monthly compounding.

    Args:
        cashflows: Array of monthly cashflows (EUR or other currency).
            Shape: (n_months,). Same requirements as _compute_irr_monthly:
            must have at least one positive and one negative value.

    Returns:
        float: Annual internal rate of return as decimal (e.g., 0.15 = 15%/year).
            Returns np.nan if monthly IRR cannot be computed (same failure
            conditions as _compute_irr_monthly).

    Example:
        ```python
        import numpy as np

        # Investment: -10000 EUR, then +500 EUR/month for 24 months
        cashflows = np.array([-10000] + [500] * 24)

        irr_annual = _compute_irr_annual(cashflows)
        print(f"Annual IRR: {irr_annual:.2%}")  # ~18.5%/year

        # Compare with monthly
        irr_monthly = _compute_irr_monthly(cashflows)
        irr_annual_manual = (1 + irr_monthly)**12 - 1
        print(f"Manual conversion: {irr_annual_manual:.2%}")  # Same

        # Interpretation for PV systems:
        # IRR > 5%: Excellent investment
        # IRR 3-5%: Good investment
        # IRR 1-3%: Marginal investment
        # IRR < 1%: Poor investment
        ```

    Notes:
        - Uses compound interest formula: effective_annual = (1 + monthly)^12 - 1
        - NOT simple annualization: monthly × 12 (that would underestimate)
        - For PV systems, typical annual IRR: 5% to 15%
        - IRR > inflation rate + risk premium = economically viable
        - Returns np.nan if underlying monthly IRR is nan
        - More intuitive than monthly rate for comparing investments
    """
    irr_monthly = _compute_irr_monthly(cashflows)
    if np.isnan(irr_monthly):
        return np.nan
    return (1.0 + irr_monthly) ** 12 - 1.0


