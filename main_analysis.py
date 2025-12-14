from __future__ import annotations

import warnings

warnings.warn(
    "main_analysis.py is deprecated. Use the CLI instead: 'pv-sim analyze'",
    DeprecationWarning,
    stacklevel=2
)

from sim_stochastic_pv.cli import main


if __name__ == "__main__":
    main(["analyze"])
