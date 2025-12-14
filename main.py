from __future__ import annotations

import warnings

warnings.warn(
    "main.py is deprecated. Use the CLI instead: 'pv-sim analyze' or 'pv-sim optimize'",
    DeprecationWarning,
    stacklevel=2
)

from main_analysis import main as analysis_main


if __name__ == "__main__":
    analysis_main()
