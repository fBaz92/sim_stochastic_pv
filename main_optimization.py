import warnings

warnings.warn(
    "main_optimization.py is deprecated. Use the CLI instead: 'pv-sim optimize'",
    DeprecationWarning,
    stacklevel=2
)

from sim_stochastic_pv.cli import main


if __name__ == "__main__":
    main(["optimize"])
