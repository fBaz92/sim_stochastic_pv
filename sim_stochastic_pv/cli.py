from __future__ import annotations

import argparse
from typing import Sequence

from .application import SimulationApplication
from .db.session import init_db
from .persistence import PersistenceService
from .result_builder import ResultBuilder


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the CLI argument parser used by entry points.

    Returns:
        Configured argparse.ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(description="Simulatore stocastico PV CLI")
    sub = parser.add_subparsers(dest="command")

    analyze = sub.add_parser("analyze", help="Esegui analisi singolo scenario")
    analyze.add_argument(
        "--no-save",
        action="store_true",
        help="Non salvare i file di output nella cartella results",
    )

    optimize = sub.add_parser("optimize", help="Esegui ottimizzazione multi scenario")
    optimize.add_argument(
        "--no-save",
        action="store_true",
        help="Non salvare i file di output nella cartella results",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """
    CLI entry point for running single analyses or optimization batches.

    Args:
        argv: Optional sequence of CLI args (defaults to sys.argv).
    """
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    init_db()
    persistence = PersistenceService()
    result_builder = ResultBuilder()

    app = SimulationApplication(
        save_outputs=not getattr(args, "no_save", False),
        persistence=persistence,
        result_builder=result_builder,
    )

    if args.command == "analyze":
        summary = app.run_analysis()
        print("Analisi completata:", summary)
    elif args.command == "optimize":
        summary = app.run_optimization()
        print("Ottimizzazione completata:", summary)
    else:
        parser.error(f"Comando non riconosciuto: {args.command}")
