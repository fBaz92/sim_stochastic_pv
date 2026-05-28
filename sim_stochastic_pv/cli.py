from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .application import SimulationApplication
from .db.session import init_db
from .persistence import PersistenceService
from .output import ResultBuilder


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
    analyze.add_argument(
        "--scenario-file",
        "--file",
        dest="scenario_file",
        type=str,
        default=None,
        help="Percorso a un file JSON con la definizione dello scenario",
    )
    analyze.add_argument(
        "--n-mc",
        type=int,
        default=None,
        help="Numero di simulazioni Monte Carlo da eseguire",
    )
    analyze.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed RNG per la riproducibilità (default: 123)",
    )

    optimize = sub.add_parser("optimize", help="Esegui ottimizzazione multi scenario")
    optimize.add_argument(
        "--no-save",
        action="store_true",
        help="Non salvare i file di output nella cartella results",
    )
    optimize.add_argument(
        "--scenario-file",
        "--file",
        dest="scenario_file",
        type=str,
        default=None,
        help="Percorso a un file JSON con la definizione dello scenario",
    )
    optimize.add_argument(
        "--n-mc",
        type=int,
        default=None,
        help="Numero di simulazioni Monte Carlo per scenario durante l'ottimizzazione",
    )
    optimize.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Seed RNG per la riproducibilità (default: 123)",
    )

    # Hardware management
    hardware = sub.add_parser("hardware", help="Gestisci il catalogo hardware nel database")
    hardware_sub = hardware.add_subparsers(dest="hardware_command")

    hw_list = hardware_sub.add_parser("list", help="Elenca gli hardware registrati")
    hw_list.add_argument(
        "--type",
        choices=["all", "inverter", "panel", "battery"],
        default="all",
        help="Filtra per tipologia",
    )

    hw_inv = hardware_sub.add_parser("upsert-inverter", help="Crea o aggiorna un inverter")
    hw_inv.add_argument("--name", required=True)
    hw_inv.add_argument("--manufacturer")
    hw_inv.add_argument("--model-number")
    hw_inv.add_argument("--p-ac-max-kw", type=float, required=True, dest="p_ac_max_kw")
    hw_inv.add_argument("--p-dc-max-kw", type=float, dest="p_dc_max_kw")
    hw_inv.add_argument("--price-eur", type=float, dest="price_eur")
    hw_inv.add_argument("--install-cost-eur", type=float, dest="install_cost_eur")
    hw_inv.add_argument("--datasheet")
    hw_inv.add_argument(
        "--integrated-battery-capacity-kwh",
        type=float,
        dest="integrated_battery_capacity_kwh",
    )
    hw_inv.add_argument(
        "--integrated-battery-cycles-life",
        type=int,
        dest="integrated_battery_cycles_life",
    )
    hw_inv.add_argument(
        "--integrated-battery-price-eur",
        type=float,
        dest="integrated_battery_price_eur",
    )
    hw_inv.add_argument(
        "--integrated-battery-count-options",
        type=str,
        dest="integrated_battery_count_options",
        help="Lista di numeri separati da virgola (es. 0,1,2)",
    )

    hw_panel = hardware_sub.add_parser("upsert-panel", help="Crea o aggiorna un pannello")
    hw_panel.add_argument("--name", required=True)
    hw_panel.add_argument("--manufacturer")
    hw_panel.add_argument("--model-number")
    hw_panel.add_argument("--power-w", type=float, required=True, dest="power_w")
    hw_panel.add_argument("--price-eur", type=float, dest="price_eur")
    hw_panel.add_argument("--datasheet")

    hw_batt = hardware_sub.add_parser("upsert-battery", help="Crea o aggiorna una batteria")
    hw_batt.add_argument("--name", required=True)
    hw_batt.add_argument("--manufacturer")
    hw_batt.add_argument("--model-number")
    hw_batt.add_argument("--capacity-kwh", type=float, required=True, dest="capacity_kwh")
    hw_batt.add_argument("--cycles-life", type=int, dest="cycles_life", default=5000)
    hw_batt.add_argument("--price-eur", type=float, dest="price_eur")
    hw_batt.add_argument("--datasheet")

    # Scenario management
    scenario = sub.add_parser("scenario", help="Gestisci scenari salvati")
    scenario_sub = scenario.add_subparsers(dest="scenario_command")

    scenario_list = scenario_sub.add_parser("list", help="Elenca gli scenari salvati")
    scenario_list.add_argument(
        "--json",
        action="store_true",
        help="Stampa le configurazioni complete in formato JSON",
    )

    scenario_save = scenario_sub.add_parser("save", help="Salva/aggiorna uno scenario da file JSON")
    scenario_save.add_argument("--name", required=True, help="Nome configurazione")
    scenario_save.add_argument("--file", required=True, help="Percorso file JSON")

    scenario_run = scenario_sub.add_parser("run", help="Esegui uno scenario")
    _add_execution_arguments(scenario_run, "scenario", 123)

    # Optimization management (a.k.a. "campaign" / "design" in the UI glossary
    # — see CLAUDE.md §Glossario. All three command names dispatch to the same
    # subparser group and back to the same DB config_type='optimization', so
    # backward compat is preserved for any scripts still calling `optimization`.)
    optimization = sub.add_parser(
        "optimization",
        aliases=["campaign", "design"],
        help="Gestisci campagne multi-scenario (alias: campaign, design)",
    )
    optimization_sub = optimization.add_subparsers(dest="optimization_command")

    optimization_list = optimization_sub.add_parser("list", help="Elenca le ottimizzazioni salvate")
    optimization_list.add_argument(
        "--json",
        action="store_true",
        help="Stampa la configurazione completa in JSON",
    )

    optimization_save = optimization_sub.add_parser("save", help="Salva/aggiorna un'ottimizzazione da file JSON")
    optimization_save.add_argument("--name", required=True)
    optimization_save.add_argument("--file", required=True)

    optimization_run = optimization_sub.add_parser("run", help="Esegui un'ottimizzazione")
    _add_execution_arguments(optimization_run, "ottimizzazione", 123)

    return parser


def _add_execution_arguments(
    parser: argparse.ArgumentParser,
    config_type: str,
    default_seed: int
) -> None:
    """Add common execution arguments to a run subparser.

    Args:
        parser: Subparser to add arguments to
        config_type: Type of config ("scenario" or "ottimizzazione")
        default_seed: Default RNG seed value
    """
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file",
        help=f"Config JSON inline per {config_type} (non viene salvato)"
    )
    group.add_argument(
        "--name",
        help=f"Nome {config_type} salvato"
    )
    group.add_argument(
        "--id",
        type=int,
        help=f"ID {config_type} salvato"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help=f"Seed RNG (default {default_seed})"
    )
    parser.add_argument(
        "--n-mc",
        type=int,
        help="Numero di simulazioni Monte Carlo"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Non salvare report su disco anche se ResultBuilder è disponibile"
    )


def _load_json_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise SystemExit(f"File non trovato: {file_path}")
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"File JSON non valido ({file_path}): {exc}") from exc


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, default=str, ensure_ascii=False))


def _parse_int_list(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def _select_configuration(
    persistence: PersistenceService,
    *,
    config_id: int | None,
    name: str | None,
    expected_type: str,
) -> Any:
    record = None
    if config_id is not None:
        record = persistence.get_configuration_by_id(config_id)
    elif name:
        record = persistence.get_configuration_by_name(name)

    if record is None:
        target = f"ID={config_id}" if config_id is not None else f"nome '{name}'"
        raise SystemExit(f"Configurazione non trovata ({target}).")
    if record.config_type != expected_type:
        raise SystemExit(
            f"Configurazione '{record.name}' è di tipo {record.config_type}, atteso {expected_type}."
        )
    return record


def _scenario_payload_from_args(
    args: argparse.Namespace,
    persistence: PersistenceService,
) -> dict[str, Any]:
    if args.file:
        return _load_json_file(args.file)
    config = _select_configuration(
        persistence,
        config_id=args.id,
        name=args.name,
        expected_type="scenario",
    )
    return persistence.hydrate_scenario_from_ids(config.data)


def _optimization_payload_from_args(
    args: argparse.Namespace,
    persistence: PersistenceService,
) -> dict[str, Any]:
    if args.file:
        return _load_json_file(args.file)
    config = _select_configuration(
        persistence,
        config_id=args.id,
        name=args.name,
        expected_type="optimization",
    )
    return persistence.hydrate_scenario_from_ids(config.data)


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

    save_outputs = not getattr(args, "no_save", False)

    app = SimulationApplication(
        save_outputs=save_outputs,
        persistence=persistence,
        result_builder=None,
    )

    if args.command == "analyze":
        if app.save_outputs and app.result_builder is None:
            app.result_builder = ResultBuilder()
        scenario_path = getattr(args, "scenario_file", None)
        if scenario_path:
            scenario_data = _load_json_file(scenario_path)
            # Validate before running so errors are reported with exit code 1
            from .validation import validate_scenario
            errors = validate_scenario(scenario_data)
            if errors:
                print("❌ Errori di configurazione:", file=sys.stderr)
                for err in errors:
                    print(f"  • {err}", file=sys.stderr)
                sys.exit(1)
        else:
            scenario_data = None
        summary = app.run_analysis(
            n_mc=getattr(args, "n_mc", None),
            seed=getattr(args, "seed", 123),
            scenario_data=scenario_data,
        )
        _print_json(summary)
        return

    if args.command == "optimize":
        if app.save_outputs and app.result_builder is None:
            app.result_builder = ResultBuilder()
        scenario_path = getattr(args, "scenario_file", None)
        if scenario_path:
            scenario_data = _load_json_file(scenario_path)
            # Validate before running so errors are reported with exit code 1
            from .validation import validate_optimization
            errors = validate_optimization(scenario_data)
            if errors:
                print("❌ Errori di configurazione:", file=sys.stderr)
                for err in errors:
                    print(f"  • {err}", file=sys.stderr)
                sys.exit(1)
        else:
            scenario_data = None
        summary = app.run_optimization(
            n_mc=getattr(args, "n_mc", None),
            seed=getattr(args, "seed", 123),
            scenario_data=scenario_data,
        )
        _print_json(summary)
        return

    if args.command == "hardware":
        if not args.hardware_command:
            parser.error("Specificare un sottocomando hardware (list/upsert-...).")

        if args.hardware_command == "list":
            types = (
                ["inverter", "panel", "battery"]
                if args.type == "all"
                else [args.type]
            )
            payload: dict[str, Any] = {}
            if "inverter" in types:
                payload["inverters"] = [
                    {
                        "id": inv.id,
                        "name": inv.name,
                        "manufacturer": inv.manufacturer,
                        "p_ac_max_kw": (inv.specs or {}).get("p_ac_max_kw", inv.nominal_power_kw),
                        "price_eur": (inv.specs or {}).get("price_eur"),
                    }
                    for inv in persistence.list_inverters()
                ]
            if "panel" in types:
                payload["panels"] = [
                    {
                        "id": panel.id,
                        "name": panel.name,
                        "power_w": panel.power_w,
                        "price_eur": (panel.specs or {}).get("price_eur"),
                    }
                    for panel in persistence.list_panels()
                ]
            if "battery" in types:
                payload["batteries"] = [
                    {
                        "id": battery.id,
                        "name": battery.name,
                        "capacity_kwh": battery.capacity_kwh,
                        "price_eur": (battery.specs or {}).get("price_eur"),
                    }
                    for battery in persistence.list_batteries()
                ]
            _print_json(payload)
            return

        if args.hardware_command == "upsert-inverter":
            payload: dict[str, Any] = {
                "name": args.name,
                "manufacturer": args.manufacturer,
                "model_number": args.model_number,
                "p_ac_max_kw": args.p_ac_max_kw,
                "p_dc_max_kw": args.p_dc_max_kw,
                "price_eur": args.price_eur,
                "install_cost_eur": args.install_cost_eur,
                "datasheet": args.datasheet,
            }
            if args.integrated_battery_capacity_kwh is not None:
                payload["integrated_battery_specs"] = {
                    "capacity_kwh": args.integrated_battery_capacity_kwh,
                    "cycles_life": args.integrated_battery_cycles_life or 0,
                }
                if args.integrated_battery_price_eur is not None:
                    payload["integrated_battery_price_eur"] = args.integrated_battery_price_eur
                options = _parse_int_list(args.integrated_battery_count_options)
                if options:
                    payload["integrated_battery_count_options"] = options
            persistence.upsert_inverter(payload)
            print(f"Inverter '{args.name}' salvato/aggiornato.")
            return

        if args.hardware_command == "upsert-panel":
            payload = {
                "name": args.name,
                "manufacturer": args.manufacturer,
                "model_number": args.model_number,
                "power_w": args.power_w,
                "price_eur": args.price_eur,
                "datasheet": args.datasheet,
            }
            persistence.upsert_panel(payload)
            print(f"Pannello '{args.name}' salvato/aggiornato.")
            return

        if args.hardware_command == "upsert-battery":
            payload = {
                "name": args.name,
                "manufacturer": args.manufacturer,
                "model_number": args.model_number,
                "datasheet": args.datasheet,
                "specs": {
                    "capacity_kwh": args.capacity_kwh,
                    "cycles_life": args.cycles_life,
                    "price_eur": args.price_eur,
                },
                "price_eur": args.price_eur,
            }
            persistence.upsert_battery(payload)
            print(f"Batteria '{args.name}' salvata/aggiornata.")
            return

        parser.error(f"Sottocomando hardware non riconosciuto: {args.hardware_command}")

    if args.command == "scenario":
        if not args.scenario_command:
            parser.error("Specificare un sottocomando scenario (list/save/run).")

        if args.scenario_command == "list":
            configs = persistence.list_configurations("scenario")
            if args.json:
                data = [
                    {"id": cfg.id, "name": cfg.name, "data": cfg.data}
                    for cfg in configs
                ]
            else:
                data = [{"id": cfg.id, "name": cfg.name} for cfg in configs]
            _print_json(data)
            return

        if args.scenario_command == "save":
            data = _load_json_file(args.file)
            record = persistence.save_configuration(args.name, "scenario", data)
            print(f"Scenario '{record.name}' salvato con ID {record.id}.")
            return

        if args.scenario_command == "run":
            # Need to reconfigure save flag for this run (subcommand may set --no-save)
            save_outputs = not getattr(args, "no_save", False)
            app.save_outputs = save_outputs
            if save_outputs and app.result_builder is None:
                app.result_builder = ResultBuilder()
            if not save_outputs:
                app.result_builder = None
            payload = _scenario_payload_from_args(args, persistence)

            # Validate configuration
            from .validation import validate_scenario
            errors = validate_scenario(payload)
            if errors:
                print("❌ Errori di configurazione:", file=sys.stderr)
                for err in errors:
                    print(f"  • {err}", file=sys.stderr)
                sys.exit(1)

            summary = app.run_analysis(
                n_mc=args.n_mc,
                seed=args.seed or 123,
                scenario_data=payload,
            )
            _print_json(summary)
            return

        parser.error(f"Sottocomando scenario non riconosciuto: {args.scenario_command}")

    if args.command in {"optimization", "campaign", "design"}:
        if not args.optimization_command:
            parser.error(
                f"Specificare un sottocomando {args.command} (list/save/run)."
            )

        if args.optimization_command == "list":
            configs = persistence.list_configurations("optimization")
            if args.json:
                data = [
                    {"id": cfg.id, "name": cfg.name, "data": cfg.data}
                    for cfg in configs
                ]
            else:
                data = [{"id": cfg.id, "name": cfg.name} for cfg in configs]
            _print_json(data)
            return

        if args.optimization_command == "save":
            data = _load_json_file(args.file)
            record = persistence.save_configuration(args.name, "optimization", data)
            print(f"Ottimizzazione '{record.name}' salvata con ID {record.id}.")
            return

        if args.optimization_command == "run":
            save_outputs = not getattr(args, "no_save", False)
            app.save_outputs = save_outputs
            if save_outputs and app.result_builder is None:
                app.result_builder = ResultBuilder()
            if not save_outputs:
                app.result_builder = None
            payload = _optimization_payload_from_args(args, persistence)

            # Validate configuration
            from .validation import validate_optimization
            errors = validate_optimization(payload)
            if errors:
                print("❌ Errori di configurazione:", file=sys.stderr)
                for err in errors:
                    print(f"  • {err}", file=sys.stderr)
                sys.exit(1)

            summary = app.run_optimization(
                seed=args.seed or 123,
                n_mc=args.n_mc,
                scenario_data=payload,
            )
            _print_json(summary)
            return

        parser.error(f"Sottocomando optimization non riconosciuto: {args.optimization_command}")

    parser.error(f"Comando non riconosciuto: {args.command}")


if __name__ == "__main__":
    main()
