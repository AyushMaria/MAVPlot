#!/usr/bin/env python3
"""
cli.py

Headless CLI entry point for MAVPlot.  (#1)

Usage:
    python cli.py <log_file> [options]

Examples:
    python cli.py flight.tlog --prompt "Plot altitude over time"
    python cli.py flight.tlog --prompt "Show battery voltage" --retries 5
    python cli.py flight.tlog  # interactive prompt mode
"""

import argparse
import logging
import sys

from llm.gptPlotCreator import PlotCreator


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mavplot",
        description="MAVPlot — natural language drone flight log visualiser (headless CLI)",
    )
    p.add_argument(
        "log_file",
        help="Path to a MAVLink log file (.tlog, .bin, .log)",
    )
    p.add_argument(
        "--prompt", "-p",
        default=None,
        help="Plot request (e.g. 'Plot altitude over time'). "
             "If omitted, enters interactive prompt mode.",
    )
    p.add_argument(
        "--retries", "-r",
        type=int,
        default=3,
        help="Max self-healing retries on script failure (default: 3)",
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    return p


def run_once(creator: PlotCreator, prompt: str) -> None:
    """Execute one plot request and print the result path."""
    print(f"\n🔍 Finding relevant data types for: '{prompt}'")
    data_types = creator.find_relevant_data_types(prompt)

    print("✍️  Generating plot script with GPT...")
    creator.create_plot(prompt, data_types)

    print("⚙️  Running script...")
    plot_result, code = creator.run_script()

    print("\n" + "─" * 60)
    print("Generated code:")
    print("─" * 60)
    print(code)
    print("─" * 60)

    plot_path = creator.plot_path
    if plot_path:
        print(f"\n✅ Plot saved to: {plot_path}")
    else:
        print("\n⚠️  Plot file not found — check the generated code above for errors.")


def interactive_mode(creator: PlotCreator) -> None:
    """Run a REPL loop accepting plot requests until the user quits."""
    print("\nEntering interactive mode. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            prompt = input("📊 Plot request > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        try:
            run_once(creator, prompt)
        except Exception as exc:
            print(f"❌ Error: {exc}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(args.verbose)
    log = logging.getLogger("mavplot.cli")

    # Initialise PlotCreator
    creator = PlotCreator(max_retries=args.retries)

    # Validate and register log file  (#9 handled inside set_logfile_name)
    try:
        creator.set_logfile_name(args.log_file)
    except ValueError as exc:
        log.error("%s", exc)
        sys.exit(1)

    # Parse log and build vector index
    print(f"\n📂 Parsing log file: {args.log_file}")
    try:
        creator.parse_mavlink_log()
    except Exception as exc:
        log.error("Failed to parse log file: %s", exc)
        sys.exit(1)
    print("✅ Log parsed and vector index built.\n")

    # Single prompt or interactive mode
    if args.prompt:
        try:
            run_once(creator, args.prompt)
        except Exception as exc:
            log.error("Plot generation failed: %s", exc)
            sys.exit(1)
    else:
        interactive_mode(creator)


if __name__ == "__main__":
    main()
