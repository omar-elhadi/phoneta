"""Command-line entry point for Phoneta."""

from __future__ import annotations

import argparse
import sys

from phoneta import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="phoneta",
        description="Offline, privacy-first pronunciation evaluator (English + French).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Phoneta desktop app."""
    build_parser().parse_args(argv)
    from phoneta.app import run
    return run()


if __name__ == "__main__":
    sys.exit(main())
