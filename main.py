#!/usr/bin/env python3
"""
Command‑line entry point for the signal‑processing pipeline.

Example:
    python main.py data.csv --recompute --debug

Options:
    --recompute      Ignore cached results and force full reprocessing
    --debug          Print extra diagnostic messages
    --cache-root DIR Custom directory for cache files (default: ~/.signal_pipeline_cache)
    --coef FORMULA   Override coefficient formula (Python expression, see README)

The script prints a short summary and relies on PipelineSettings for the heavy lifting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reo_tools.accessor import ResultAccessor
from reo_tools.cache import CSVCache
from reo_tools.pipeline import SignalPipeline
from reo_tools.reader import CSVReader
from reo_tools.settings import PipelineSettings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Signal‑processing pipeline CLI")
    parser.add_argument("file", help="Path to a CSV file with raw RVG data")
    parser.add_argument("--recompute", action="store_true", help="Force recalculation, ignore cache")
    parser.add_argument("--debug", action="store_true", help="Verbose pipeline debug output")
    parser.add_argument("--cache-root", type=str, default=None, help="Directory to store cached Parquet files")
    parser.add_argument(
        "--coef", dest="coef_formula", default=None, help="Custom coefficient formula (Python expression)"
    )
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> PipelineSettings:
    cache_backend = CSVCache()
    return PipelineSettings(
        cache_backend=cache_backend,
        reader=CSVReader(),
        recompute=args.recompute,
        coef_formula=args.coef_formula,
        debug=args.debug,
    )


def main() -> None:
    args = _parse_args()
    cfg = _build_config(args)

    pipeline = SignalPipeline(cfg)
    raw_result = pipeline.run(Path(args.file))

    # Wrap for convenient access in interactive sessions or further processing
    result = ResultAccessor(raw_result)

    print("\n=== Available datasets ===")
    for name in result.datasets():
        print(f"- {name}: {len(result.all(name))} rows")

    # Show basic info about the main time‑series table
    if "time_series" in result.datasets():
        print("\n=== Columns in 'time_series' ===")
        print(", ".join(result.headers()["time_series"]))


if __name__ == "__main__":
    main()
