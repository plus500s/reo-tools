#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from reo_tools.accessor import ResultAccessor
from reo_tools.cache import CSVCache, DataCache, ParquetCache
from reo_tools.pipeline import SignalPipeline
from reo_tools.reader import BaseReader, CSVReader
from reo_tools.settings import PipelineSettings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="reo-tools", description="Run the reo-tools pipeline")
    p.add_argument("file", type=Path, help="Path to the CSV file with raw signal data")
    p.add_argument("--output", "-o", type=Path, help="Directory to store resulting files")
    p.add_argument("--cache", choices=["csv", "parquet"], default="csv", help="Cache backend to use")
    p.add_argument("--reader", choices=["csv"], default="csv", help="Reader to use for input files")
    p.add_argument("--coef", dest="coef_formula", help="Custom coefficient formula (Python expression)")
    p.add_argument("--recompute", action="store_true", help="Ignore cache and force a full recomputation")
    p.add_argument("--debug", action="store_true", help="Print full traceback on error")
    p.add_argument("--show-progress", "-p", action="store_true", help="Show progress messages during processing")
    return p.parse_args()


def select_cache(kind: str, root: Path | None) -> DataCache:
    if kind == "csv":
        return CSVCache(cache_root=root)
    if kind == "parquet":
        return ParquetCache(cache_root=root)
    raise ValueError(f"Unsupported cache type: {kind!r}")


def select_reader(kind: str) -> BaseReader:
    if kind == "csv":
        return CSVReader()
    raise ValueError(f"Unsupported reader type: {kind!r}")


def build_cfg(args: argparse.Namespace) -> PipelineSettings:
    return PipelineSettings(
        reader=select_reader(args.reader),
        cache_backend=select_cache(args.cache, args.output),
        coef_formula=args.coef_formula,
        recompute=args.recompute,
        debug=args.debug,
        show_progress=args.show_progress,
    )


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)

    try:
        raw_result = SignalPipeline(cfg).run(args.file)
        if args.debug:
            accessor = ResultAccessor(raw_result)
            print("\nPipeline outputs:", file=sys.stderr)
            for name in accessor.datasets():
                print(f"• {name}: {len(accessor.all(name))} rows", file=sys.stderr)

        payload = {"status": "ok"}
        exit_code = 0
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        payload = {"status": "error", "message": str(exc)}
        exit_code = 1

    print(json.dumps(payload, ensure_ascii=False), flush=True, file=sys.stderr)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
