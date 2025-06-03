from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

__all__ = [
    "DataCache",
    "ParquetCache",
    "NoOpCache",
    "CSVCache",
]

DEFAULT_ROOT: str = ".pipeline_cache"


class DataCache(ABC):
    """Abstract cache interface."""

    EXT = ""

    def __init__(self, cache_root: Path | str | None = None) -> None:
        # default to .pipeline_cache in project root
        self.root = Path(cache_root).expanduser().resolve() if cache_root else Path.cwd() / DEFAULT_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save(self, scope: str | None, data: Mapping[str, pd.DataFrame]) -> Path:
        """Save a mapping of name→DataFrame under the given scope."""
        ...

    @abstractmethod
    def load(self, scope: str | None, columns: Sequence[str] | None = None) -> Mapping[str, pd.DataFrame] | None:
        """Load all cached DataFrames under the given scope."""
        ...

    def delete(self, scope: str | None) -> None:
        """Delete all cached DataFrames under the given scope."""
        target = (self.root / scope) if scope else self.root
        if not target.exists():
            return
        if scope is None:
            for path in target.glob(f"*{self.EXT}"):
                if path.is_file():
                    path.unlink(missing_ok=True)
        else:
            shutil.rmtree(target, ignore_errors=True)

    def list_scopes(self) -> Sequence[str]:
        """List all available scopes (subdirectories under the cache root)."""
        return [p.name for p in self.root.iterdir() if p.is_dir()]

    def _scope_dir(self, scope: str | None, *, create: bool = False) -> Path:
        """Return path to the scope directory.
        If *create* is True, the directory is created (mkdir -p)."""
        path = self.root if scope is None else self.root / scope
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path


class NoOpCache(DataCache):
    """Dummy cache – performs no operations."""

    def save(self, scope: str | None, data: Mapping[str, Any]) -> Path:
        return Path()

    def load(self, scope: str | None, columns: Sequence[str] | None = None) -> None:
        return None

    def delete(self, scope: str | None) -> None:
        return None

    def list_scopes(self) -> Sequence[str]:
        return []


class ParquetCache(DataCache):
    """On-disk cache: each DataFrame stored as Parquet in per-scope folders.

    Requires *pyarrow*.
    """

    EXT = ".parquet"

    def save(self, scope: str | None, data: Mapping[str, pd.DataFrame]) -> Path:
        if pq is None:
            raise RuntimeError("pyarrow is not installed – cannot use ParquetCache")
        target = self._scope_dir(scope, create=True)
        for name, df in data.items():
            path = target / f"{name}{self.EXT}"
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, path)
        return target

    def load(self, scope: str | None, columns: Sequence[str] | None = None) -> Mapping[str, pd.DataFrame] | None:
        if pq is None:
            raise RuntimeError("pyarrow is not installed – cannot use ParquetCache")
        target = self._scope_dir(scope)
        out: dict[str, pd.DataFrame] = {}
        for path in target.glob(f"*{self.EXT}"):
            name = path.stem
            table = pq.read_table(path, columns=columns if columns else None)
            out[name] = table.to_pandas()
        return out or None


class CSVCache(DataCache):
    """On-disk cache: each DataFrame stored as a CSV in per-scope folders."""

    EXT = ".csv"

    def save(self, scope: str | None, data: Mapping[str, pd.DataFrame]) -> Path:
        target = self._scope_dir(scope, create=True)
        opts = pcsv.WriteOptions(include_header=True)
        for name, df in data.items():
            path = target / f"{name}{self.EXT}"
            table = pa.Table.from_pandas(df, preserve_index=False)
            pcsv.write_csv(table, path, write_options=opts)
        return target

    def load(self, scope: str | None, columns: Sequence[str] | None = None) -> Mapping[str, pd.DataFrame] | None:
        target = self._scope_dir(scope)
        out: dict[str, pd.DataFrame] = {}
        for path in target.glob(f"*{self.EXT}"):
            name = path.stem
            read_opts = pcsv.ReadOptions(use_threads=True)
            parse_opts = pcsv.ParseOptions()
            convert_opts = pcsv.ConvertOptions(column_types=None, include_columns=columns or None)
            table = pcsv.read_csv(
                str(path), read_options=read_opts, parse_options=parse_opts, convert_options=convert_opts
            )
            out[name] = table.to_pandas()
        return out or None
