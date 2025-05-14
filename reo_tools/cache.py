"""Caching back-ends for RVG pipeline.

Expose three classes:
    DataCache    – abstract interface (save/load/delete/list_scopes).
    ParquetCache – columnar on-disk cache using Apache Parquet.
    NoOpCache    – dummy cache that always recomputes.

Key design points
-----------------
* **Scope** – optional sub-folder isolating different data sources.
  If *None*, files go directly into `<project>/.signal_pipeline_cache/`.
* **Cache key** – dataset name; each DataFrame is stored as `<name>.parquet` in the scope folder.
* **Column-selective load** – `ParquetCache.load(scope, columns=...)` reads only requested columns.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    warnings.warn(
        "pyarrow is not installed – ParquetCache unavailable",
        ImportWarning,
        stacklevel=2,
    )
    pa = pq = None  # type: ignore

__all__ = [
    "DataCache",
    "ParquetCache",
    "NoOpCache",
    "CSVCache",
]


class DataCache(ABC):
    """Abstract cache interface."""

    def __init__(self, cache_root: Path | str | None = None) -> None:
        # default to .pipeline_cache in project root
        self.root = Path(cache_root).expanduser().resolve() if cache_root else Path.cwd() / ".pipeline_cache"
        self.root.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def save(self, scope: str | None, data: Mapping[str, pd.DataFrame]) -> None:
        """Save a mapping of name→DataFrame under the given scope."""
        ...

    @abstractmethod
    def load(self, scope: str | None, columns: list[str] | None = None) -> Mapping[str, pd.DataFrame] | None:
        """Load all cached DataFrames under the given scope."""
        ...

    @abstractmethod
    def delete(self, scope: str | None) -> None:
        """Delete all cached files under the given scope."""
        ...

    @abstractmethod
    def list_scopes(self) -> Sequence[str]:
        """List all available scopes (subdirectories)."""
        ...


class NoOpCache(DataCache):
    """Dummy cache – performs no operations."""

    def save(self, scope: str | None, data: Mapping[str, Any]) -> None:
        return None

    def load(self, scope: str | None, columns: list[str] | None = None) -> None:  # type: ignore[override]
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

    def save(self, scope: str | None, data: Mapping[str, pd.DataFrame]) -> None:
        if pq is None:
            raise RuntimeError("pyarrow is not installed – cannot use ParquetCache")
        target = self.root / scope if scope else self.root
        target.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            path = target / f"{name}{self.EXT}"
            table = pa.Table.from_pandas(df)
            pq.write_table(table, path)

    def load(self, scope: str | None, columns: list[str] | None = None) -> Mapping[str, pd.DataFrame] | None:
        if pq is None:
            raise RuntimeError("pyarrow is not installed – cannot use ParquetCache")
        target = self.root / scope if scope else self.root
        if not target.exists():
            return None
        out: dict[str, pd.DataFrame] = {}
        for path in target.glob(f"*{self.EXT}"):
            name = path.stem
            table = pq.read_table(path, columns=columns)
            out[name] = table.to_pandas()
        return out or None

    def delete(self, scope: str | None) -> None:
        target = self.root / scope if scope else self.root
        if not target.exists():
            return
        for path in target.glob(f"*{self.EXT}"):
            path.unlink(missing_ok=True)

    def list_scopes(self) -> Sequence[str]:
        # scopes are subdirectories under root
        return [p.name for p in self.root.iterdir() if p.is_dir()]


class CSVCache(DataCache):
    """On-disk cache: each DataFrame stored as a CSV in per-scope folders."""

    EXT = ".csv"

    def save(self, scope: str | None, data: Mapping[str, pd.DataFrame]) -> None:
        target = (self.root / scope) if scope else self.root
        target.mkdir(parents=True, exist_ok=True)
        for name, df in data.items():
            path = target / f"{name}{self.EXT}"
            df.to_csv(path, index=False)

    def load(self, scope: str | None, columns: list[str] | None = None) -> Mapping[str, pd.DataFrame] | None:
        target = (self.root / scope) if scope else self.root
        if not target.exists():
            return None
        out: dict[str, pd.DataFrame] = {}
        for path in target.glob(f"*{self.EXT}"):
            name = path.stem
            # read only requested columns (including 't'!)
            df = pd.read_csv(path, usecols=columns if columns else None)
            out[name] = df
        return out or None

    def delete(self, scope: str | None) -> None:
        target = (self.root / scope) if scope else self.root
        if not target.exists():
            return
        for path in target.glob(f"*{self.EXT}"):
            path.unlink(missing_ok=True)

    def list_scopes(self) -> Sequence[str]:
        return [p.name for p in self.root.iterdir() if p.is_dir()]
