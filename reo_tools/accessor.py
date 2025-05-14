# accessor.py
"""Result accessor for pipeline outputs.

Provides convenient read-only access to cached or freshly computed DataFrames:
  - list available datasets
  - retrieve column names (headers)
  - fetch a full DataFrame or a single column
"""

from collections.abc import Mapping

import pandas as pd


class ResultAccessor:
    """Thin wrapper over a dict of DataFrames."""

    def __init__(self, datasets: Mapping[str, pd.DataFrame]) -> None:
        self._datasets = datasets

    def datasets(self) -> list[str]:
        """Return list of dataset names."""
        return list(self._datasets.keys())

    def headers(self) -> dict[str, list[str]]:
        """Return dict mapping dataset names to list of column headers."""
        return {name: df.columns.tolist() for name, df in self._datasets.items()}

    def all(self, dataset: str) -> pd.DataFrame:
        """Return the full DataFrame for the given dataset."""
        try:
            return self._datasets[dataset]
        except KeyError:
            raise KeyError(f"Dataset '{dataset}' not found")

    def column(self, dataset: str, column: str) -> pd.Series:
        """Return a single column from the given dataset."""
        try:
            return self._datasets[dataset][column]
        except KeyError:
            raise KeyError(f"Dataset '{dataset}' or column '{column}' not found")
