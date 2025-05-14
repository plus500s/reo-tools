# reader.py
"""Reader abstractions for RVG pipeline.

Defines:
  - BaseReader: template for reading, mapping, and validating raw data schema.
  - CsvReader: concrete implementation for CSV files with autodetect and flexible mapping,
    including time parsing after mapping.

Usage:
    reader = CsvReader(
        column_map={"Time(s)": "t", "Signal1": "U1", "Signal2": "U2"},
        skiprows=17,
        sep='\t',
        encoding='cp1251',
        auto_sep=True,
        has_header=True,
    )
    df = reader.read("data.csv")  # DataFrame with columns t, U1, U2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pandas as pd


class BaseReader(ABC):
    """
    Abstract reader enforcing the RVG pipeline schema.
    Subclasses must implement _load_raw() to load raw data.
    """

    REQUIRED_COLUMNS = ("t", "U1", "U2")

    def __init__(
        self,
        column_map: Mapping[str | int, str] | None = None,
        has_header: bool = True,
    ) -> None:
        # mapping: source header name or column index -> canonical name
        self.column_map = column_map or {}
        self.has_header = has_header

    def read(self, file_path: str | Path) -> pd.DataFrame:
        """Load raw data, apply mapping, validate schema, and post-process."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        # Load raw data
        df = self._load_raw(path)
        # Apply column mapping and headerless handling
        df = self._apply_mapping(df)
        # Validate required schema
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        # Keep only required columns in order
        df = df.loc[:, list(self.REQUIRED_COLUMNS)]
        # Post-process (optional hook)
        return self._post_process(df)

    @abstractmethod
    def _load_raw(self, path: Path) -> pd.DataFrame:
        """Load raw DataFrame from path. May contain extra columns."""
        ...

    def _apply_mapping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns based on column_map and assign canonical names if headerless."""
        rename_map: dict[str, str] = {}
        for src, tgt in self.column_map.items():
            if isinstance(src, int):  # positional mapping
                if src < 0 or src >= len(df.columns):
                    raise ValueError(f"Column index out of range: {src}")
                rename_map[df.columns[src]] = tgt
            else:  # name mapping, case-insensitive
                matches = [col for col in df.columns if col.lower() == str(src).lower()]
                if not matches:
                    raise ValueError(f"Column not found for mapping: {src}")
                rename_map[matches[0]] = tgt
        if rename_map:
            df = df.rename(columns=rename_map)

        # headerless mode: assign canonical names if no header and no mapping
        if not self.has_header and not rename_map:
            if len(df.columns) < len(self.REQUIRED_COLUMNS):
                raise ValueError("Insufficient columns for headerless mapping")
            df.columns = list(self.REQUIRED_COLUMNS) + list(df.columns[len(self.REQUIRED_COLUMNS) :])
        return df

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional per-subclass post-processing of the DataFrame. Default no-op."""
        return df


class CSVReader(BaseReader):
    """CSV reader with autodetect separator, mapping, and time parsing."""

    def __init__(
        self,
        column_map: Mapping[str | int, str] | None = None,
        skiprows: int = 17,
        sep: str = "\t",
        encoding: str = "cp1251",
        auto_sep: bool = False,
        has_header: bool = False,
    ) -> None:
        super().__init__(column_map=column_map, has_header=has_header)
        self.skiprows = skiprows
        self.sep = sep
        self.encoding = encoding
        self.auto_sep = auto_sep

    def _load_raw(self, path: Path) -> pd.DataFrame:
        # Determine separator
        sep = self.sep
        if self.auto_sep:
            sample = path.read_bytes()[:1024].decode(self.encoding, errors="ignore")
            for candidate in (",", ";", "\t", "|"):
                if candidate in sample:
                    sep = candidate
                    break
        header = 0 if self.has_header else None
        try:
            df = pd.read_csv(
                path, skiprows=self.skiprows, sep=sep, encoding=self.encoding, header=header, on_bad_lines="error"
            )
        except UnicodeDecodeError as e:
            raise ValueError(f"Encoding error: {e}")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV parse error: {e}")
        return df

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        # Parse time column to float seconds
        df["t"] = df["t"].apply(self._parse_time)
        return df

    @staticmethod
    def _parse_time(value: str) -> float:
        """Parse 'HH:MM:SS.sss' into seconds."""
        try:
            dt = datetime.strptime(value.strip(), "%H:%M:%S.%f")
        except Exception:
            raise ValueError(f"Invalid time value: '{value}'")
        return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
