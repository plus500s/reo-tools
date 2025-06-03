from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseReader(ABC):
    """
    Abstract reader enforcing the RVG pipeline schema.
    Subclasses must implement _load_raw() to load raw data.
    """

    REQUIRED_COLUMNS: tuple[str, str, str] = ("t", "U1", "U2")

    def read(self, file_path: str | Path) -> pd.DataFrame:
        """Load raw data, validate schema, and post-process."""
        path = Path(file_path).expanduser().resolve(strict=True)

        # Load raw data
        df = self._load_raw(path)
        # Ensure DataFrame is valid
        self._validate_dataframe(df)
        # Keep only required columns in order
        df = self._normalize_columns(df)
        # Post-process (optional hook)
        df = self._post_load(df)
        # Validate required schema
        self._ensure_types(df)
        return df

    @abstractmethod
    def _load_raw(self, path: Path) -> pd.DataFrame:
        """Load raw data from the given path. Must be implemented by subclasses."""
        ...

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Ensure the DataFrame is valid for processing."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame, got {type(df).__name__}")
        if df.empty:
            raise ValueError("DataFrame is empty")
        if missing := set(self.REQUIRED_COLUMNS) - set(df.columns):
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

    def _ensure_types(self, df: pd.DataFrame) -> None:
        """Ensure all mandatory columns are numeric."""
        kinds = df.dtypes[list(self.REQUIRED_COLUMNS)].apply(lambda dt: dt.kind)
        if (~kinds.isin(list("fi"))).any():
            bad = kinds[~kinds.isin(list("fi"))]
            raise TypeError(f"Invalid column types: {', '.join(bad.index)} ({', '.join(bad.astype(str))})")

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return view with only the required columns in canonical order."""
        return df.loc[:, list(self.REQUIRED_COLUMNS)].copy(deep=False)

    def _post_load(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optional per-subclass post-processing of the DataFrame. Default no-op."""
        return df


class CSVReader(BaseReader):
    """Reader for CSV files with specific format requirements."""

    def __init__(
        self,
        skiprows: int = 17,
        sep: str | None = "\t",
        encoding: str = "cp1251",
    ) -> None:
        self.skiprows = skiprows
        self.sep = sep
        self.encoding = encoding

    def _load_raw(self, path: Path) -> pd.DataFrame:
        """Load raw CSV data with specified separator and encoding."""
        try:
            df = pd.read_csv(
                path,
                sep=self.sep,
                skiprows=self.skiprows,
                encoding=self.encoding,
                names=list(self.REQUIRED_COLUMNS),
                usecols=list(self.REQUIRED_COLUMNS),
                on_bad_lines="error",
                engine="c",
            )

        except UnicodeDecodeError as e:
            raise ValueError(f"Encoding error: {e}") from e
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV parse error: {e}") from e
        return df

    def _post_load(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-process the DataFrame to parse time."""
        df["t"] = pd.to_timedelta(df["t"]).dt.total_seconds().astype("float64")
        return df
