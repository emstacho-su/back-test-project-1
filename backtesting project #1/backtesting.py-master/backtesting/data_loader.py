"""Utilities for loading and cleaning OHLCV data from Excel workbooks.

The helpers in this module normalize commonly encountered spreadsheet layouts
into the :class:`pandas.DataFrame` format expected by :class:`~backtesting.Backtest`.
They are intentionally forgiving about column naming and whitespace, so that
messy human-generated logs can still be transformed into clean market data.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, MutableMapping, Optional

import pandas as pd

# Canonical OHLCV column names Backtest expects
_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

# Friendly synonyms that often appear in ad-hoc Excel logs
_DEFAULT_SYNONYMS: Mapping[str, Iterable[str]] = {
    "open": ("o", "bidopen", "askopen", "open price", "opening"),
    "high": ("h", "bidhigh", "askhigh", "high price"),
    "low": ("l", "bidlow", "asklow", "low price"),
    "close": ("c", "bidclose", "askclose", "close price", "closing"),
    "volume": ("v", "vol", "tickvol", "tick volume", "ticks"),
    "datetime": ("timestamp", "datetime", "dt", "date/time"),
    "date": ("date", "day"),
    "time": ("time", "timestamp", "hour"),
}


def _normalize_columns(df: pd.DataFrame) -> MutableMapping[str, str]:
    """Return a mapping of normalized column name -> original name.

    Strips whitespace and lowercases names to make downstream matching easier.
    """

    return {str(col).strip().lower(): col for col in df.columns}


def _resolve_column(name: str, columns: Mapping[str, str], custom_map: Mapping[str, str]) -> Optional[str]:
    """Pick the best matching column for a canonical field.

    1. Honor explicit ``custom_map`` entries (case-insensitive).
    2. Look for exact normalized name matches.
    3. Fall back to synonyms.
    """

    lower_map = {k.lower(): v for k, v in custom_map.items()}
    if name in lower_map:
        return lower_map[name]

    if name in columns:
        return columns[name]

    for synonym in _DEFAULT_SYNONYMS.get(name, ()):  # type: ignore[arg-type]
        if synonym in columns:
            return columns[synonym]
    return None


def load_excel_ohlcv(
    path: str,
    *,
    sheet_name: int | str | None = 0,
    column_map: Mapping[str, str] | None = None,
    datetime_column: str | None = None,
    date_column: str | None = None,
    time_column: str | None = None,
    dropna: bool = True,
    tz: str | None = None,
    resample_rule: str | None = None,
) -> pd.DataFrame:
    """Load OHLCV data from an Excel sheet into a normalized DataFrame.

    Parameters
    ----------
    path
        Path to the Excel workbook.
    sheet_name
        Sheet to load. Defaults to the first sheet (``0``).
    column_map
        Optional explicit mapping from canonical OHLCV field names
        (``Open``, ``High``, ``Low``, ``Close``, ``Volume``, ``datetime``)
        to the column names present in the workbook. Matching is
        case-insensitive. Any keys omitted will be resolved heuristically.
    datetime_column
        Name of a combined date/time column to parse. If omitted, the loader
        will try to combine ``date_column`` and ``time_column`` instead.
    date_column, time_column
        Split date/time column names to combine into a single index when
        ``datetime_column`` is not provided.
    dropna
        When ``True`` (default), rows missing any OHLC value are removed. When
        ``False``, partial rows are forward-filled where possible.
    tz
        Optional timezone to localize the resulting ``DatetimeIndex``.
    resample_rule
        Optional pandas offset alias (e.g. ``"5T"``) to resample aggregated
        bars after loading. The sampler uses OHLC aggregation and sums volume.

    Returns
    -------
    pandas.DataFrame
        Cleaned OHLCV data indexed by ``DatetimeIndex`` and sorted ascending.
    """

    df = pd.read_excel(path, sheet_name=sheet_name)
    normalized = _normalize_columns(df)
    column_map = {k.lower(): v for k, v in (column_map or {}).items()}

    # Identify datetime column(s)
    date_col = date_column or _resolve_column("date", normalized, column_map)
    time_col = time_column or _resolve_column("time", normalized, column_map)
    dt_col = datetime_column or (None if (date_col and time_col) else _resolve_column("datetime", normalized, column_map))

    if dt_col:
        dt_values = pd.to_datetime(df[dt_col], errors="coerce")
    elif date_col and time_col:
        dt_values = pd.to_datetime(
            df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
            errors="coerce",
        )
    else:
        raise ValueError(
            "Could not determine datetime information. Provide 'datetime_column' or both 'date_column' and 'time_column'."
        )

    ohlcv: Dict[str, pd.Series] = {}
    for field in _OHLCV_COLUMNS:
        col = _resolve_column(field.lower(), normalized, column_map)
        if col is None:
            raise ValueError(f"Required column for '{field}' not found in Excel file '{path}'.")
        series = pd.to_numeric(df[col], errors="coerce") if field != "Volume" else pd.to_numeric(df[col], errors="coerce")
        ohlcv[field] = series

    cleaned = pd.DataFrame(ohlcv)
    cleaned.insert(0, "Datetime", dt_values)

    if dropna:
        cleaned = cleaned.dropna(subset=["Datetime", * _OHLCV_COLUMNS])
    else:
        cleaned[[* _OHLCV_COLUMNS]] = cleaned[[* _OHLCV_COLUMNS]].ffill()
        cleaned = cleaned.dropna(subset=["Datetime"])

    cleaned = cleaned.sort_values("Datetime")
    cleaned = cleaned.drop_duplicates(subset=["Datetime"], keep="last")

    idx = pd.DatetimeIndex(cleaned["Datetime"], tz=tz)
    cleaned = cleaned.drop(columns=["Datetime"])
    cleaned.index = idx

    if resample_rule:
        normalized_rule = re.sub(r"(?<=\d)T\b", "min", resample_rule)
        cleaned = cleaned.resample(normalized_rule, label="left", closed="left").agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna(subset=["Open", "High", "Low", "Close"])

    return cleaned
