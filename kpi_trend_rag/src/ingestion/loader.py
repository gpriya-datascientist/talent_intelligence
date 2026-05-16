"""
loader.py
---------
Ingest CSV files and Power BI exports (.xlsx, .pbix-derived tables).
Produces a normalised DataFrame with columns:
  date | kpi_name | dimension | value | unit | source_file
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(path: str | Path) -> pd.DataFrame:
    """
    Auto-detect file format and return a normalised KPI DataFrame.

    Supported formats
    -----------------
    .csv        Plain comma-/semicolon-/tab-separated values
    .xlsx       Excel workbooks (Power BI table exports included)
    .xls        Legacy Excel
    .tsv        Tab-separated values

    Parameters
    ----------
    path : str | Path
        Absolute or relative path to the source file.

    Returns
    -------
    pd.DataFrame
        Normalised frame with schema defined at module level.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")

    suffix = path.suffix.lower()
    logger.info(f"Loading {suffix} file: {path.name}")

    if suffix in {".csv", ".tsv"}:
        raw = _read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        raw = _read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    normalised = _normalise(raw, source=path.name)
    logger.success(
        f"Loaded {len(normalised):,} rows | "
        f"{normalised['kpi_name'].nunique()} KPIs | "
        f"date range {normalised['date'].min().date()} → {normalised['date'].max().date()}"
    )
    return normalised


def load_directory(directory: str | Path, pattern: str = "*") -> pd.DataFrame:
    """
    Load all supported files from a directory and concatenate.

    Parameters
    ----------
    directory : str | Path
    pattern   : glob pattern, default "*" (all supported types)
    """
    directory = Path(directory)
    frames: list[pd.DataFrame] = []

    for ext in ("*.csv", "*.tsv", "*.xlsx", "*.xls"):
        for f in sorted(directory.glob(ext)):
            try:
                frames.append(load_file(f))
            except Exception as exc:
                logger.warning(f"Skipping {f.name}: {exc}")

    if not frames:
        raise RuntimeError(f"No supported files found in {directory}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "kpi_name", "dimension", "source_file"])
    logger.info(f"Combined {len(frames)} files → {len(combined):,} total rows")
    return combined


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> pd.DataFrame:
    """Detect delimiter and read CSV/TSV."""
    sample = path.read_text(encoding="utf-8", errors="replace")[:4096]
    delimiter = _detect_delimiter(sample)
    df = pd.read_csv(path, delimiter=delimiter, encoding="utf-8", errors="replace")
    logger.debug(f"CSV read: {len(df)} rows, delimiter='{delimiter}'")
    return df


def _detect_delimiter(sample: str) -> str:
    counts = {d: sample.count(d) for d in (",", ";", "\t", "|")}
    return max(counts, key=counts.get)


def _read_excel(path: Path) -> pd.DataFrame:
    """
    Read first non-empty sheet from Excel.
    Power BI exports commonly contain a single flat table.
    """
    xl = pd.ExcelFile(path, engine="openpyxl")
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        if not df.empty:
            logger.debug(f"Excel sheet '{sheet}': {len(df)} rows")
            return df
    raise ValueError(f"No non-empty sheets found in {path.name}")


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

_DATE_ALIASES = {"date", "period", "month", "week", "year", "timestamp", "time", "dt"}
_KPI_ALIASES  = {"kpi", "kpi_name", "metric", "metric_name", "indicator", "measure", "name"}
_VALUE_ALIASES = {"value", "actual", "amount", "measure_value", "val", "result", "score"}
_DIM_ALIASES   = {"dimension", "dim", "category", "segment", "region", "department", "group"}
_UNIT_ALIASES  = {"unit", "units", "currency", "uom"}


def _find_col(df: pd.DataFrame, aliases: set[str]) -> Optional[str]:
    """Return first column name matching any alias (case-insensitive)."""
    lower_map = {c.lower().replace(" ", "_"): c for c in df.columns}
    for alias in aliases:
        if alias in lower_map:
            return lower_map[alias]
    return None


def _normalise(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Map raw columns → canonical schema.
    Falls back to heuristic detection when column names are non-standard.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    date_col  = _find_col(df, _DATE_ALIASES)
    kpi_col   = _find_col(df, _KPI_ALIASES)
    val_col   = _find_col(df, _VALUE_ALIASES)
    dim_col   = _find_col(df, _DIM_ALIASES)
    unit_col  = _find_col(df, _UNIT_ALIASES)

    # Fallback: if no explicit KPI column, try to pivot wide→long
    if kpi_col is None and val_col is None:
        df = _wide_to_long(df, date_col)
        date_col  = "date"
        kpi_col   = "kpi_name"
        val_col   = "value"
        dim_col   = None
        unit_col  = None

    # Build output frame
    out = pd.DataFrame()
    out["date"]       = pd.to_datetime(df[date_col], errors="coerce")
    out["kpi_name"]   = df[kpi_col].astype(str).str.strip() if kpi_col else "Unknown KPI"
    out["value"]      = pd.to_numeric(df[val_col], errors="coerce") if val_col else float("nan")
    out["dimension"]  = df[dim_col].astype(str).str.strip() if dim_col else "Overall"
    out["unit"]       = df[unit_col].astype(str).str.strip() if unit_col else ""
    out["source_file"] = source
    out["row_hash"]   = _hash_rows(out)

    out = out.dropna(subset=["date", "value"])
    out = out.sort_values(["kpi_name", "dimension", "date"]).reset_index(drop=True)
    return out


def _wide_to_long(df: pd.DataFrame, date_col: Optional[str]) -> pd.DataFrame:
    """Melt a wide table (each KPI is a column) into long format."""
    if date_col is None:
        # Guess: first column that looks like dates
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().head(5))
                date_col = col
                break
            except Exception:
                continue

    if date_col is None:
        raise ValueError("Cannot detect date column for wide-format table.")

    id_vars = [date_col]
    value_vars = [c for c in df.columns if c != date_col]
    melted = df.melt(id_vars=id_vars, value_vars=value_vars,
                     var_name="kpi_name", value_name="value")
    melted = melted.rename(columns={date_col: "date"})
    return melted


def _hash_rows(df: pd.DataFrame) -> pd.Series:
    """Deterministic hash per row for deduplication."""
    combined = (
        df["date"].astype(str)
        + "|" + df["kpi_name"]
        + "|" + df["dimension"]
        + "|" + df["value"].astype(str)
    )
    return combined.apply(lambda s: hashlib.md5(s.encode()).hexdigest()[:12])
