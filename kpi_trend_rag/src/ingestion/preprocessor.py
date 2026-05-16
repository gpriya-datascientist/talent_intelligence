"""
preprocessor.py
---------------
Clean, enrich, and feature-engineer normalised KPI DataFrames.

Pipeline
--------
1. Drop obvious data quality issues (nulls, outliers)
2. Compute period-over-period deltas and growth rates
3. Add rolling statistics (7d, 30d, 90d)
4. Flag anomalies (IQR + z-score ensemble)
5. Attach rich text context strings used for RAG embedding
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df   : Normalised KPI DataFrame from loader.load_file
    freq : Resampling frequency ('D'=daily, 'W'=weekly, 'ME'=month-end)

    Returns
    -------
    pd.DataFrame with additional enrichment columns.
    """
    df = _clean(df)
    df = _resample(df, freq)
    df = _add_growth(df)
    df = _add_rolling(df)
    df = _flag_anomalies(df)
    df = _build_context_strings(df)
    logger.success(f"Preprocessing complete: {len(df):,} rows, {df['kpi_name'].nunique()} KPIs")
    return df


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing critical fields and obvious entry errors."""
    before = len(df)
    df = df.dropna(subset=["date", "value", "kpi_name"])
    df = df[df["value"].apply(lambda x: np.isfinite(x))]
    logger.debug(f"Clean: {before - len(df)} rows removed, {len(df)} remain")
    return df.copy()


def _resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Aggregate to uniform time frequency per (kpi_name, dimension).
    Uses mean for numeric values to handle multiple readings per period.
    """
    df["date"] = pd.to_datetime(df["date"])
    groups = []
    for (kpi, dim), grp in df.groupby(["kpi_name", "dimension"], sort=False):
        grp = grp.set_index("date").resample(freq)["value"].mean().reset_index()
        grp["kpi_name"]   = kpi
        grp["dimension"]  = dim
        grp["unit"]       = df.loc[
            (df["kpi_name"] == kpi) & (df["dimension"] == dim), "unit"
        ].iloc[0] if "unit" in df.columns else ""
        groups.append(grp)

    resampled = pd.concat(groups, ignore_index=True)
    resampled = resampled.dropna(subset=["value"])
    resampled = resampled.sort_values(["kpi_name", "dimension", "date"]).reset_index(drop=True)
    return resampled


def _add_growth(df: pd.DataFrame) -> pd.DataFrame:
    """Add period-over-period and year-over-year growth columns."""
    df = df.copy()
    df["pop_delta"]  = df.groupby(["kpi_name", "dimension"])["value"].diff()
    df["pop_pct"]    = df.groupby(["kpi_name", "dimension"])["value"].pct_change() * 100
    df["yoy_delta"]  = df.groupby(["kpi_name", "dimension"])["value"].diff(52)  # 52 weeks
    df["yoy_pct"]    = (
        df.groupby(["kpi_name", "dimension"])["value"]
        .pct_change(52) * 100
    )
    return df


def _add_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling mean and std for 4, 12, 26 periods."""
    df = df.copy()
    for window, label in [(4, "4p"), (12, "12p"), (26, "26p")]:
        df[f"roll_mean_{label}"] = (
            df.groupby(["kpi_name", "dimension"])["value"]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )
        df[f"roll_std_{label}"] = (
            df.groupby(["kpi_name", "dimension"])["value"]
            .transform(lambda s: s.rolling(window, min_periods=2).std())
        )
    return df


def _flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensemble anomaly detection:
      - IQR method (flag if outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR])
      - z-score   (flag if |z| > 3)
    A row is anomalous if BOTH methods flag it.
    """
    df = df.copy()
    df["anomaly"] = False
    df["z_score"]  = np.nan

    for (kpi, dim), grp in df.groupby(["kpi_name", "dimension"]):
        idx = grp.index
        vals = grp["value"]

        # IQR
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        iqr_flag = (vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)

        # z-score
        mu, sigma = vals.mean(), vals.std()
        if sigma > 0:
            z = (vals - mu) / sigma
            df.loc[idx, "z_score"] = z
            z_flag = z.abs() > 3
        else:
            z_flag = pd.Series(False, index=idx)

        df.loc[idx, "anomaly"] = iqr_flag & z_flag

    n_anomalies = df["anomaly"].sum()
    if n_anomalies:
        logger.warning(f"Anomaly detection: {n_anomalies} anomalous data points flagged")
    return df


def _build_context_strings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rich natural-language context strings for each row.
    These are embedded into ChromaDB for RAG retrieval.
    """
    df = df.copy()
    df["context"] = df.apply(_row_to_context, axis=1)
    return df


def _row_to_context(row: pd.Series) -> str:
    """Render a single row as a descriptive sentence for embedding."""
    date_str  = pd.Timestamp(row["date"]).strftime("%B %d, %Y")
    kpi       = row["kpi_name"]
    dim       = row["dimension"]
    val       = _fmt(row["value"])
    unit      = f" {row['unit']}" if row.get("unit") else ""
    anomaly   = " ⚠ Anomaly detected." if row.get("anomaly") else ""

    pop_str = ""
    if pd.notna(row.get("pop_pct")):
        direction = "up" if row["pop_pct"] > 0 else "down"
        pop_str = f" This is {direction} {abs(row['pop_pct']):.1f}% from the prior period."

    yoy_str = ""
    if pd.notna(row.get("yoy_pct")):
        direction = "up" if row["yoy_pct"] > 0 else "down"
        yoy_str = f" Year-over-year it is {direction} {abs(row['yoy_pct']):.1f}%."

    roll_str = ""
    if pd.notna(row.get("roll_mean_12p")):
        roll_str = f" The 12-period rolling average is {_fmt(row['roll_mean_12p'])}{unit}."

    return (
        f"On {date_str}, {kpi} ({dim}) recorded a value of {val}{unit}.{pop_str}"
        f"{yoy_str}{roll_str}{anomaly}"
    )


def _fmt(value: float) -> str:
    """Format a numeric value for human-readable output."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.2f}"
