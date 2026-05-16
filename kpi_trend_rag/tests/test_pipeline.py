"""
tests/test_pipeline.py
----------------------
Unit and integration tests for the KPI RAG pipeline.
Run with:  pytest tests/ -v --cov=src
"""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ingestion.loader import load_file, _wide_to_long, _detect_delimiter
from src.ingestion.preprocessor import preprocess, _flag_anomalies, _build_context_strings
from src.llm.forecaster import forecast_kpi, _linear_fit, _ets_fit, _compute_metrics


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_long_csv(tmp_path: Path) -> Path:
    """Long-format CSV (date | kpi_name | dimension | value | unit)."""
    content = (
        "date,kpi_name,dimension,value,unit\n"
        "2023-01-02,Revenue,EMEA,120000,USD\n"
        "2023-01-09,Revenue,EMEA,130000,USD\n"
        "2023-01-16,Revenue,EMEA,125000,USD\n"
        "2023-01-23,Revenue,EMEA,135000,USD\n"
        "2023-01-30,Revenue,EMEA,140000,USD\n"
        "2023-02-06,Revenue,EMEA,145000,USD\n"
        "2023-02-13,Revenue,EMEA,150000,USD\n"
        "2023-02-20,Revenue,EMEA,155000,USD\n"
        "2023-02-27,Revenue,EMEA,160000,USD\n"
        "2023-03-06,Revenue,EMEA,165000,USD\n"
        "2023-03-13,Revenue,EMEA,170000,USD\n"
        "2023-03-20,Revenue,EMEA,175000,USD\n"
    )
    p = tmp_path / "long.csv"
    p.write_text(content)
    return p


@pytest.fixture()
def sample_wide_csv(tmp_path: Path) -> Path:
    """Wide-format CSV (date | KPI1 | KPI2 ...)."""
    content = (
        "date,Revenue,Churn Rate,NPS\n"
        "2023-01-02,120000,5.1,42\n"
        "2023-01-09,130000,5.3,41\n"
        "2023-01-16,125000,4.9,43\n"
        "2023-01-23,135000,5.0,44\n"
        "2023-01-30,140000,4.8,45\n"
        "2023-02-06,145000,4.7,46\n"
        "2023-02-13,150000,5.2,44\n"
        "2023-02-20,155000,5.1,45\n"
        "2023-02-27,160000,4.9,47\n"
        "2023-03-06,165000,4.6,48\n"
        "2023-03-13,170000,4.5,49\n"
        "2023-03-20,175000,4.4,50\n"
    )
    p = tmp_path / "wide.csv"
    p.write_text(content)
    return p


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Minimal preprocessed DataFrame for downstream tests."""
    rng = np.random.default_rng(0)
    n = 52
    dates = pd.date_range("2022-01-03", periods=n, freq="W")
    vals = 100_000 + np.linspace(0, 20_000, n) + rng.normal(0, 1_000, n)
    df = pd.DataFrame({
        "date":       dates,
        "kpi_name":   "Revenue",
        "dimension":  "Overall",
        "value":      vals,
        "unit":       "USD",
        "source_file": "test",
    })
    return preprocess(df, freq="W")


# ---------------------------------------------------------------------------
# Ingestion tests
# ---------------------------------------------------------------------------

class TestLoader:
    def test_load_long_csv(self, sample_long_csv: Path):
        df = load_file(sample_long_csv)
        assert not df.empty
        assert "kpi_name" in df.columns
        assert "value" in df.columns
        assert "date" in df.columns
        assert df["kpi_name"].iloc[0] == "Revenue"

    def test_load_wide_csv(self, sample_wide_csv: Path):
        df = load_file(sample_wide_csv)
        assert not df.empty
        kpis = set(df["kpi_name"].unique())
        assert "Revenue" in kpis
        assert "Churn Rate" in kpis
        assert "NPS" in kpis

    def test_delimiter_detection(self):
        assert _detect_delimiter("a,b,c\n1,2,3") == ","
        assert _detect_delimiter("a;b;c\n1;2;3") == ";"
        assert _detect_delimiter("a\tb\tc\n1\t2\t3") == "\t"

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_file(tmp_path / "nonexistent.csv")

    def test_values_are_numeric(self, sample_long_csv: Path):
        df = load_file(sample_long_csv)
        assert pd.api.types.is_numeric_dtype(df["value"])

    def test_dates_are_datetime(self, sample_long_csv: Path):
        df = load_file(sample_long_csv)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])

    def test_no_duplicate_hashes(self, sample_long_csv: Path):
        df = load_file(sample_long_csv)
        assert df["row_hash"].nunique() == len(df)

    def test_excel_load(self, tmp_path: Path):
        """Create a minimal xlsx and confirm it loads."""
        xlsx_path = tmp_path / "test.xlsx"
        pd.DataFrame({
            "date":     ["2023-01-02", "2023-01-09"],
            "kpi_name": ["Revenue", "Revenue"],
            "value":    [100, 110],
        }).to_excel(xlsx_path, index=False)
        df = load_file(xlsx_path)
        assert len(df) == 2


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_output_columns(self, sample_df: pd.DataFrame):
        expected = {"date", "kpi_name", "dimension", "value", "pop_pct", "anomaly", "context"}
        assert expected.issubset(set(sample_df.columns))

    def test_rolling_means_present(self, sample_df: pd.DataFrame):
        assert "roll_mean_4p" in sample_df.columns
        assert "roll_mean_12p" in sample_df.columns

    def test_context_strings_non_empty(self, sample_df: pd.DataFrame):
        assert sample_df["context"].str.len().min() > 20

    def test_anomaly_detection_flags_injected_outlier(self):
        n = 40
        dates = pd.date_range("2022-01-01", periods=n, freq="W")
        vals  = [100.0] * n
        vals[20] = 99_999.0   # extreme outlier
        df = pd.DataFrame({
            "date":       dates,
            "kpi_name":   "Test KPI",
            "dimension":  "Overall",
            "value":      vals,
            "unit":       "",
            "source_file": "test",
        })
        result = preprocess(df)
        assert result["anomaly"].any(), "Expected at least one anomaly to be flagged"

    def test_no_nan_in_value_after_preprocess(self, sample_df: pd.DataFrame):
        assert sample_df["value"].isna().sum() == 0


# ---------------------------------------------------------------------------
# Forecasting tests
# ---------------------------------------------------------------------------

class TestForecaster:
    def test_linear_forecast_shape(self, sample_df: pd.DataFrame):
        result = forecast_kpi(sample_df, "Revenue", "Overall", horizon=8, test_periods=4)
        assert len(result.forecast) == 8
        assert "predicted" in result.forecast.columns

    def test_forecast_metrics_present(self, sample_df: pd.DataFrame):
        result = forecast_kpi(sample_df, "Revenue", "Overall", horizon=4, test_periods=4)
        assert "mape" in result.metrics
        assert result.metrics["mape"] >= 0

    def test_forecast_mape_reasonable(self, sample_df: pd.DataFrame):
        """MAPE should be under 50% on synthetic linear data."""
        result = forecast_kpi(sample_df, "Revenue", "Overall", horizon=4, test_periods=4)
        assert result.metrics["mape"] < 50.0

    def test_insufficient_data_raises(self):
        df = pd.DataFrame({
            "date":       pd.date_range("2022-01-01", periods=5, freq="W"),
            "kpi_name":   "X",
            "dimension":  "Y",
            "value":      [1, 2, 3, 4, 5],
            "unit":       "",
            "source_file": "test",
        })
        df = preprocess(df)
        with pytest.raises(ValueError, match="Insufficient data"):
            forecast_kpi(df, "X", "Y", horizon=4, test_periods=4)

    def test_linear_fit_returns_correct_length(self):
        dates = pd.date_range("2022-01-01", periods=20, freq="W")
        series = pd.DataFrame({"date": dates, "value": np.linspace(1, 20, 20)})
        out = _linear_fit(series, horizon=5)
        assert len(out) == 5

    def test_compute_metrics_perfect_forecast(self):
        a = np.array([100.0, 200.0, 300.0])
        m = _compute_metrics(a, a)
        assert m["mape"] == 0.0
        assert m["rmse"] == 0.0

    def test_compute_metrics_empty_arrays(self):
        m = _compute_metrics(np.array([]), np.array([]))
        assert m["mape"] == 999.0


# ---------------------------------------------------------------------------
# ChromaDB (mocked)
# ---------------------------------------------------------------------------

class TestVectorStore:
    def test_store_ingest_and_query(self, sample_df: pd.DataFrame, tmp_path: Path):
        from src.vectorstore.chroma_store import KPIVectorStore

        store = KPIVectorStore(
            collection_name="test_col",
            persist_dir=tmp_path / "chroma",
            embed_model="all-MiniLM-L6-v2",
        )
        store.ingest(sample_df)
        assert store.stats()["total_documents"] > 0

        results = store.query("revenue trend", n_results=3)
        assert isinstance(results, list)
        assert len(results) <= 3

    def test_store_list_kpis(self, sample_df: pd.DataFrame, tmp_path: Path):
        from src.vectorstore.chroma_store import KPIVectorStore

        store = KPIVectorStore(
            collection_name="test_kpi_list",
            persist_dir=tmp_path / "chroma2",
        )
        store.ingest(sample_df)
        kpis = store.list_kpis()
        assert "Revenue" in kpis

    def test_store_reset(self, sample_df: pd.DataFrame, tmp_path: Path):
        from src.vectorstore.chroma_store import KPIVectorStore

        store = KPIVectorStore(
            collection_name="test_reset",
            persist_dir=tmp_path / "chroma3",
        )
        store.ingest(sample_df)
        assert store.stats()["total_documents"] > 0
        store.reset()
        assert store.stats()["total_documents"] == 0
