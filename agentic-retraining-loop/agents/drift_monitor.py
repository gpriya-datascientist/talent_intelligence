"""
DriftMonitor Agent
------------------
Computes PSI (Population Stability Index) and KS-test on each feature
between reference and production distributions.
Flags drift if any feature PSI > threshold.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger

from graph.state import RetrainingState
from config.settings import settings


def _psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index between two arrays."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_perc = np.histogram(expected, bins=np.percentile(expected, breakpoints))[0] / len(expected)
    actual_perc   = np.histogram(actual,   bins=np.percentile(expected, breakpoints))[0] / len(actual)
    # Avoid log(0)
    expected_perc = np.where(expected_perc == 0, 1e-6, expected_perc)
    actual_perc   = np.where(actual_perc   == 0, 1e-6, actual_perc)
    return float(np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc)))


def drift_monitor_node(state: RetrainingState) -> RetrainingState:
    logger.info(f"[DriftMonitor] run_id={state['run_id']} — starting drift check")

    try:
        ref  = pd.read_parquet(state["reference_data_path"])
        prod = pd.read_parquet(state["production_data_path"])

        feature_cols = [c for c in ref.columns if c.startswith("feature_")]
        drift_details: dict[str, float] = {}

        for col in feature_cols:
            psi_score = _psi(ref[col].values, prod[col].values)
            ks_stat, _ = stats.ks_2samp(ref[col].values, prod[col].values)
            drift_details[col] = round(max(psi_score, ks_stat), 4)
            logger.debug(f"  {col}: PSI={psi_score:.4f}  KS={ks_stat:.4f}")

        max_drift = max(drift_details.values())
        drift_detected = max_drift > settings.drift_threshold

        logger.info(f"[DriftMonitor] max_drift={max_drift:.4f}  detected={drift_detected}")

        return {
            **state,
            "drift_score":    max_drift,
            "drift_details":  drift_details,
            "drift_detected": drift_detected,
        }

    except Exception as e:
        logger.error(f"[DriftMonitor] error: {e}")
        return {**state, "error": str(e), "drift_detected": False, "drift_score": 0.0, "drift_details": {}}
