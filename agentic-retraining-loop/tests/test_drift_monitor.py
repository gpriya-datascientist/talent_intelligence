import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from agents.drift_monitor import drift_monitor_node


@pytest.fixture
def sample_paths(tmp_path):
    rng = np.random.default_rng(0)
    cols = [f"feature_{i}" for i in range(1, 6)]

    ref = pd.DataFrame(rng.normal(0, 1, (500, 5)), columns=cols)
    ref["label"] = 0
    ref_path = tmp_path / "ref.parquet"
    ref.to_parquet(ref_path)

    prod_ok = pd.DataFrame(rng.normal(0, 1, (500, 5)), columns=cols)
    prod_ok["label"] = 0
    prod_ok_path = tmp_path / "prod_ok.parquet"
    prod_ok.to_parquet(prod_ok_path)

    prod_drift = pd.DataFrame(rng.normal(5, 1, (500, 5)), columns=cols)
    prod_drift["label"] = 1
    prod_drift_path = tmp_path / "prod_drift.parquet"
    prod_drift.to_parquet(prod_drift_path)

    return str(ref_path), str(prod_ok_path), str(prod_drift_path)


def _base_state(run_id, prod_path, ref_path):
    return {
        "run_id": run_id,
        "production_data_path": prod_path,
        "reference_data_path": ref_path,
        "config": {"drift_threshold": 0.2},
        "drift_detected": False, "drift_score": 0.0, "drift_details": {},
        "eval_passed": False, "deployed": False,
    }


def test_no_drift(sample_paths):
    ref, prod_ok, _ = sample_paths
    result = drift_monitor_node(_base_state("test-nodrift", prod_ok, ref))
    assert result["drift_detected"] is False


def test_drift_detected(sample_paths):
    ref, _, prod_drift = sample_paths
    result = drift_monitor_node(_base_state("test-drift", prod_drift, ref))
    assert result["drift_detected"] is True
    assert result["drift_score"] > 0.2
