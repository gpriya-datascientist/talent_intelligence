"""
demo_run.py
-----------
End-to-end demo — runs both scenarios directly (no API needed).

Usage:
    python scripts/demo_run.py
"""
from graph.retraining_graph import retraining_graph
from loguru import logger
import json

BASE_STATE = {
    "drift_detected": False,
    "drift_score":    0.0,
    "drift_details":  {},
    "eval_passed":    False,
    "deployed":       False,
}

# ── Scenario A: no drift ──────────────────────────────────────
print("\n" + "="*60)
print("SCENARIO A — No drift (expect early exit after Monitor)")
print("="*60)

result_a = retraining_graph.invoke({
    **BASE_STATE,
    "run_id":                "demo-nodrift-001",
    "production_data_path":  "data/sample/production_nodrift.parquet",
    "reference_data_path":   "data/sample/reference.parquet",
    "config":                {"drift_threshold": 0.2, "min_accuracy_delta": 0.01},
})
print(f"drift_detected : {result_a['drift_detected']}")
print(f"drift_score    : {result_a['drift_score']}")
print(f"deployed       : {result_a['deployed']}")

# ── Scenario B: drift detected ────────────────────────────────
print("\n" + "="*60)
print("SCENARIO B — Drift detected (full pipeline runs)")
print("="*60)

result_b = retraining_graph.invoke({
    **BASE_STATE,
    "run_id":                "demo-drift-001",
    "production_data_path":  "data/sample/production_drift.parquet",
    "reference_data_path":   "data/sample/reference.parquet",
    "config":                {"drift_threshold": 0.2, "min_accuracy_delta": 0.01},
})
print(f"drift_detected     : {result_b['drift_detected']}")
print(f"drift_score        : {result_b['drift_score']}")
print(f"baseline_accuracy  : {result_b.get('baseline_accuracy')}")
print(f"new_accuracy       : {result_b.get('new_accuracy')}")
print(f"eval_delta         : {result_b.get('eval_delta')}")
print(f"approved           : {result_b.get('approved')}")
print(f"deployed           : {result_b['deployed']}")
