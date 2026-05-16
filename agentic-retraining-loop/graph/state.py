from __future__ import annotations
from typing import Optional, Dict, Any
from typing_extensions import TypedDict


class RetrainingState(TypedDict):
    run_id: str
    production_data_path: str
    reference_data_path: str
    config: Dict[str, Any]

    # Set by DriftMonitor
    drift_detected: bool
    drift_score: float
    drift_details: Dict[str, float]   # per-feature PSI / KS scores

    # Set by TrainAgent
    new_data_path: Optional[str]
    model_artifact_path: Optional[str]
    mlflow_run_id: Optional[str]

    # Set by EvalAgent
    baseline_accuracy: Optional[float]
    new_accuracy: Optional[float]
    eval_delta: Optional[float]
    eval_passed: bool

    # Set by ApprovalAgent / human
    approved: Optional[bool]
    approval_message: Optional[str]

    # Set by DeployAgent
    deployed: bool
    deployed_model_path: Optional[str]

    # Meta
    error: Optional[str]
