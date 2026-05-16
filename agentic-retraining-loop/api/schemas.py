from pydantic import BaseModel
from typing import Optional, Dict, Any


class TriggerRequest(BaseModel):
    run_id:                 str
    production_data_path:   str
    reference_data_path:    str
    config:                 Dict[str, Any] = {}


class ApproveRequest(BaseModel):
    run_id:   str
    approved: bool
    message:  Optional[str] = None


class RunStatusResponse(BaseModel):
    run_id:             str
    status:             str
    drift_detected:     Optional[bool]
    drift_score:        Optional[float]
    baseline_accuracy:  Optional[float]
    new_accuracy:       Optional[float]
    eval_delta:         Optional[float]
    approved:           Optional[bool]
    deployed:           Optional[bool]
    error:              Optional[str]
