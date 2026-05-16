from __future__ import annotations

import asyncio
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.schemas import TriggerRequest, ApproveRequest, RunStatusResponse
from db.session import get_db
from db.models import RetrainingRun
from graph.retraining_graph import retraining_graph

router = APIRouter(prefix="/api/v1")


@router.post("/trigger", response_model=RunStatusResponse)
async def trigger_run(req: TriggerRequest, db: Session = Depends(get_db)):
    """Start a new retraining pipeline run."""
    run = RetrainingRun(run_id=req.run_id, status="running")
    db.add(run)
    db.commit()

    async def _run():
        initial_state = {
            "run_id":                req.run_id,
            "production_data_path":  req.production_data_path,
            "reference_data_path":   req.reference_data_path,
            "config":                req.config,
            "drift_detected":        False,
            "drift_score":           0.0,
            "drift_details":         {},
            "eval_passed":           False,
            "deployed":              False,
        }
        result = retraining_graph.invoke(initial_state)

        # Persist result
        for key, val in result.items():
            if hasattr(run, key):
                setattr(run, key, val)
        run.status = "error" if result.get("error") else "completed"
        db.commit()

    asyncio.create_task(_run())
    return RunStatusResponse(run_id=req.run_id, status="running", drift_detected=None,
                             drift_score=None, baseline_accuracy=None, new_accuracy=None,
                             eval_delta=None, approved=None, deployed=None, error=None)


@router.post("/approve")
async def approve_run(req: ApproveRequest, db: Session = Depends(get_db)):
    """Human sign-off endpoint (called by Slack webhook or UI)."""
    run = db.query(RetrainingRun).filter_by(run_id=req.run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    run.approved = req.approved
    run.approval_message = req.message
    db.commit()
    return {"status": "ok", "approved": req.approved}


@router.get("/status/{run_id}", response_model=RunStatusResponse)
def get_status(run_id: str, db: Session = Depends(get_db)):
    """Poll run status."""
    run = db.query(RetrainingRun).filter_by(run_id=run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunStatusResponse(
        run_id=run.run_id, status=run.status,
        drift_detected=run.drift_detected, drift_score=run.drift_score,
        baseline_accuracy=run.baseline_accuracy, new_accuracy=run.new_accuracy,
        eval_delta=run.eval_delta, approved=run.approved,
        deployed=run.deployed, error=run.error,
    )
