from __future__ import annotations
from sqlalchemy import create_engine, Column, String, Float, Boolean, DateTime, JSON
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class RetrainingRun(Base):
    __tablename__ = "retraining_runs"

    run_id              = Column(String, primary_key=True)
    created_at          = Column(DateTime, default=datetime.utcnow)
    drift_score         = Column(Float,   nullable=True)
    drift_detected      = Column(Boolean, nullable=True)
    drift_details       = Column(JSON,    nullable=True)
    baseline_accuracy   = Column(Float,   nullable=True)
    new_accuracy        = Column(Float,   nullable=True)
    eval_delta          = Column(Float,   nullable=True)
    eval_passed         = Column(Boolean, nullable=True)
    approved            = Column(Boolean, nullable=True)
    approval_message    = Column(String,  nullable=True)
    deployed            = Column(Boolean, nullable=True)
    deployed_model_path = Column(String,  nullable=True)
    mlflow_run_id       = Column(String,  nullable=True)
    error               = Column(String,  nullable=True)
    status              = Column(String,  default="running")
