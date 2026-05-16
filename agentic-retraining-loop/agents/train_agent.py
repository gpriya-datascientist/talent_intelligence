"""
TrainAgent  —  mlflow is optional, works without it
"""
from __future__ import annotations
import pickle
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from graph.state import RetrainingState
from config.settings import settings

try:
    import mlflow
    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False


def train_agent_node(state: RetrainingState) -> RetrainingState:
    logger.info(f"[TrainAgent] run_id={state['run_id']} — starting retraining")
    try:
        ref  = pd.read_parquet(state["reference_data_path"])
        prod = pd.read_parquet(state["production_data_path"])
        combined = pd.concat([ref, prod], ignore_index=True)

        feature_cols = [c for c in combined.columns if c.startswith("feature_")]
        X = combined[feature_cols].values
        y = combined["label"].values

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_val, model.predict(X_val))

        artifact_path = Path(settings.model_registry_path) / f"retrained_{state['run_id']}.pkl"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "wb") as f:
            pickle.dump(model, f)

        mlflow_run_id = f"local-{state['run_id']}"
        if MLFLOW_OK:
            try:
                mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
                with mlflow.start_run(run_name=f"retrain-{state['run_id']}") as run:
                    mlflow.log_metric("val_accuracy", acc)
                    mlflow.log_param("n_estimators", 100)
                    mlflow.log_artifact(str(artifact_path))
                    mlflow_run_id = run.info.run_id
            except Exception:
                pass  # mlflow server not running — skip silently

        logger.info(f"[TrainAgent] saved {artifact_path}  val_acc={acc:.4f}")
        return {
            **state,
            "model_artifact_path": str(artifact_path),
            "mlflow_run_id":       mlflow_run_id,
            "new_data_path":       state["production_data_path"],
        }
    except Exception as e:
        logger.error(f"[TrainAgent] error: {e}")
        return {**state, "error": str(e)}
