"""
EvalAgent
---------
Compares new model accuracy vs baseline model on a held-out test set.
Sets eval_passed=True if delta >= min_accuracy_delta.
"""
from __future__ import annotations

import pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from loguru import logger

from graph.state import RetrainingState
from config.settings import settings

BASELINE_MODEL_PATH = "data/models/baseline_model.pkl"


def eval_agent_node(state: RetrainingState) -> RetrainingState:
    logger.info(f"[EvalAgent] run_id={state['run_id']} — evaluating new model")

    try:
        ref = pd.read_parquet(state["reference_data_path"])
        feature_cols = [c for c in ref.columns if c.startswith("feature_")]
        X_test = ref[feature_cols].values
        y_test = ref["label"].values

        # Load baseline
        with open(BASELINE_MODEL_PATH, "rb") as f:
            baseline_model = pickle.load(f)
        baseline_acc = accuracy_score(y_test, baseline_model.predict(X_test))

        # Load new model
        with open(state["model_artifact_path"], "rb") as f:
            new_model = pickle.load(f)
        new_acc = accuracy_score(y_test, new_model.predict(X_test))

        delta = new_acc - baseline_acc
        eval_passed = delta >= settings.min_accuracy_delta

        logger.info(
            f"[EvalAgent] baseline={baseline_acc:.4f}  new={new_acc:.4f}  "
            f"delta={delta:+.4f}  passed={eval_passed}"
        )

        return {
            **state,
            "baseline_accuracy": round(baseline_acc, 4),
            "new_accuracy":      round(new_acc, 4),
            "eval_delta":        round(delta, 4),
            "eval_passed":       eval_passed,
        }

    except Exception as e:
        logger.error(f"[EvalAgent] error: {e}")
        return {**state, "error": str(e), "eval_passed": False}
