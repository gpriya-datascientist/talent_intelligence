"""
DeployAgent
-----------
Swaps the production model by copying the new artifact to the
canonical baseline path. Logs the swap to the DB.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from loguru import logger

from graph.state import RetrainingState

PRODUCTION_MODEL_PATH = "data/models/baseline_model.pkl"


def deploy_agent_node(state: RetrainingState) -> RetrainingState:
    logger.info(f"[DeployAgent] run_id={state['run_id']} — deploying new model")

    try:
        src = Path(state["model_artifact_path"])
        dst = Path(PRODUCTION_MODEL_PATH)

        # Keep a backup of old model
        backup = dst.with_name(f"baseline_model_backup_{state['run_id']}.pkl")
        if dst.exists():
            shutil.copy2(dst, backup)
            logger.info(f"[DeployAgent] backed up old model → {backup}")

        shutil.copy2(src, dst)
        logger.info(f"[DeployAgent] ✅ deployed {src} → {dst}")

        return {
            **state,
            "deployed":             True,
            "deployed_model_path":  str(dst),
        }

    except Exception as e:
        logger.error(f"[DeployAgent] error: {e}")
        return {**state, "error": str(e), "deployed": False}
