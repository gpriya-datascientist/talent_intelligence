"""
scheduler.py
------------
Automatic weekly retraining trigger using APScheduler.

What it does each week
----------------------
1. Scans DATA_RAW_DIR for new / updated files
2. Re-runs the full ingestion → preprocessing pipeline
3. Resets and re-ingests ChromaDB with fresh embeddings
4. Persists a JSON run report to logs/

Run standalone:  python -m src.scheduler.scheduler
Or import and call schedule_jobs() from any entrypoint.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from src.ingestion.loader import load_directory
from src.ingestion.preprocessor import preprocess
from src.vectorstore.chroma_store import KPIVectorStore

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------

RETRAIN_DAY    = os.getenv("RETRAIN_SCHEDULE_DAY",    "monday")
RETRAIN_HOUR   = int(os.getenv("RETRAIN_SCHEDULE_HOUR",   "2"))
RETRAIN_MINUTE = int(os.getenv("RETRAIN_SCHEDULE_MINUTE", "0"))
DATA_RAW_DIR   = Path(os.getenv("DATA_RAW_DIR", "./data/raw"))
LOG_DIR        = Path(os.getenv("LOG_DIR", "./logs"))
FREQ           = os.getenv("DEFAULT_FREQ", "W")

LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Retraining job
# ---------------------------------------------------------------------------

def retrain_job() -> dict:
    """
    Full retrain pipeline.  Called by the scheduler every week.

    Returns
    -------
    dict  run report (also written to logs/)
    """
    run_id    = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    started   = datetime.utcnow().isoformat()
    report    = {"run_id": run_id, "started": started, "status": "RUNNING"}

    logger.info(f"[Scheduler] Retraining job started — run_id={run_id}")

    try:
        # 1. Ingest all files from the raw data directory
        if not DATA_RAW_DIR.exists() or not any(DATA_RAW_DIR.iterdir()):
            logger.warning(f"[Scheduler] No files found in {DATA_RAW_DIR}. Skipping retrain.")
            report.update({"status": "SKIPPED", "reason": "No source files found"})
            _write_report(report)
            return report

        df_raw = load_directory(DATA_RAW_DIR)
        report["rows_ingested"] = len(df_raw)
        report["kpis_found"]    = df_raw["kpi_name"].nunique()
        report["files_scanned"] = df_raw["source_file"].nunique()

        # 2. Preprocess
        df_processed = preprocess(df_raw, freq=FREQ)
        report["rows_after_preprocessing"] = len(df_processed)

        # 3. Reset vector store and re-embed
        store = KPIVectorStore()
        store.reset()
        store.ingest(df_processed)
        stats = store.stats()
        report["chroma_docs_after"] = stats["total_documents"]

        # 4. Finish
        finished = datetime.utcnow().isoformat()
        report.update({
            "status":   "SUCCESS",
            "finished": finished,
        })
        logger.success(
            f"[Scheduler] Retrain complete — {stats['total_documents']} docs embedded "
            f"| run_id={run_id}"
        )

    except Exception as exc:
        logger.error(f"[Scheduler] Retrain FAILED: {exc}")
        report.update({
            "status":    "FAILED",
            "error":     str(exc),
            "finished":  datetime.utcnow().isoformat(),
        })

    _write_report(report)
    return report


def _write_report(report: dict) -> None:
    path = LOG_DIR / f"retrain_{report['run_id']}.json"
    path.write_text(json.dumps(report, indent=2))
    logger.info(f"[Scheduler] Run report saved: {path}")


# ---------------------------------------------------------------------------
# Scheduler setup
# ---------------------------------------------------------------------------

def schedule_jobs(blocking: bool = True) -> None:
    """
    Register and start the APScheduler.

    Parameters
    ----------
    blocking : If True, blocks the calling thread (use for standalone runs).
               If False, runs in background (use when embedding in another app).
    """
    SchedulerClass = BlockingScheduler if blocking else BackgroundScheduler

    scheduler = SchedulerClass(timezone="UTC")

    trigger = CronTrigger(
        day_of_week=RETRAIN_DAY,
        hour=RETRAIN_HOUR,
        minute=RETRAIN_MINUTE,
    )

    scheduler.add_job(
        retrain_job,
        trigger=trigger,
        id="weekly_retrain",
        name="Weekly KPI Retraining",
        replace_existing=True,
        misfire_grace_time=3600,   # 1 hour grace window
        coalesce=True,             # skip missed runs, run once
    )

    next_run = scheduler.get_jobs()[0].next_run_time
    logger.info(
        f"[Scheduler] Registered weekly retrain — "
        f"day={RETRAIN_DAY} hour={RETRAIN_HOUR}:00 UTC | "
        f"next run ≈ {next_run}"
    )

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("[Scheduler] Scheduler stopped.")


def get_background_scheduler() -> BackgroundScheduler:
    """
    Return a configured (but not yet started) background scheduler.
    Useful for embedding in a FastAPI lifespan or Streamlit app.
    """
    scheduler = BackgroundScheduler(timezone="UTC")
    scheduler.add_job(
        retrain_job,
        CronTrigger(day_of_week=RETRAIN_DAY, hour=RETRAIN_HOUR, minute=RETRAIN_MINUTE),
        id="weekly_retrain",
        name="Weekly KPI Retraining",
        replace_existing=True,
        misfire_grace_time=3600,
        coalesce=True,
    )
    return scheduler


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting KPI retraining scheduler (standalone mode)…")
    schedule_jobs(blocking=True)
