"""
observability.py — Langfuse v4+ compatible. CallbackHandler now in langfuse.langchain
"""
import time
import logging
logger = logging.getLogger("talent_intelligence")

LANGFUSE_AVAILABLE = False
_Langfuse_cls = None
_CB_cls = None

try:
    from langfuse import Langfuse as _L
    from langfuse.langchain import CallbackHandler as _C
    _Langfuse_cls = _L
    _CB_cls = _C
    LANGFUSE_AVAILABLE = True
except ImportError:
    pass

from backend.config import settings

_langfuse_client = None


def get_langfuse():
    global _langfuse_client
    if not LANGFUSE_AVAILABLE or not settings.LANGFUSE_PUBLIC_KEY:
        return None
    if _langfuse_client is None:
        try:
            _langfuse_client = _Langfuse_cls(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
            logger.info("Langfuse connected")
        except Exception as e:
            logger.warning(f"Langfuse init failed: {e}")
    return _langfuse_client


def get_langfuse_handler(trace_name: str, wish_id: str = None, metadata: dict = None):
    if not LANGFUSE_AVAILABLE or not settings.LANGFUSE_PUBLIC_KEY:
        return None
    try:
        # Try minimal init — no extra kwargs that might fail
        handler = _CB_cls(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
        return handler
    except Exception as e:
        logger.warning(f"Langfuse handler failed: {e}")
        return None


def _langfuse_score(client, name: str, value: float, trace_id: str, comment: str = None):
    """Compatibility wrapper for Langfuse score API across versions."""
    try:
        # Langfuse v3+ API
        client.create_score(name=name, value=value, trace_id=trace_id, comment=comment)
    except AttributeError:
        try:
            # Older API fallback
            client.score(name=name, value=value, trace_id=trace_id)
        except Exception:
            pass


def log_pipeline_event(wish_id: str, stage: str, duration_ms: int, success: bool, metadata: dict = None):
    client = get_langfuse()
    if not client:
        return
    try:
        _langfuse_score(client, f"pipeline_{stage}", 1.0 if success else 0.0, wish_id)
        _langfuse_score(client, f"latency_{stage}_ms", float(duration_ms), wish_id)
    except Exception as e:
        logger.warning(f"Langfuse log_pipeline_event: {e}")


def log_confidence_score(wish_id: str, stage: str, confidence: float):
    client = get_langfuse()
    if not client:
        return
    try:
        _langfuse_score(client, f"confidence_{stage}", confidence, wish_id)
    except Exception as e:
        logger.warning(f"Langfuse log_confidence_score: {e}")


def log_po_feedback(wish_id: str, candidate_id: str, rating: int):
    client = get_langfuse()
    if not client:
        return
    try:
        _langfuse_score(client, "po_feedback", float(rating), wish_id,
                        comment=f"candidate={candidate_id}")
    except Exception as e:
        logger.warning(f"Langfuse log_po_feedback: {e}")


def log_eval_score(eval_name: str, score: float, trace_id: str = None):
    client = get_langfuse()
    if not client:
        return
    try:
        _langfuse_score(client, eval_name, score, trace_id or "global")
    except Exception as e:
        logger.warning(f"Langfuse log_eval_score: {e}")


class StageTimer:
    """Async context manager that times a stage and logs to Langfuse."""
    def __init__(self, wish_id: str, stage: str):
        self.wish_id     = wish_id
        self.stage       = stage
        self.duration_ms = 0
        self._start      = 0.0

    async def __aenter__(self):
        self._start = time.monotonic()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = round((time.monotonic() - self._start) * 1000)
        log_pipeline_event(self.wish_id, self.stage, self.duration_ms, exc_type is None)
        return False
