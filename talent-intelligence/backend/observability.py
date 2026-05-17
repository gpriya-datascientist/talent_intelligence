"""
observability.py — Langfuse integration for tracing every LangChain chain.
Wrap any chain call with trace_chain() to get full observability.
"""
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from functools import wraps
from datetime import datetime
from backend.config import settings


# Singleton Langfuse client
_langfuse_client: Langfuse | None = None


def get_langfuse() -> Langfuse:
    global _langfuse_client
    if _langfuse_client is None and settings.LANGFUSE_PUBLIC_KEY:
        _langfuse_client = Langfuse(
            public_key=settings.LANGFUSE_PUBLIC_KEY,
            secret_key=settings.LANGFUSE_SECRET_KEY,
            host=settings.LANGFUSE_HOST,
        )
    return _langfuse_client


def get_langfuse_handler(trace_name: str, metadata: dict = None) -> CallbackHandler | None:
    """
    Returns a LangChain callback handler that streams trace data to Langfuse.
    Pass as callbacks=[handler] to any LangChain chain invoke.
    """
    client = get_langfuse()
    if not client:
        return None
    return CallbackHandler(
        public_key=settings.LANGFUSE_PUBLIC_KEY,
        secret_key=settings.LANGFUSE_SECRET_KEY,
        host=settings.LANGFUSE_HOST,
        trace_name=trace_name,
        metadata=metadata or {},
    )


def log_pipeline_event(
    wish_id: str,
    stage: str,
    duration_ms: int,
    success: bool,
    metadata: dict = None,
) -> None:
    """
    Log a pipeline stage completion event to Langfuse.
    Used to track per-stage latency and failure rates.
    """
    client = get_langfuse()
    if not client:
        return
    client.score(
        name=f"pipeline_{stage}",
        value=1.0 if success else 0.0,
        trace_id=wish_id,
        comment=f"Duration: {duration_ms}ms | Stage: {stage}",
    )


def log_eval_score(eval_name: str, score: float, trace_id: str = None) -> None:
    """Log eval metric results back to Langfuse for trend tracking."""
    client = get_langfuse()
    if not client:
        return
    client.score(name=eval_name, value=score, trace_id=trace_id)
