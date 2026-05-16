"""
ApprovalAgent
-------------
Sends a Slack message with eval summary and waits for human sign-off.
In demo mode (no Slack token) it auto-approves.
"""
from __future__ import annotations

import os
from loguru import logger

from graph.state import RetrainingState
from config.settings import settings


def _send_slack_alert(state: RetrainingState) -> None:
    try:
        from slack_sdk import WebClient
        client = WebClient(token=settings.slack_bot_token)
        client.chat_postMessage(
            channel=settings.slack_channel_id,
            text=(
                f":robot_face: *Retraining approval needed* — `{state['run_id']}`\n"
                f">Baseline accuracy: `{state['baseline_accuracy']}`\n"
                f">New model accuracy: `{state['new_accuracy']}`\n"
                f">Delta: `{state['eval_delta']:+.4f}`\n"
                f"Reply with `approve {state['run_id']}` or `reject {state['run_id']}`"
            ),
        )
        logger.info("[ApprovalAgent] Slack alert sent")
    except Exception as e:
        logger.warning(f"[ApprovalAgent] Slack send failed: {e}")


def approval_agent_node(state: RetrainingState) -> RetrainingState:
    logger.info(f"[ApprovalAgent] run_id={state['run_id']} — requesting approval")

    _send_slack_alert(state)

    # Demo mode: auto-approve when no real Slack flow is wired
    demo_mode = not settings.slack_bot_token or settings.slack_bot_token.startswith("xoxb-your")
    if demo_mode:
        logger.info("[ApprovalAgent] Demo mode — auto-approving")
        return {**state, "approved": True, "approval_message": "auto-approved (demo)"}
    # In production: approval is injected via POST /approve endpoint
    # The graph is interrupted here and resumed by the API layer
    return {**state, "approved": None, "approval_message": "awaiting human response"}
