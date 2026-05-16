from graph.state import RetrainingState


def should_retrain(state: RetrainingState) -> str:
    """After DriftMonitor: route to data collection or end."""
    if state.get("error"):
        return "end"
    return "collect_data" if state["drift_detected"] else "end"


def should_deploy(state: RetrainingState) -> str:
    """After EvalAgent: route to approval or end."""
    if state.get("error"):
        return "end"
    return "request_approval" if state["eval_passed"] else "end"


def approval_decision(state: RetrainingState) -> str:
    """After ApprovalAgent: deploy or end."""
    if state.get("approved") is True:
        return "deploy"
    return "end"
