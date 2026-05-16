"""
retraining_graph.py  —  langgraph replaced with plain Python runner
No external dependencies needed.
"""
from __future__ import annotations
from graph.edges import should_retrain, should_deploy, approval_decision
from agents.drift_monitor  import drift_monitor_node
from agents.train_agent    import train_agent_node
from agents.eval_agent     import eval_agent_node
from agents.approval_agent import approval_agent_node
from agents.deploy_agent   import deploy_agent_node


class _Graph:
    """Runs the same node sequence as the LangGraph pipeline, no langgraph needed."""

    def invoke(self, state: dict) -> dict:
        # Step 1: Drift Monitor
        state = drift_monitor_node(state)
        if should_retrain(state) == "end":
            return state

        # Step 2: Collect Data (reuses train node)
        state = train_agent_node(state)

        # Step 3: Retrain
        state = train_agent_node(state)

        # Step 4: Evaluate
        state = eval_agent_node(state)
        if should_deploy(state) == "end":
            return state

        # Step 5: Human Approval (auto-approves in demo mode)
        state = approval_agent_node(state)
        if approval_decision(state) == "end":
            return state

        # Step 6: Deploy
        state = deploy_agent_node(state)
        return state


retraining_graph = _Graph()
