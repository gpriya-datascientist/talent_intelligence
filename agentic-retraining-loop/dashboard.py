"""
dashboard.py  —  Agentic Retraining Loop · Streamlit Dashboard
---------------------------------------------------------------
Run with:  streamlit run dashboard.py
"""
from __future__ import annotations

import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

# ── page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="Retraining Agent Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── session state ──────────────────────────────────────────────
def _init():
    defaults = {
        "run_history":     [],   # list of result dicts
        "current_run":     None,
        "log_lines":       [],
        "running":         False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── helpers ────────────────────────────────────────────────────

SAMPLE_DIR   = Path("data/sample")
MODELS_DIR   = Path("data/models")
BASELINE_PKL = MODELS_DIR / "baseline_model.pkl"

DATA_OPTIONS = {
    "🟢 No drift  (production_nodrift.parquet)": str(SAMPLE_DIR / "production_nodrift.parquet"),
    "🔴 Drift detected  (production_drift.parquet)": str(SAMPLE_DIR / "production_drift.parquet"),
    "📁 Upload my own file": "__upload__",
}

AGENT_STEPS = [
    ("monitor",          "Drift Monitor"),
    ("collect_data",     " Collect Data"),
    ("retrain",          "️ Retrain Model"),
    ("evaluate",         " Evaluate"),
    ("request_approval", "Human Approval"),
    ("deploy",           "Deploy"),
]

def _log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state["log_lines"].append(f"[{ts}] {msg}")


def _run_pipeline(prod_path: str, ref_path: str, config: dict) -> dict:
    """Execute the LangGraph pipeline with live step logging."""
    from graph.retraining_graph import retraining_graph

    run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    _log(f"Starting run {run_id}")

    initial_state = {
        "run_id":                run_id,
        "production_data_path":  prod_path,
        "reference_data_path":   ref_path,
        "config":                config,
        "drift_detected":        False,
        "drift_score":           0.0,
        "drift_details":         {},
        "eval_passed":           False,
        "deployed":              False,
    }

    result = retraining_graph.invoke(initial_state)
    result["run_id"] = run_id
    result["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _log(f"Run {run_id} completed — deployed={result.get('deployed', False)}")
    return result


def _feature_drift_chart(drift_details: dict) -> go.Figure:
    features = list(drift_details.keys())
    scores   = list(drift_details.values())
    colors   = ["#D85A30" if s > 0.2 else "#1D9E75" for s in scores]
    fig = go.Figure(go.Bar(
        x=features, y=scores,
        marker_color=colors,
        text=[f"{s:.3f}" for s in scores],
        textposition="outside",
    ))
    fig.add_hline(y=0.2, line_dash="dash", line_color="#BA7517",
                  annotation_text="threshold (0.2)", annotation_position="top right")
    fig.update_layout(
        title="Per-feature drift score (PSI / KS)",
        yaxis_title="Drift score", xaxis_title="",
        height=320, margin=dict(l=20, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


def _accuracy_chart(baseline: float, new: float) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=["Baseline model", "Retrained model"],
        y=[baseline, new],
        marker_color=["#378ADD", "#1D9E75" if new >= baseline else "#D85A30"],
        text=[f"{baseline:.4f}", f"{new:.4f}"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Accuracy comparison",
        yaxis=dict(title="Accuracy", range=[max(0, min(baseline, new) - 0.05), 1.0]),
        height=300, margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig



# ── sidebar ────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Retraining Agent")
    st.markdown("---")

    st.markdown("###  Input Data")
    data_choice = st.selectbox("Production data", list(DATA_OPTIONS.keys()))
    prod_path = DATA_OPTIONS[data_choice]

    uploaded_file = None
    if prod_path == "__upload__":
        uploaded_file = st.file_uploader("Upload Parquet file", type=["parquet"])
        if uploaded_file:
            save_path = SAMPLE_DIR / uploaded_file.name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(uploaded_file.read())
            prod_path = str(save_path)
            st.success(f"Saved: {uploaded_file.name}")

    ref_path = str(SAMPLE_DIR / "reference.parquet")
    st.caption(f"Reference: `data/sample/reference.parquet`")

    st.markdown("---")
    st.markdown("### Config")
    drift_threshold    = st.slider("Drift threshold (PSI/KS)", 0.05, 0.5, 0.2, 0.05)
    min_accuracy_delta = st.slider("Min accuracy delta", 0.001, 0.05, 0.01, 0.001, format="%.3f")

    st.markdown("---")

    # Check prerequisites
    sample_ok   = (SAMPLE_DIR / "reference.parquet").exists()
    baseline_ok = BASELINE_PKL.exists()

    if not sample_ok:
        st.warning("Sample data missing. Run:\n`python scripts/generate_sample_data.py`")
    if not baseline_ok:
        st.warning(" Baseline model missing. Run:\n`python scripts/train_baseline_model.py`")

    can_run = sample_ok and baseline_ok and (prod_path != "__upload__" or uploaded_file)

    run_btn = st.button(
        "▶ Run Retraining Pipeline",
        type="primary",
        use_container_width=True,
        disabled=not can_run or st.session_state["running"],
    )

    st.markdown("---")
    st.markdown("### Setup shortcuts")
    st.code("python scripts/generate_sample_data.py\npython scripts/train_baseline_model.py", language="bash")



# ── main header ────────────────────────────────────────────────

st.markdown("# Agentic Retraining Loop")
st.markdown("Autonomous ML retraining · LangGraph · MLflow · FastAPI")
st.markdown("---")

# ── run pipeline on button click ───────────────────────────────

if run_btn and can_run:
    st.session_state["running"]   = True
    st.session_state["log_lines"] = []
    st.session_state["current_run"] = None

    config = {
        "drift_threshold":    drift_threshold,
        "min_accuracy_delta": min_accuracy_delta,
    }

    with st.spinner("Pipeline running…"):
        try:
            result = _run_pipeline(prod_path, ref_path, config)
            st.session_state["current_run"] = result
            st.session_state["run_history"].insert(0, result)
        except Exception as e:
            _log(f"ERROR: {e}")
            st.error(f"Pipeline error: {e}")

    st.session_state["running"] = False
    st.rerun()

# ── agent progress stepper ─────────────────────────────────────

result = st.session_state.get("current_run")

st.markdown("### Agent Pipeline")

step_cols = st.columns(len(AGENT_STEPS))
for i, (key, label) in enumerate(AGENT_STEPS):
    with step_cols[i]:
        if result is None:
            st.markdown(f"<div style='text-align:center;padding:8px;border:1px solid var(--border);border-radius:8px;color:gray'>{label}<br><small>—</small></div>", unsafe_allow_html=True)
        else:
            drift_ok = result.get("drift_detected", False)
            eval_ok  = result.get("eval_passed", False)
            deployed = result.get("deployed", False)
            approved = result.get("approved", False)

            skipped = (
                (key in ("collect_data","retrain","evaluate","request_approval","deploy") and not drift_ok) or
                (key in ("request_approval","deploy") and not eval_ok) or
                (key == "deploy" and not approved)
            )

            if skipped:
                color, icon = "#888", "⏭"
                badge = "skipped"
            else:
                color, icon = "#1D9E75", "✅"
                badge = "done"

            st.markdown(
                f"<div style='text-align:center;padding:8px;border:1.5px solid {color};"
                f"border-radius:8px;color:{color}'>{label}<br><small>{badge}</small></div>",
                unsafe_allow_html=True,
            )

st.markdown("---")


# ── results panel ──────────────────────────────────────────────

if result:
    tab_drift, tab_eval, tab_deploy, tab_logs = st.tabs([
        " Drift Analysis", "Evaluation", "Deployment", "Logs"
    ])

    # ── Tab 1: Drift ──
    with tab_drift:
        c1, c2, c3 = st.columns(3)
        c1.metric("Drift Detected",
                  "YES" if result.get("drift_detected") else "NO")
        c2.metric("Max Drift Score",
                  f"{result.get('drift_score', 0.0):.4f}",
                  delta=f"threshold {drift_threshold}")
        c3.metric("Run ID", result.get("run_id", "—"))

        details = result.get("drift_details", {})
        if details:
            st.plotly_chart(_feature_drift_chart(details), use_container_width=True)

            with st.expander("Raw drift scores per feature"):
                df_drift = pd.DataFrame({
                    "Feature": list(details.keys()),
                    "Score":   list(details.values()),
                    "Status":  ["DRIFT" if v > drift_threshold else "OK"
                                for v in details.values()],
                })
                st.dataframe(df_drift, use_container_width=True, hide_index=True)
        else:
            st.info("No drift details available — drift not detected, pipeline exited early.")

    # ── Tab 2: Evaluation ──
    with tab_eval:
        if not result.get("drift_detected"):
            st.info("Drift not detected — retraining was not triggered.")
        else:
            baseline_acc = result.get("baseline_accuracy")
            new_acc      = result.get("new_accuracy")
            delta        = result.get("eval_delta")
            eval_passed  = result.get("eval_passed", False)

            if baseline_acc is not None:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Baseline Accuracy", f"{baseline_acc:.4f}")
                c2.metric("New Model Accuracy", f"{new_acc:.4f}",
                          delta=f"{delta:+.4f}")
                c3.metric("Eval Passed", "YES" if eval_passed else "NO")
                c4.metric("MLflow Run", result.get("mlflow_run_id", "—")[:8] + "…"
                          if result.get("mlflow_run_id") else "—")

                st.plotly_chart(_accuracy_chart(baseline_acc, new_acc),
                                use_container_width=True)
            else:
                st.warning("Evaluation did not run.")

    # ── Tab 3: Deployment ──
    with tab_deploy:
        if not result.get("eval_passed"):
            st.info("Evaluation did not pass — deployment was not triggered.")
        else:
            c1, c2 = st.columns(2)
            c1.metric("Approval",  "Approved" if result.get("approved") else "Rejected")
            c2.metric("Deployed",  "YES"       if result.get("deployed") else "NO")

            if result.get("deployed"):
                st.success(f"Model deployed → `{result.get('deployed_model_path')}`")
                st.markdown(f"**Approval message:** {result.get('approval_message', '—')}")
                st.markdown(f"**Finished at:** {result.get('finished_at', '—')}")
            else:
                st.warning("Model was not deployed.")

    # ── Tab 4: Logs ──
    with tab_logs:
        logs = st.session_state.get("log_lines", [])
        if logs:
            st.code("\n".join(logs), language="text")
        else:
            st.info("No log output yet.")

else:
    # Empty state
    st.markdown("#### Choose your input data in the sidebar and click **Run Retraining Pipeline**")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Scenario A — No drift**")
        st.markdown("Select `production_nodrift.parquet`. Drift monitor exits early — no retraining triggered.")
    with col2:
        st.markdown("**Scenario B — Drift detected**")
        st.markdown("Select `production_drift.parquet`. Full pipeline runs: retrain → eval → approve → deploy.")

st.markdown("---")


# ── run history table ──────────────────────────────────────────

history = st.session_state.get("run_history", [])

if history:
    st.markdown("### Run History")
    rows = []
    for r in history:
        rows.append({
            "Run ID":           r.get("run_id", "—"),
            "Finished":         r.get("finished_at", "—"),
            "Drift":            "YES" if r.get("drift_detected") else "NO",
            "Drift Score":      f"{r.get('drift_score', 0):.4f}",
            "Baseline Acc":     f"{r.get('baseline_accuracy', '—')}",
            "New Acc":          f"{r.get('new_accuracy', '—')}",
            "Delta":            f"{r.get('eval_delta', '—')}",
            "Approved":         "✅" if r.get("approved") else ("—" if r.get("approved") is None else "❌"),
            "Deployed":         "✅" if r.get("deployed") else "❌",
        })
    df_hist = pd.DataFrame(rows)
    st.dataframe(df_hist, use_container_width=True, hide_index=True)

    if len(history) > 1:
        st.markdown("#### Drift score trend across runs")
        fig_trend = px.line(
            x=[r.get("run_id") for r in reversed(history)],
            y=[r.get("drift_score", 0) for r in reversed(history)],
            markers=True,
            labels={"x": "Run", "y": "Drift Score"},
        )
        fig_trend.add_hline(y=drift_threshold, line_dash="dash",
                            line_color="#BA7517", annotation_text="threshold")
        fig_trend.update_layout(height=260, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig_trend, use_container_width=True)

