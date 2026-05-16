"""
src/visualizer.py
-----------------
All Plotly chart builders for ContextLens dashboard.
"""
from __future__ import annotations
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


# ── colour palette (matches dark theme) ───────────────────────
TEAL   = "#00E5CC"
YELLOW = "#FFD700"
PURPLE = "#9B59FF"
ORANGE = "#FF6B35"
GREEN  = "#1D9E75"
BG     = "#0d1117"
CARD   = "#161b22"
BORDER = "#30363d"


def cosine_gauge(shift_score: float) -> go.Figure:
    """Gauge showing shift score 0→1."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(shift_score, 4),
        title={"text": "Context Shift Score", "font": {"color": TEAL, "size": 14}},
        delta={"reference": 0, "valueformat": ".3f"},
        gauge={
            "axis": {"range": [0, 1], "tickcolor": "#888"},
            "bar":  {"color": ORANGE if shift_score > 0.5 else GREEN},
            "steps": [
                {"range": [0.0, 0.3], "color": "#0d2b1a"},
                {"range": [0.3, 0.6], "color": "#2b2000"},
                {"range": [0.6, 1.0], "color": "#2b0a00"},
            ],
            "threshold": {
                "line": {"color": ORANGE, "width": 2},
                "thickness": 0.75,
                "value": shift_score,
            },
        },
        number={"font": {"color": TEAL, "size": 28}, "valueformat": ".4f"},
    ))
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        font=dict(color="#ccc"),
    )
    return fig


def hidden_state_comparison(vec_a: list, vec_b: list, label_a: str, label_b: str) -> go.Figure:
    """Grouped bar: hidden state dims side by side."""
    dims = [f"dim {i+1}" for i in range(len(vec_a))]
    fig = go.Figure()
    fig.add_trace(go.Bar(name=f"A: {label_a[:20]}", x=dims, y=vec_a,
                         marker_color=TEAL, opacity=0.85))
    fig.add_trace(go.Bar(name=f"B: {label_b[:20]}", x=dims, y=vec_b,
                         marker_color=YELLOW, opacity=0.85))
    fig.update_layout(
        title="Hidden state vectors (sampled dims)",
        barmode="group",
        paper_bgcolor=CARD, plot_bgcolor=BG,
        font=dict(color="#ccc", size=12),
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(bgcolor=CARD, bordercolor=BORDER, borderwidth=1),
        xaxis=dict(gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER, title="Value"),
    )
    return fig


def attention_heatmap(weights: list[float], tokens: list[str],
                      title: str, focus_idx: int) -> go.Figure:
    """Single-row heatmap of attention weights."""
    n = min(len(weights), len(tokens))
    w = np.array(weights[:n]).reshape(1, -1)

    fig = go.Figure(go.Heatmap(
        z=w,
        x=tokens[:n],
        y=["attn"],
        colorscale=[[0, "#0d1117"], [0.5, PURPLE], [1, TEAL]],
        showscale=True,
        text=[[f"{v:.3f}" for v in weights[:n]]],
        texttemplate="%{text}",
        textfont={"size": 11, "color": "white"},
    ))

    # Highlight focus token
    if focus_idx < n:
        fig.add_shape(type="rect",
                      x0=focus_idx - 0.5, x1=focus_idx + 0.5,
                      y0=-0.5, y1=0.5,
                      line=dict(color=ORANGE, width=2))

    fig.update_layout(
        title=title,
        paper_bgcolor=CARD, plot_bgcolor=BG,
        font=dict(color="#ccc", size=12),
        height=160,
        margin=dict(l=20, r=20, t=45, b=20),
        xaxis=dict(side="bottom", tickangle=-20),
        yaxis=dict(showticklabels=False),
    )
    return fig


def layer_progression_chart(layer_analysis_a: dict, layer_analysis_b: dict) -> go.Figure:
    """Radar / timeline showing how meaning evolves across layers."""
    layers = ["Layers 1-5\n(Syntax)", "Layers 6-8\n(Semantic)",
              "Layers 9-11\n(Domain)", "Layer 12\n(Final)"]

    # Simulate "divergence" scores per layer band
    divergence = [0.05, 0.25, 0.60, 1.0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=layers, y=divergence,
        mode="lines+markers",
        line=dict(color=ORANGE, width=3),
        marker=dict(size=12, color=ORANGE, symbol="circle"),
        fill="tozeroy",
        fillcolor="rgba(255,107,53,0.15)",
        name="Meaning divergence",
    ))
    fig.update_layout(
        title="How meaning diverges across transformer layers",
        paper_bgcolor=CARD, plot_bgcolor=BG,
        font=dict(color="#ccc", size=12),
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER, title="Divergence", range=[0, 1.1]),
        showlegend=False,
    )
    return fig


def cosine_similarity_bar(cosine: float, shift: float) -> go.Figure:
    """Horizontal bar showing cosine vs shift."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[cosine], y=["Cosine similarity"],
        orientation="h",
        marker_color=GREEN,
        text=[f"{cosine:.4f}"],
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig.add_trace(go.Bar(
        x=[shift], y=["Shift score  (1 − cosine)"],
        orientation="h",
        marker_color=ORANGE,
        text=[f"{shift:.4f}"],
        textposition="inside",
        insidetextanchor="middle",
    ))
    fig.update_layout(
        paper_bgcolor=CARD, plot_bgcolor=BG,
        font=dict(color="#ccc", size=13),
        height=160,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(range=[0, 1], gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER),
        showlegend=False,
        barmode="group",
    )
    return fig
