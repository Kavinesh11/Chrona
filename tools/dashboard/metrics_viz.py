from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_metrics_timeline_fig(
    eval_records: List[Dict[str, Any]],
    selected_metrics: Optional[List[str]] = None,
) -> go.Figure:
    """
    Creates an interactive line chart tracking NMI, AMI, and ARI across chronological training epochs/blocks.
    """
    if selected_metrics is None:
        selected_metrics = ["nmi", "ami", "ari"]

    if not eval_records:
        fig = go.Figure()
        fig.update_layout(
            title="No evaluation metrics found in run logs",
            template="plotly_dark",
        )
        return fig

    epochs = []
    data_dict: Dict[str, List[float]] = {m: [] for m in selected_metrics}

    for idx, rec in enumerate(eval_records):
        epochs.append(rec.get("epoch", idx))
        for m in selected_metrics:
            # Check various keys for metrics (e.g., test_nmi, validation_nmi, nmi)
            val = (
                rec.get(f"test_{m}")
                or rec.get(f"val_{m}")
                or rec.get(f"validation_{m}")
                or rec.get(m)
                or rec.get(f"with_isolated_nodes_test_{m}")
            )
            data_dict[m].append(val if val is not None else np.nan)

    df = pd.DataFrame({"Step": epochs, **data_dict})

    fig = go.Figure()
    colors = {"nmi": "#3b82f6", "ami": "#10b981", "ari": "#8b5cf6", "loss": "#ef4444"}

    for m in selected_metrics:
        if m in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["Step"],
                    y=df[m],
                    mode="lines+markers",
                    name=m.upper(),
                    line=dict(color=colors.get(m, "#f59e0b"), width=3),
                    marker=dict(size=7),
                )
            )

    fig.update_layout(
        title="Clustering Performance Metrics over Time (NMI / AMI / ARI)",
        xaxis_title="Chronological Window / Block",
        yaxis_title="Score (0.0 to 1.0)",
        yaxis=dict(range=[0, 1.05]),
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_radar_metric_fig(metrics_dict: Dict[str, float], title: str = "Model Overview") -> go.Figure:
    """
    Creates a radar/spider chart comparing NMI, AMI, ARI scores.
    """
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    # Close loop
    categories.append(categories[0])
    values.append(values[0])

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=[c.upper() for c in categories],
            fill="toself",
            name=title,
            fillcolor="rgba(59, 130, 246, 0.4)",
            line=dict(color="#3b82f6", width=2),
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1.0])),
        showlegend=False,
        template="plotly_dark",
        title=title,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    return fig
