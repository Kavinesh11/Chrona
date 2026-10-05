from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import sparse


def create_node_filter_breakdown_fig(
    total_nodes: int = 1000,
    kept_sentiment: int = 540,
    kept_centrality: int = 500,
) -> go.Figure:
    """
    Creates a bar chart comparing node retention rates across node-filtering strategies.
    """
    df = pd.DataFrame(
        {
            "Strategy": [
                "No Filtering (--filter_method none)",
                "Sentiment Confidence (--filter_method sentiment)",
                "Graph Centrality (--filter_method centrality)",
            ],
            "Nodes Retained": [total_nodes, kept_sentiment, kept_centrality],
            "Noise Pruned": [0, total_nodes - kept_sentiment, total_nodes - kept_centrality],
            "Retention Rate (%)": [
                100.0,
                (kept_sentiment / total_nodes) * 100,
                (kept_centrality / total_nodes) * 100,
            ],
        }
    )

    fig = px.bar(
        df,
        x="Strategy",
        y="Nodes Retained",
        text="Retention Rate (%)",
        color="Strategy",
        title="Node Filtering & Noise Pruning Comparison (§4.4)",
        color_discrete_sequence=["#64748b", "#3b82f6", "#8b5cf6"],
    )

    fig.update_traces(texttemplate="%{text:.1f}% Kept", textposition="outside")
    fig.update_layout(
        template="plotly_dark",
        yaxis_title="Active Messages in GNN Graph",
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig


def create_degree_distribution_plotly(
    adjacency: sparse.csr_matrix, block: int = 0
) -> go.Figure:
    """
    Creates an interactive histogram of node degrees for a given message graph block.
    """
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()

    fig = px.histogram(
        x=degrees,
        nbins=min(50, max(10, int(degrees.max()) or 10)),
        title=f"Message-Graph Node Degree Distribution (Block {block})",
        labels={"x": "Node Degree (Connected Messages)", "count": "Number of Tweets"},
        color_discrete_sequence=["#06b6d4"],
    )

    mean_deg = float(np.mean(degrees))
    median_deg = float(np.median(degrees))

    fig.add_vline(
        x=mean_deg,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Mean Degree: {mean_deg:.1f}",
    )
    fig.add_vline(
        x=median_deg,
        line_dash="dot",
        line_color="#10b981",
        annotation_text=f"Median: {median_deg:.1f}",
    )

    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig
