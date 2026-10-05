from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_event_stream_heatmap(
    block_labels: Dict[int, np.ndarray], top_k_events: int = 12
) -> go.Figure:
    """
    Creates a heatmap showing event category message volume across chronological blocks M_0 to M_21.
    """
    if not block_labels:
        fig = go.Figure()
        fig.update_layout(title="No block data available", template="plotly_dark")
        return fig

    # Collect cluster counts per block
    all_events = set()
    block_counts: Dict[int, Dict[int, int]] = {}

    for blk, lbls in sorted(block_labels.items()):
        counts = pd.Series(lbls).value_counts().to_dict()
        block_counts[blk] = counts
        all_events.update(counts.keys())

    # Pick top K overall events by total volume
    totals = {}
    for ev in all_events:
        totals[ev] = sum(block_counts[b].get(ev, 0) for b in block_counts)

    top_events = [ev for ev, _ in sorted(totals.items(), key=lambda kv: -kv[1])[:top_k_events]]

    matrix = []
    block_names = [f"Block {b}" for b in sorted(block_counts.keys())]
    event_names = [f"Event {ev}" for ev in top_events]

    for ev in top_events:
        row = [block_counts[b].get(ev, 0) for b in sorted(block_counts.keys())]
        matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=block_names,
            y=event_names,
            colorscale="Viridis",
            hoverongaps=False,
            colorbar=dict(title="Tweet Volume"),
        )
    )

    fig.update_layout(
        title="Streaming Event Volume Lifecycle (Blocks M_0 to M_21)",
        xaxis_title="Chronological Streaming Window",
        yaxis_title="Event Community",
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig


def create_event_growth_stacked_area(
    block_labels: Dict[int, np.ndarray], top_k_events: int = 8
) -> go.Figure:
    """
    Creates an interactive stacked area chart tracking volume of emerging events over time.
    """
    if not block_labels:
        fig = go.Figure()
        fig.update_layout(title="No block data available", template="plotly_dark")
        return fig

    block_indices = sorted(block_labels.keys())
    
    # Calculate top events
    total_counts = {}
    for b in block_indices:
        for lbl, count in pd.Series(block_labels[b]).value_counts().items():
            total_counts[lbl] = total_counts.get(lbl, 0) + count

    top_events = [lbl for lbl, _ in sorted(total_counts.items(), key=lambda kv: -kv[1])[:top_k_events]]

    data = {"Block": block_indices}
    for ev in top_events:
        data[f"Event {ev}"] = [
            int((block_labels[b] == ev).sum()) for b in block_indices
        ]

    df = pd.DataFrame(data)

    fig = px.area(
        df,
        x="Block",
        y=[f"Event {ev}" for ev in top_events],
        title="Event Cluster Emergence & Decay Stream",
        labels={"value": "Message Count", "variable": "Event Cluster"},
        color_discrete_sequence=px.colors.qualitative.Plotly,
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Chronological Block Index",
        yaxis_title="Active Messages in Graph",
        margin=dict(l=40, r=40, t=50, b=40),
    )

    return fig
