from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def compute_embedding_projection(
    features: np.ndarray, method: str = "PCA", n_components: int = 3, random_state: int = 42
) -> np.ndarray:
    """
    Reduces feature dimensionality to 2D or 3D via PCA or t-SNE.
    """
    if features.shape[1] <= n_components:
        return features

    if method == "t-SNE":
        perplexity = min(30, max(5, features.shape[0] // 5))
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        return reducer.fit_transform(features)
    else:  # Default PCA
        reducer = PCA(n_components=n_components, random_state=random_state)
        return reducer.fit_transform(features)


def create_embedding_cluster_fig(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "PCA",
    is_3d: bool = True,
    top_k_clusters: int = 10,
    text_samples: Optional[List[str]] = None,
) -> go.Figure:
    """
    Creates an interactive 2D or 3D Plotly scatter plot of message embeddings.
    Points are colored by event cluster, with hover details showing tweet IDs and sample text.
    """
    n_dim = 3 if is_3d else 2
    coords = compute_embedding_projection(features, method=method, n_components=n_dim)

    # Determine top K clusters by size
    counts = pd.Series(labels).value_counts()
    top_labels = set(counts.head(top_k_clusters).index)

    df_dict = {
        "Comp_1": coords[:, 0],
        "Comp_2": coords[:, 1],
        "Msg_ID": np.arange(len(labels)),
        "Cluster": [f"Event {lbl}" if lbl in top_labels else "Other Events" for lbl in labels],
        "Raw_Label": labels,
    }
    if is_3d:
        df_dict["Comp_3"] = coords[:, 2]

    if text_samples and len(text_samples) == len(labels):
        df_dict["Snippet"] = [s[:80] + "..." if len(s) > 80 else s for s in text_samples]
    else:
        df_dict["Snippet"] = [f"Tweet Message #{i}" for i in range(len(labels))]

    df = pd.DataFrame(df_dict)

    hover_data = {
        "Msg_ID": True,
        "Raw_Label": True,
        "Snippet": True,
        "Comp_1": ":.2f",
        "Comp_2": ":.2f",
    }
    if is_3d:
        hover_data["Comp_3"] = ":.2f"

    if is_3d:
        fig = px.scatter_3d(
            df,
            x="Comp_1",
            y="Comp_2",
            z="Comp_3",
            color="Cluster",
            hover_name="Snippet",
            hover_data=hover_data,
            title=f"3D Message Embeddings Space ({method} Projection)",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            opacity=0.85,
        )
        fig.update_traces(marker=dict(size=4, line=dict(width=0.5, color="DarkSlateGrey")))
        fig.update_layout(
            margin=dict(l=0, r=0, b=0, t=40),
            scene=dict(
                xaxis_title="Comp 1",
                yaxis_title="Comp 2",
                zaxis_title="Comp 3",
            ),
        )
    else:
        fig = px.scatter(
            df,
            x="Comp_1",
            y="Comp_2",
            color="Cluster",
            hover_name="Snippet",
            hover_data=hover_data,
            title=f"2D Message Embeddings Space ({method} Projection)",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            opacity=0.85,
        )
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, b=20, t=40),
        )

    return fig
