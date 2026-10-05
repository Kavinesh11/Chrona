from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
from scipy import sparse

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except ImportError:
    PYVIS_AVAILABLE = False


def create_network_graph(
    adjacency: sparse.csr_matrix,
    labels: Optional[np.ndarray] = None,
    max_nodes: int = 150,
    min_degree: int = 1,
) -> nx.Graph:
    """
    Converts a sparse adjacency matrix into a NetworkX Graph, filtering peripheral low-degree nodes.
    """
    n_nodes = adjacency.shape[0]
    sub_nodes = min(n_nodes, max_nodes)

    # Calculate degrees
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    top_indices = np.argsort(-degrees)[:sub_nodes]

    # Sub-matrix
    sub_adj = adjacency[top_indices, :][:, top_indices]
    
    G = nx.Graph()
    for idx, orig_id in enumerate(top_indices):
        deg = int(degrees[orig_id])
        if deg < min_degree:
            continue
        cluster_id = int(labels[orig_id]) if (labels is not None and orig_id < len(labels)) else 0
        G.add_node(
            int(orig_id),
            degree=deg,
            cluster=cluster_id,
            label=f"Tweet #{orig_id}",
            title=f"Tweet #{orig_id}<br>Degree: {deg}<br>Cluster: Event {cluster_id}",
        )

    cx = sparse.coo_matrix(sub_adj)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        if i < j:
            u_id = int(top_indices[i])
            v_id = int(top_indices[j])
            G.add_edge(u_id, v_id, weight=float(v))

    return G


def generate_pyvis_html(
    G: nx.Graph,
    height: str = "500px",
    width: str = "100%",
    dark_mode: bool = True,
) -> str:
    """
    Renders a PyVis interactive graph to HTML string.
    """
    if not PYVIS_AVAILABLE:
        return "<div style='color:red;'>PyVis package not installed. Run `pip install pyvis` to enable graph visualization.</div>"

    net = Network(height=height, width=width, bgcolor="#0e1117" if dark_mode else "#ffffff", font_color="#ffffff" if dark_mode else "#000000")
    net.from_nx(G)

    # Color nodes by cluster ID using discrete color palette
    colors = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ef4444", "#06b6d4", "#ec4899", "#84cc16"]
    for node in net.nodes:
        cluster = node.get("cluster", 0)
        deg = node.get("degree", 1)
        node["color"] = colors[cluster % len(colors)]
        node["size"] = max(8, min(30, 8 + np.log2(deg + 1) * 4))

    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -3000,
          "centralGravity": 0.3,
          "springLength": 95
        },
        "minVelocity": 0.75
      }
    }
    """)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        temp_path = f.name

    net.save_graph(temp_path)
    with open(temp_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    Path(temp_path).unlink(missing_ok=True)
    return html_content
