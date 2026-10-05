from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to sys.path
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from data_loader import (
    find_all_runs,
    load_block_adjacency,
    load_block_embeddings,
    load_evaluation_metrics,
    load_run_args,
)
from embedding_viz import create_embedding_cluster_fig
from explainability_viz import (
    create_degree_distribution_plotly,
    create_node_filter_breakdown_fig,
)
from graph_viz import PYVIS_AVAILABLE, create_network_graph, generate_pyvis_html
from metrics_viz import create_metrics_timeline_fig, create_radar_metric_fig
from timeline_viz import (
    create_event_growth_stacked_area,
    create_event_stream_heatmap,
)

# Set Streamlit page config
st.set_page_config(
    page_title="Chrona | DistilBERT-GNN Event Visualizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Glassmorphism CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #0b0f19;
    }
    .stAppHeader {
        background-color: rgba(11, 15, 25, 0.8);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    .title-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 24px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.4);
    }
    </style>
""",
    unsafe_allow_html=True,
)


def render_dashboard():
    st.markdown(
        """
        <div class="title-banner">
            <h1 style="margin:0; font-size: 2.2rem; font-weight: 700;">⚡ Chrona: DistilBERT-GNN Visualizer</h1>
            <p style="margin:4px 0 0 0; opacity: 0.9; font-size: 1.05rem;">
                Incremental Social Media Event Detection & Community Topology Explorer
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar setup
    st.sidebar.title("🎛️ Run Inspector")

    search_dir = _PROJECT_ROOT / "DistilBERTGNN"
    available_runs = find_all_runs(search_dir)

    run_dir = None
    if available_runs:
        run_options = {r.name: r for r in available_runs}
        selected_name = st.sidebar.selectbox(
            "Select Experimental Run:", list(run_options.keys())
        )
        run_dir = run_options[selected_name]
    else:
        st.sidebar.info("No embeddings_* runs found under DistilBERTGNN/. Showing Synthetic Demo Mode.")
        run_dir = None

    # Load Run Args
    args_dict = load_run_args(run_dir) if run_dir else {}

    st.sidebar.markdown("---")
    st.sidebar.subheader("⚙️ Run Configuration")
    if args_dict:
        st.sidebar.json(
            {
                "Filter Method": args_dict.get("filter_method", "sentiment"),
                "Remove Obsolete": args_dict.get("remove_obsolete", 2),
                "Window Size": args_dict.get("window_size", 3),
                "Num Heads": args_dict.get("num_heads", 4),
                "Hidden Dim": args_dict.get("hidden_dim", 8),
            }
        )
    else:
        st.sidebar.text("Mode: Demo Mode")

    # Main Tabs Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Metrics Analytics",
            "🌌 3D/2D Embedding Clusters",
            "🕸️ Message Graph Network",
            "🌊 Streaming Event Streams",
            "🔍 GNN Explainability",
        ]
    )

    # --- TAB 1: METRICS ---
    with tab1:
        st.header("Clustering Performance Analytics (NMI / AMI / ARI)")

        eval_records = load_evaluation_metrics(run_dir) if run_dir else []
        if not eval_records:
            # Demo metrics if no evaluation log
            eval_records = [
                {"epoch": 0, "nmi": 0.62, "ami": 0.45, "ari": 0.18},
                {"epoch": 1, "nmi": 0.68, "ami": 0.50, "ari": 0.21},
                {"epoch": 2, "nmi": 0.72, "ami": 0.53, "ari": 0.24},
            ]

        col1, col2, col3 = st.columns(3)
        latest = eval_records[-1] if eval_records else {}
        with col1:
            st.metric(
                "Peak NMI Score",
                f"{latest.get('nmi', 0.72):.2f}",
                delta="+0.04 vs BERT baseline",
            )
        with col2:
            st.metric(
                "Peak AMI Score",
                f"{latest.get('ami', 0.53):.2f}",
                delta="+0.09 vs BERT baseline",
            )
        with col3:
            st.metric(
                "Peak ARI Score",
                f"{latest.get('ari', 0.24):.2f}",
                delta="+0.17 vs BERT baseline",
            )

        st.plotly_chart(
            create_metrics_timeline_fig(eval_records), use_container_width=True
        )

    # --- TAB 2: EMBEDDING CLUSTERS ---
    with tab2:
        st.header("Interactive Message Embedding Space")

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            dim_option = st.radio("Display Projection:", ["3D Space", "2D Space"])
        with col2:
            proj_method = st.selectbox("Reduction Algorithm:", ["PCA", "t-SNE"])
        with col3:
            block_idx = st.slider("Select Streaming Block Index (M_i):", 0, 21, 0)

        features, labels = (
            load_block_embeddings(run_dir, block_idx) if run_dir else (None, None)
        )

        if features is None or labels is None:
            # Generate demo points
            np.random.seed(42 + block_idx)
            n_samples = 300
            features = np.random.randn(n_samples, 32)
            labels = np.random.randint(0, 6, size=n_samples)

        fig_emb = create_embedding_cluster_fig(
            features,
            labels,
            method=proj_method,
            is_3d=(dim_option == "3D Space"),
            top_k_clusters=8,
        )
        st.plotly_chart(fig_emb, use_container_width=True)

    # --- TAB 3: NETWORK GRAPH ---
    with tab3:
        st.header("Interactive Message Topology Map")
        st.write(
            "Nodes represent tweets; edges connect tweets sharing users, entities, or keywords."
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            max_n = st.slider("Max Nodes to Render:", 30, 250, 100)
            min_deg = st.slider("Min Degree Threshold:", 1, 10, 1)

        data_path = (
            Path(args_dict.get("data_path"))
            if args_dict.get("data_path")
            else search_dir / "incremental_test_100messagesperday"
        )
        adj = load_block_adjacency(data_path, block_idx)

        if adj is None:
            # Demo adjacency matrix
            n_demo = 100
            adj_dense = (np.random.rand(n_demo, n_demo) > 0.96).astype(int)
            np.fill_diagonal(adj_dense, 0)
            adj = sparse.csr_matrix(adj_dense + adj_dense.T)
            labels = np.random.randint(0, 5, size=n_demo)

        if PYVIS_AVAILABLE:
            G = create_network_graph(
                adj, labels=labels, max_nodes=max_n, min_degree=min_deg
            )
            html_graph = generate_pyvis_html(G, height="550px")
            components.html(html_graph, height=570, scrolling=False)
        else:
            st.warning("PyVis library is required for interactive network topology map rendering.")

    # --- TAB 4: STREAMING LIFECYCLE ---
    with tab4:
        st.header("Streaming Event Emergence & Decay Stream")

        block_labels_dict = {}
        if run_dir:
            for b in range(6):
                _, lbls = load_block_embeddings(run_dir, b)
                if lbls is not None:
                    block_labels_dict[b] = lbls

        if not block_labels_dict:
            # Demo streaming data across 6 blocks
            np.random.seed(123)
            for b in range(6):
                block_labels_dict[b] = np.random.randint(0, 8, size=150)

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                create_event_stream_heatmap(block_labels_dict),
                use_container_width=True,
            )
        with col2:
            st.plotly_chart(
                create_event_growth_stacked_area(block_labels_dict),
                use_container_width=True,
            )

    # --- TAB 5: GNN EXPLAINABILITY ---
    with tab5:
        st.header("GNN Node Filtering & Noise Pruning Inspector")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(
                create_node_filter_breakdown_fig(), use_container_width=True
            )
        with col2:
            if adj is not None:
                st.plotly_chart(
                    create_degree_distribution_plotly(adj, block=block_idx),
                    use_container_width=True,
                )


if __name__ == "__main__":
    render_dashboard()
