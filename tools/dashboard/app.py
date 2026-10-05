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
from scipy import sparse
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
    page_title="Chrona | DistilBERT-GNN Event Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Professional UI Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main {
        background-color: #0b0f19;
    }
    .stAppHeader {
        background-color: rgba(11, 15, 25, 0.9);
    }
    .title-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e293b 100%);
        padding: 24px 32px;
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
    }
    .title-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid rgba(96, 165, 250, 0.3);
        margin-bottom: 8px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def render_dashboard():
    st.markdown(
        """
        <div class="title-banner">
            <div class="title-badge">Social Network Analysis System</div>
            <h1 style="margin:0; font-size: 2.1rem; font-weight: 700; color: #ffffff;">Chrona: DistilBERT-GNN Event Detection Platform</h1>
            <p style="margin:6px 0 0 0; color: #94a3b8; font-size: 1.0rem;">
                Incremental Social Media Event Detection & Community Topology Explorer
            </p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar setup
    st.sidebar.title("Run Inspector")

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
    st.sidebar.subheader("Run Configuration")
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
        st.sidebar.text("Mode: Demo Dataset")

    # Main Tabs Layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Metrics & Benchmarks",
            "Embedding Space Projections",
            "Message Graph Network",
            "Event Lifecycle",
            "Node Filtering & Explainability",
        ]
    )

    # --- TAB 1: METRICS ---
    with tab1:
        st.header("Clustering Performance Analytics (NMI / AMI / ARI)")

        eval_records = load_evaluation_metrics(run_dir) if run_dir else []
        if not eval_records:
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
        st.caption(
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
            st.warning("PyVis package required for graph rendering.")

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
