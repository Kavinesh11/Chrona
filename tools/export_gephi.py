"""
Export a block of the DistilBERT-GNN message graph to GEXF for Gephi, and run a
classical (non-learned) NetworkX community-detection baseline on it.

Reads, for a given incremental block `i`, from `<data_path>/<i>/`:
  - s_bool_A_tid_tid.npz   homogeneous message-message adjacency (scipy sparse, boolean)
  - labels.npy             ground-truth event id per node, aligned row-for-row with the
                            adjacency matrix (both written together by custom_message_graph.py)

For each node it attaches:
  - ground_truth_event   : int, the labeled event category
  - degree                : int
  - degree_centrality     : float (nx.degree_centrality)
  - louvain_community     : int, from nx.algorithms.community.louvain_communities
                             (modularity-maximizing partition — a purely graph-structural
                             baseline, computed with no text/embeddings/learning at all)

and prints NMI/AMI/ARI between ground_truth_event and louvain_community, so the classical
SNA baseline can be quoted next to DistilBERT-GNN's learned-representation NMI/AMI/ARI
from evaluate.txt (see tools/summarize_run.py).

Usage (run from tools/, or with tools/ on PYTHONPATH):
    python export_gephi.py --data-path ../DistilBERTGNN/incremental_test_100messagesperday --block 0
    python export_gephi.py --data-path <path> --block 0 --max-nodes 800 --out block0.gexf

`--max-nodes` keeps the export small enough for Gephi to lay out interactively; it keeps
the highest-degree nodes (same top-degree sampling strategy tools/dashboard/graph_viz.py
uses for the live pyvis view), not a random sample, so the structurally interesting part
of the graph is what you see.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn import metrics


def load_block_graph(data_path: Path, block: int) -> tuple[nx.Graph, np.ndarray]:
    block_dir = data_path / str(block)
    adj_path = block_dir / "s_bool_A_tid_tid.npz"
    labels_path = block_dir / "labels.npy"

    if not adj_path.exists():
        raise FileNotFoundError(
            f"{adj_path} not found — run custom_message_graph.py first "
            f"(see CLAUDE.md pipeline order)."
        )
    if not labels_path.exists():
        raise FileNotFoundError(f"{labels_path} not found next to {adj_path}.")

    adjacency = sparse.load_npz(adj_path)
    labels = np.load(labels_path)

    G = nx.from_scipy_sparse_array(adjacency)
    if G.number_of_nodes() != len(labels):
        raise ValueError(
            f"Node count ({G.number_of_nodes()}) != label count ({len(labels)}) "
            f"for block {block} — graph and labels are out of sync."
        )
    return G, labels


def subsample_top_degree(G: nx.Graph, labels: np.ndarray, max_nodes: int) -> tuple[nx.Graph, np.ndarray]:
    if G.number_of_nodes() <= max_nodes:
        return G, labels
    degrees = dict(G.degree())
    keep = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
    sub = G.subgraph(keep).copy()
    # relabel 0..N-1 so node ids stay dense/contiguous for Gephi, keep a backref to the
    # original row index so it's still traceable to features_<i>.tsv / labels_tags_<i>.tsv
    mapping = {orig: new for new, orig in enumerate(sub.nodes())}
    sub_labels = labels[list(sub.nodes())]
    for orig, new in mapping.items():
        sub.nodes[orig]["orig_node_id"] = int(orig)
    sub = nx.relabel_nodes(sub, mapping)
    return sub, sub_labels


def annotate_and_detect_communities(G: nx.Graph, labels: np.ndarray) -> dict:
    nx.set_node_attributes(G, {n: int(labels[n]) for n in G.nodes()}, "ground_truth_event")
    nx.set_node_attributes(G, dict(G.degree()), "degree")
    nx.set_node_attributes(G, nx.degree_centrality(G), "degree_centrality")

    communities = nx.algorithms.community.louvain_communities(G, seed=42)
    louvain_labels = np.empty(G.number_of_nodes(), dtype=int)
    node_index = {n: idx for idx, n in enumerate(G.nodes())}
    for community_id, members in enumerate(communities):
        for n in members:
            louvain_labels[node_index[n]] = community_id
    nx.set_node_attributes(G, {n: int(louvain_labels[node_index[n]]) for n in G.nodes()}, "louvain_community")

    scores = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "num_louvain_communities": len(communities),
        "num_ground_truth_events": len(set(labels.tolist())),
        "modularity": nx.algorithms.community.modularity(G, communities),
        "nmi": metrics.normalized_mutual_info_score(labels, louvain_labels),
        "ami": metrics.adjusted_mutual_info_score(labels, louvain_labels),
        "ari": metrics.adjusted_rand_score(labels, louvain_labels),
    }
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True, type=Path, help="Root data dir (same as args.data_path)")
    parser.add_argument("--block", required=True, type=int, help="Incremental block index i")
    parser.add_argument("--max-nodes", type=int, default=None, help="Keep only the top-N nodes by degree")
    parser.add_argument("--out", type=Path, default=None, help="Output .gexf path (default: block<i>.gexf)")
    args = parser.parse_args()

    G, labels = load_block_graph(args.data_path, args.block)
    if args.max_nodes:
        G, labels = subsample_top_degree(G, labels, args.max_nodes)

    scores = annotate_and_detect_communities(G, labels)

    out_path = args.out or Path(f"block{args.block}.gexf")
    nx.write_gexf(G, out_path)

    print(f"Wrote {out_path}")
    print(f"  nodes={scores['num_nodes']}  edges={scores['num_edges']}")
    print(f"  ground-truth events={scores['num_ground_truth_events']}  "
          f"louvain communities={scores['num_louvain_communities']}  "
          f"modularity={scores['modularity']:.4f}")
    print(f"  Louvain vs ground truth -> NMI={scores['nmi']:.4f}  "
          f"AMI={scores['ami']:.4f}  ARI={scores['ari']:.4f}")
    print("  (compare these against DistilBERT-GNN's NMI/AMI/ARI in evaluate.txt — "
          "see tools/summarize_run.py — to quantify what learning adds over a "
          "structure-only baseline)")


if __name__ == "__main__":
    main()
