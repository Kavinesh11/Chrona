"""
Core analysis for the DistilBERT-GNN case study, run on REAL Twitter blocks.

For each incremental block i it reads:
  <data_path>/<i>/s_bool_A_tid_tid.npz   homogeneous message graph (scipy sparse)
  <data_path>/<i>/labels.npy             ground-truth event id per message node

and, when a global DistilBERT feature array + per-block index map are available,
also the content-space features for those same messages.

It reports, per block, three ways of recovering events and scores each against
the ground-truth event labels with NMI / AMI / ARI:

  * STRUCTURE-ONLY  : Louvain community detection on the message graph
                       (nx.community.louvain_communities) — no text, no learning.
  * CONTENT-ONLY    : KMeans on the DistilBERT message embeddings
                       (k = #ground-truth events in the block) — no graph.
  * (DistilBERT-GNN : the learned model combines both; its scores come from a
                       main.py training run's evaluate.txt and are not recomputed
                       here — this script measures the two classical baselines
                       the paper's method is meant to beat.)

Outputs:
  <out>/analysis.json       full per-block + aggregate numbers
  <out>/baseline_compare.png  grouped bars: NMI/AMI/ARI, structure vs content, per block
  prints a summary table to stdout
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy import sparse


def _import_sklearn():
    from sklearn.cluster import KMeans
    from sklearn import metrics
    return KMeans, metrics


def louvain_scores(adjacency, labels, metrics):
    import networkx as nx
    G = nx.from_scipy_sparse_array(adjacency)
    communities = nx.algorithms.community.louvain_communities(G, seed=42)
    pred = np.empty(G.number_of_nodes(), dtype=int)
    idx = {n: k for k, n in enumerate(G.nodes())}
    for cid, members in enumerate(communities):
        for n in members:
            pred[idx[n]] = cid
    mod = nx.algorithms.community.modularity(G, communities)
    return {
        "communities": len(communities),
        "modularity": float(mod),
        "nmi": float(metrics.normalized_mutual_info_score(labels, pred)),
        "ami": float(metrics.adjusted_mutual_info_score(labels, pred)),
        "ari": float(metrics.adjusted_rand_score(labels, pred)),
    }


def kmeans_scores(features, labels, KMeans, metrics):
    k = len(set(labels.tolist()))
    if k < 2 or features.shape[0] < k:
        return None
    km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(features)
    pred = km.labels_
    return {
        "k": int(k),
        "nmi": float(metrics.normalized_mutual_info_score(labels, pred)),
        "ami": float(metrics.adjusted_mutual_info_score(labels, pred)),
        "ari": float(metrics.adjusted_rand_score(labels, pred)),
    }


def block_features(data_path: Path, block: int):
    """Per-block DistilBERT features saved alongside the graph, if real."""
    fp = data_path / str(block) / "features.npy"
    if not fp.exists():
        return None
    f = np.load(fp)
    # treat an all-zero / near-constant array as a placeholder, not real features
    if f.size == 0 or float(np.abs(f).sum()) < 1e-9:
        return None
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--blocks", type=int, nargs="*", default=None, help="Specific blocks (default: all found)")
    args = ap.parse_args()

    KMeans, metrics = _import_sklearn()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.blocks:
        blocks = args.blocks
    else:
        blocks = sorted(int(p.name) for p in args.data_path.iterdir()
                        if p.name.isdigit() and (p / "s_bool_A_tid_tid.npz").exists())

    results = []
    for b in blocks:
        bd = args.data_path / str(b)
        adj = sparse.load_npz(bd / "s_bool_A_tid_tid.npz")
        labels = np.load(bd / "labels.npy")
        n = adj.shape[0]
        edges = int(adj.nnz // 2)
        density = (2.0 * edges) / (n * (n - 1)) if n > 1 else 0.0
        avg_deg = (2.0 * edges) / n if n > 0 else 0.0

        rec = {
            "block": b,
            "nodes": int(n),
            "edges": edges,
            "density": round(density, 5),
            "avg_degree": round(avg_deg, 3),
            "ground_truth_events": int(len(set(labels.tolist()))),
            "structure_louvain": louvain_scores(adj, labels, metrics),
        }
        feats = block_features(args.data_path, b)
        if feats is not None and feats.shape[0] == n:
            rec["content_kmeans"] = kmeans_scores(feats, labels, KMeans, metrics)
        else:
            rec["content_kmeans"] = None
        results.append(rec)
        s = rec["structure_louvain"]
        c = rec["content_kmeans"]
        cstr = f"content NMI={c['nmi']:.3f}" if c else "content n/a"
        print(f"block {b:2d}: n={n:4d} e={edges:5d} events={rec['ground_truth_events']:3d} | "
              f"Louvain NMI={s['nmi']:.3f} AMI={s['ami']:.3f} ARI={s['ari']:.3f} mod={s['modularity']:.3f} | {cstr}")

    def mean(key_path):
        vals = []
        for r in results:
            d = r
            for k in key_path:
                d = d.get(k) if isinstance(d, dict) else None
                if d is None:
                    break
            if isinstance(d, (int, float)):
                vals.append(d)
        return round(float(np.mean(vals)), 4) if vals else None

    aggregate = {
        "n_blocks": len(results),
        "structure_louvain_mean": {m: mean(["structure_louvain", m]) for m in ("nmi", "ami", "ari", "modularity")},
        "content_kmeans_mean": {m: mean(["content_kmeans", m]) for m in ("nmi", "ami", "ari")},
    }
    out = {"data_path": str(args.data_path), "aggregate": aggregate, "blocks": results}
    (args.out / "analysis.json").write_text(json.dumps(out, indent=2))
    print("\nAGGREGATE (mean over blocks):")
    print("  structure-only Louvain:", aggregate["structure_louvain_mean"])
    print("  content-only  KMeans :", aggregate["content_kmeans_mean"])

    # ---- comparison figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["block"] for r in results]
        s_nmi = [r["structure_louvain"]["nmi"] for r in results]
        c_nmi = [r["content_kmeans"]["nmi"] if r["content_kmeans"] else np.nan for r in results]
        s_ari = [r["structure_louvain"]["ari"] for r in results]
        c_ari = [r["content_kmeans"]["ari"] if r["content_kmeans"] else np.nan for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        w = 0.38
        x = np.arange(len(xs))
        ax1.bar(x - w / 2, s_nmi, w, label="Structure (Louvain)", color="#3FD6D1")
        ax1.bar(x + w / 2, c_nmi, w, label="Content (KMeans/DistilBERT)", color="#C9E93A")
        ax1.set_title("NMI vs ground-truth events, per block"); ax1.set_xlabel("block"); ax1.set_ylabel("NMI")
        ax1.set_xticks(x); ax1.set_xticklabels(xs); ax1.legend(); ax1.grid(axis="y", alpha=0.3)

        ax2.bar(x - w / 2, s_ari, w, label="Structure (Louvain)", color="#3FD6D1")
        ax2.bar(x + w / 2, c_ari, w, label="Content (KMeans/DistilBERT)", color="#C9E93A")
        ax2.set_title("ARI vs ground-truth events, per block"); ax2.set_xlabel("block"); ax2.set_ylabel("ARI")
        ax2.set_xticks(x); ax2.set_xticklabels(xs); ax2.legend(); ax2.grid(axis="y", alpha=0.3)

        fig.suptitle("Structure-only vs Content-only event recovery on real Twitter blocks", fontsize=13)
        fig.tight_layout()
        fig.savefig(args.out / "baseline_compare.png", dpi=130)
        print(f"\nwrote {args.out / 'baseline_compare.png'}")
    except Exception as e:
        print(f"[warn] figure skipped: {e}")


if __name__ == "__main__":
    main()
