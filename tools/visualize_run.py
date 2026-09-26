from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA

from run_artifacts import as_path, latest_run_dir, load_evaluate_history

# Fixed categorical order (tab10), assigned by rank of cluster size, never re-cycled
# when the filter/selection changes which clusters are visible.
_CATEGORICAL = plt.get_cmap("tab10").colors
_SEQUENTIAL = "Blues"
_MAX_LEGEND_CLASSES = 9  # 9 distinct classes + "Other" bucket


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate figures (metrics over time, embedding clusters, degree distribution) "
        "for a DistilBERTGNN run or message-graph block, for use in the case-study report/slides."
    )
    parser.add_argument("kind", choices=["metrics", "embeddings", "graph"], help="Which figure to produce.")
    parser.add_argument("run_dir", nargs="?", help="Path to an embeddings_* run directory.")
    parser.add_argument("--root", help="Search this directory for the latest embeddings_* run (used instead of run_dir).")
    parser.add_argument("--block", type=int, default=0, help="Block index for --kind embeddings/graph (default: 0).")
    parser.add_argument("--data-path", help="Root data directory containing block subfolders (required for --kind graph).")
    parser.add_argument("--out", help="Output PNG path (default: <run_dir or data-path>/<kind>_<block>.png).")
    return parser


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return as_path(args.run_dir)
    if args.root:
        return latest_run_dir(args.root)
    raise SystemExit("Provide either run_dir or --root.")


def plot_metrics(run_dir: Path, out_path: Path) -> None:
    """Two small multiples (validation, test), one axis each, NMI/AMI/ARI as a fixed-order categorical triad."""
    history = load_evaluate_history(run_dir)
    if not history:
        raise SystemExit(f"No evaluate.txt history found in {run_dir}")

    series_colors = {"nmi": _CATEGORICAL[0], "ami": _CATEGORICAL[1], "ari": _CATEGORICAL[2]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, mode in zip(axes, ("validation", "test")):
        xs = [i for i, entry in enumerate(history) if f"{mode}_nmi" in entry or f"with_isolated_nodes_{mode}_nmi" in entry]
        for metric, color in series_colors.items():
            key_plain = f"{mode}_{metric}"
            key_scoped = f"with_isolated_nodes_{mode}_{metric}"
            ys = [entry.get(key_scoped, entry.get(key_plain)) for entry in history]
            points = [(i, y) for i, y in enumerate(ys) if y is not None]
            if not points:
                continue
            px, py = zip(*points)
            ax.plot(px, py, marker="o", markersize=4, linewidth=2, color=color, label=metric.upper())
        ax.set_title(f"{mode.capitalize()} metrics")
        ax.set_xlabel("Log entry (chronological)")
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linewidth=0.5, alpha=0.3)

    axes[0].set_ylabel("Score (0-1)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"Clustering quality over the incremental run — {run_dir.name}", y=1.12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_embeddings(run_dir: Path, block: int, out_path: Path) -> None:
    """PCA projection of message embeddings, colored by ground-truth event (identity = categorical)."""
    features_path = run_dir / f"features_{block}.tsv"
    labels_path = run_dir / f"labels_{block}.tsv"
    if not features_path.exists() or not labels_path.exists():
        raise SystemExit(f"Expected {features_path.name} and {labels_path.name} in {run_dir}")

    features = np.loadtxt(features_path, delimiter="\t")
    labels = np.loadtxt(labels_path, dtype=int)

    coords = PCA(n_components=2, random_state=0).fit_transform(features)

    counts = np.bincount(labels) if labels.min() >= 0 else None
    top_labels = (
        [lbl for lbl, _ in sorted(enumerate(counts), key=lambda kv: -kv[1])[:_MAX_LEGEND_CLASSES]]
        if counts is not None
        else sorted(set(labels))[:_MAX_LEGEND_CLASSES]
    )

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for rank, lbl in enumerate(top_labels):
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], s=14, color=_CATEGORICAL[rank % 10], label=f"Event {lbl}", alpha=0.85)
    other_mask = ~np.isin(labels, top_labels)
    if other_mask.any():
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1], s=10, color="#999999", label="Other", alpha=0.5)

    ax.set_title(f"Message embeddings by event cluster — block {block}")
    ax.set_xlabel("PCA component 1")
    ax.set_ylabel("PCA component 2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="best", fontsize=8, frameon=False, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_degree_distribution(data_path: Path, block: int, out_path: Path) -> None:
    """Sequential (single-hue) histogram of node degree — a magnitude, not an identity, encoding."""
    adj_path = data_path / str(block) / "s_bool_A_tid_tid.npz"
    if not adj_path.exists():
        raise SystemExit(f"Adjacency matrix not found: {adj_path}")

    adjacency = sparse.load_npz(adj_path)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    n_bins = min(50, max(10, int(degrees.max()) or 10))
    ax.hist(degrees, bins=n_bins, color=plt.get_cmap(_SEQUENTIAL)(0.65), edgecolor="white", linewidth=0.4)
    ax.set_title(f"Message-graph degree distribution — block {block}")
    ax.set_xlabel("Node degree")
    ax.set_ylabel("Number of messages")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.kind == "graph":
        if not args.data_path:
            parser.error("--data-path is required for --kind graph")
        data_path = as_path(args.data_path)
        out_path = as_path(args.out) if args.out else data_path / f"degree_distribution_{args.block}.png"
        plot_degree_distribution(data_path, args.block, out_path)
        return

    run_dir = _resolve_run_dir(args)
    if args.kind == "metrics":
        out_path = as_path(args.out) if args.out else run_dir / "metrics_over_time.png"
        plot_metrics(run_dir, out_path)
    else:
        out_path = as_path(args.out) if args.out else run_dir / f"embedding_clusters_{args.block}.png"
        plot_embeddings(run_dir, args.block, out_path)


if __name__ == "__main__":
    main()
