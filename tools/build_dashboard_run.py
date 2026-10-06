"""
Turn the REAL per-block pipeline output (features.npy + labels.npy + graph)
into an embeddings_* run directory the Streamlit dashboard can read, so the
dashboard's metrics/embedding/topology tabs all show real data.

The metrics written here are the DistilBERT-feature clustering baseline
(KMeans on the real DistilBERT embeddings vs ground-truth events) — i.e. the
"content-only" number. They are NOT the full DistilBERT-GNN model metrics,
which require a main.py training run (blocked on this machine by the dgl
binary mismatch). The run is named accordingly so it is self-documenting.

Writes into <data_path>/<run_name>/:
  args.txt, evaluate.txt, features_<i>.tsv, labels_<i>.tsv
and ensures each block subdir (already holding s_bool_A_tid_tid.npz) is intact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-path", required=True, type=Path)
    ap.add_argument("--run-name", default="embeddings_distilbert_baseline")
    ap.add_argument("--max-dim", type=int, default=64, help="PCA-reduce features to this many dims for the .tsv (viz speed)")
    args = ap.parse_args()

    run_dir = args.data_path / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    blocks = sorted(int(p.name) for p in args.data_path.iterdir()
                    if p.name.isdigit() and (p / "features.npy").exists() and (p / "labels.npy").exists())
    if not blocks:
        raise SystemExit(f"No blocks with features.npy+labels.npy under {args.data_path}")

    args_txt = {
        "filter_method": "none",
        "remove_obsolete": 2,
        "window_size": 3,
        "num_heads": 4,
        "hidden_dim": 8,
        "note": "DistilBERT-feature clustering baseline (content-only); not full GNN metrics",
        "data_path": str(args.data_path),
    }
    (run_dir / "args.txt").write_text(json.dumps(args_txt, indent=2))

    eval_lines = []
    for b in blocks:
        feats = np.load(args.data_path / str(b) / "features.npy").astype(np.float32)
        labels = np.load(args.data_path / str(b) / "labels.npy")
        # dimensionality reduction for the .tsv (PCA) — keeps viz fast, preserves structure
        f_viz = feats
        if feats.shape[1] > args.max_dim and feats.shape[0] > args.max_dim:
            from sklearn.decomposition import PCA
            f_viz = PCA(n_components=args.max_dim, random_state=0).fit_transform(feats)
        np.savetxt(run_dir / f"features_{b}.tsv", f_viz, delimiter="\t", fmt="%.5f")
        np.savetxt(run_dir / f"labels_{b}.tsv", labels, fmt="%d")

        k = len(set(labels.tolist()))
        eval_lines.append(f"Evaluating block {b}...")
        if k >= 2 and feats.shape[0] >= k:
            pred = KMeans(n_clusters=k, random_state=0, n_init=10).fit(feats).labels_
            nmi = metrics.normalized_mutual_info_score(labels, pred)
            ami = metrics.adjusted_mutual_info_score(labels, pred)
            ari = metrics.adjusted_rand_score(labels, pred)
            eval_lines += [f"test_nmi: {nmi:.3f}", f"test_ami: {ami:.3f}", f"test_ari: {ari:.3f}", ""]
        else:
            eval_lines += ["test_nmi: 0.000", "test_ami: 0.000", "test_ari: 0.000", ""]

    (run_dir / "evaluate.txt").write_text("\n".join(eval_lines))
    print(f"wrote run dir: {run_dir}")
    print(f"  blocks: {blocks}")
    print(f"  files: args.txt, evaluate.txt, features_<i>.tsv, labels_<i>.tsv")


if __name__ == "__main__":
    main()
