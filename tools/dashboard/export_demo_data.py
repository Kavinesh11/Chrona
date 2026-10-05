from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

_HERE = Path(__file__).resolve().parent
_DEFAULT_OUT = _HERE.parent.parent / "DistilBERTGNN" / "incremental_test_100messagesperday" / "embeddings_demo"


def generate_synthetic_demo_run(output_dir: Path = _DEFAULT_OUT, n_blocks: int = 5, n_samples: int = 200) -> None:
    """
    Generates synthetic run artifacts (args.txt, evaluate.txt, features_i.tsv, labels_i.tsv, s_bool_A_tid_tid.npz)
    for instant dashboard testing and previewing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating synthetic visualization dataset in: {output_dir}")

    # 1. args.txt
    args = {
        "filter_method": "sentiment",
        "remove_obsolete": 2,
        "window_size": 3,
        "num_heads": 4,
        "hidden_dim": 8,
        "out_dim": 32,
        "n_epochs": 15,
        "patience": 5,
        "is_demo_dataset": True,
    }
    with open(output_dir / "args.txt", "w") as f:
        json.dump(args, f, indent=2)

    # 2. evaluate.txt
    with open(output_dir / "evaluate.txt", "w") as f:
        f.write("Evaluating initial block 0...\n")
        f.write("test_nmi: 0.640\ntest_ami: 0.460\ntest_ari: 0.190\n\n")
        f.write("Evaluating block 1...\n")
        f.write("test_nmi: 0.690\ntest_ami: 0.510\ntest_ari: 0.220\n\n")
        f.write("Evaluating block 2...\n")
        f.write("test_nmi: 0.725\ntest_ami: 0.535\ntest_ari: 0.245\n\n")

    # 3. features_i.tsv and labels_i.tsv for each block
    np.random.seed(42)
    for b in range(n_blocks):
        # Create clustered features around distinct centers
        centers = np.random.randn(6, 32) * 3
        cluster_assignments = np.random.randint(0, 6, size=n_samples)
        features = centers[cluster_assignments] + np.random.randn(n_samples, 32) * 0.5

        np.savetxt(output_dir / f"features_{b}.tsv", features, delimiter="\t", fmt="%.5f")
        np.savetxt(output_dir / f"labels_{b}.tsv", cluster_assignments, fmt="%d")

        # 4. Adjacency matrix for block subfolder
        block_dir = output_dir.parent / str(b)
        block_dir.mkdir(parents=True, exist_ok=True)
        
        adj_dense = (np.random.rand(n_samples, n_samples) > 0.95).astype(float)
        np.fill_diagonal(adj_dense, 0)
        adj_sparse = sparse.csr_matrix(adj_dense + adj_dense.T)
        sparse.save_npz(block_dir / "s_bool_A_tid_tid.npz", adj_sparse)

    print(f"Successfully created synthetic demo dataset with {n_blocks} blocks and {n_samples} tweets per block!")


if __name__ == "__main__":
    generate_synthetic_demo_run()
