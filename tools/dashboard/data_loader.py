from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse


def find_all_runs(root_path: Union[str, Path]) -> List[Path]:
    """Scans root_path for all embeddings_* run directories."""
    root = Path(root_path)
    if not root.exists():
        return []
    
    runs = []
    # Check if root itself is a run dir
    if root.name.startswith("embeddings_") and (root / "args.txt").exists():
        runs.append(root)

    # Search subdirectories recursively
    for path in root.rglob("embeddings_*"):
        if path.is_dir() and (path / "args.txt").exists():
            runs.append(path)
            
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def load_run_args(run_dir: Path) -> Dict[str, Any]:
    """Loads args.txt configuration dictionary from a run directory."""
    args_file = run_dir / "args.txt"
    if not args_file.exists():
        return {}
    try:
        with open(args_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def load_evaluation_metrics(run_dir: Path) -> List[Dict[str, Any]]:
    """
    Parses evaluate.txt into structured epoch metric records.
    Returns list of dicts with epoch, nmi, ami, ari, loss, etc.
    """
    eval_file = run_dir / "evaluate.txt"
    if not eval_file.exists():
        return []

    records = []
    current: Dict[str, Any] = {}

    with open(eval_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    records.append(current)
                    current = {}
                continue

            if "Epoch" in line or "Block" in line or "Evaluating" in line:
                if current and ("nmi" in current or "validation_nmi" in current):
                    records.append(current)
                    current = {}
                current["header"] = line

            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip().lower().replace(" ", "_")
                val_str = parts[1].strip()
                try:
                    val = float(val_str)
                    current[key] = val
                except ValueError:
                    current[key] = val_str

    if current:
        records.append(current)

    return records


def load_block_embeddings(
    run_dir: Path, block: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Loads features_{block}.tsv and labels_{block}.tsv if available."""
    feat_file = run_dir / f"features_{block}.tsv"
    label_file = run_dir / f"labels_{block}.tsv"

    features = np.loadtxt(feat_file, delimiter="\t") if feat_file.exists() else None
    labels = np.loadtxt(label_file, dtype=int) if label_file.exists() else None

    return features, labels


def load_block_adjacency(
    data_path: Path, block: int
) -> Optional[sparse.csr_matrix]:
    """Loads sparse adjacency matrix s_bool_A_tid_tid.npz for a given block."""
    adj_file = data_path / str(block) / "s_bool_A_tid_tid.npz"
    if not adj_file.exists():
        return None
    try:
        return sparse.load_npz(adj_file)
    except Exception:
        return None
