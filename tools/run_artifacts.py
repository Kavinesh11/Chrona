from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_EPOCH_RE = re.compile(r"^Epoch\s+(?P<epoch>\d+)\s*$")
_COUNT_RE = re.compile(r"^Number of (?P<mode>validation|test) tweets:\s+(?P<value>\d+)\s*$")
_CLASS_RE = re.compile(
    r"^Number of classes covered by (?P<mode>validation|test) tweets:\s+(?P<value>\d+)\s*$"
)
_METRIC_RE = re.compile(r"^(?P<mode>validation|test) (?P<metric>NMI|AMI|ARI):\s+(?P<value>[-+0-9.eE]+)\s*$")


def as_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def latest_run_dir(root: str | Path) -> Path:
    root_path = as_path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_path}")

    candidates = [path for path in root_path.iterdir() if path.is_dir() and path.name.startswith("embeddings_")]
    if not candidates:
        raise FileNotFoundError(f"No embeddings_* directories found in {root_path}")

    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def load_json_if_exists(path: str | Path) -> dict[str, Any] | None:
    file_path = as_path(path)
    if not file_path.exists():
        return None
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _coerce_scalar(value: str) -> Any:
    value = value.strip()
    if re.fullmatch(r"[-+]?\d+", value):
        try:
            return int(value)
        except ValueError:
            return value
    if re.fullmatch(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", value) or re.fullmatch(r"[-+]?\d+(?:[eE][-+]?\d+)", value):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def load_graph_statistics(run_dir: str | Path) -> dict[str, Any]:
    stats_path = as_path(run_dir) / "graph_statistics.txt"
    if not stats_path.exists():
        return {}

    text = stats_path.read_text(encoding="utf-8")
    stats: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("We have ") and stripped.endswith(" nodes."):
            stats["num_nodes"] = _coerce_scalar(stripped.removeprefix("We have ").removesuffix(" nodes."))
        elif stripped.startswith("We have ") and stripped.endswith(" in-edges."):
            stats["num_in_edges"] = _coerce_scalar(stripped.removeprefix("We have ").removesuffix(" in-edges."))
        elif stripped.startswith("Average degree:"):
            stats["average_degree"] = _coerce_scalar(stripped.split(":", 1)[1])
        elif stripped.startswith("Number of isolated nodes:"):
            stats["num_isolated_nodes"] = _coerce_scalar(stripped.split(":", 1)[1])
    return stats


def parse_evaluate_text(text: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    section = "with_isolated_nodes"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        epoch_match = _EPOCH_RE.match(line)
        if epoch_match:
            if current is not None:
                blocks.append(current)
            current = {"epoch": int(epoch_match.group("epoch"))}
            section = "with_isolated_nodes"
            continue

        if current is None:
            continue

        if line == "Without isolated nodes:":
            section = "without_isolated_nodes"
            continue

        count_match = _COUNT_RE.match(line)
        if count_match:
            key = f"{section}_{count_match.group('mode')}_tweets"
            current[key] = int(count_match.group("value"))
            continue

        class_match = _CLASS_RE.match(line)
        if class_match:
            key = f"{section}_{class_match.group('mode')}_classes"
            current[key] = int(class_match.group("value"))
            continue

        metric_match = _METRIC_RE.match(line)
        if metric_match:
            key = f"{section}_{metric_match.group('mode')}_{metric_match.group('metric').lower()}"
            current[key] = float(metric_match.group("value"))

    if current is not None:
        blocks.append(current)

    return blocks


def load_evaluate_history(run_dir: str | Path) -> list[dict[str, Any]]:
    evaluate_path = as_path(run_dir) / "evaluate.txt"
    if not evaluate_path.exists():
        return []
    return parse_evaluate_text(evaluate_path.read_text(encoding="utf-8"))


def latest_metrics(run_dir: str | Path) -> dict[str, Any]:
    history = load_evaluate_history(run_dir)
    return history[-1] if history else {}


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    run_path = as_path(run_dir)
    args = load_json_if_exists(run_path / "args.txt") or {}
    graph_stats = load_graph_statistics(run_path)
    history = load_evaluate_history(run_path)

    summary = {
        "run_dir": str(run_path),
        "run_name": run_path.name,
        "exists": run_path.exists(),
        "files": sorted(path.name for path in run_path.iterdir()) if run_path.exists() else [],
        "args": args,
        "graph_statistics": graph_stats,
        "history": history,
        "latest": history[-1] if history else {},
    }
    return summary


def extract_metric(summary: dict[str, Any], metric_name: str, section: str = "with_isolated_nodes") -> Any:
    latest = summary.get("latest", {})
    for mode in ("validation", "test"):
        key = f"{section}_{mode}_{metric_name}"
        if key in latest:
            return latest[key]
    return None


def pick_metrics(summary: dict[str, Any], section: str = "with_isolated_nodes") -> dict[str, Any]:
    latest = summary.get("latest", {})
    return {
        "epoch": latest.get("epoch"),
        "validation_tweets": latest.get(f"{section}_validation_tweets"),
        "validation_classes": latest.get(f"{section}_validation_classes"),
        "validation_nmi": latest.get(f"{section}_validation_nmi"),
        "validation_ami": latest.get(f"{section}_validation_ami"),
        "validation_ari": latest.get(f"{section}_validation_ari"),
        "test_tweets": latest.get(f"{section}_test_tweets"),
        "test_classes": latest.get(f"{section}_test_classes"),
        "test_nmi": latest.get(f"{section}_test_nmi"),
        "test_ami": latest.get(f"{section}_test_ami"),
        "test_ari": latest.get(f"{section}_test_ari"),
    }
