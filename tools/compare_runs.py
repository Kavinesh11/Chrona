from __future__ import annotations

import argparse
from pathlib import Path

from run_artifacts import pick_metrics, summarize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare several DistilBERTGNN run directories.")
    parser.add_argument("run_dirs", nargs="+", help="Paths to embeddings_* run directories.")
    parser.add_argument(
        "--section",
        default="with_isolated_nodes",
        choices=["with_isolated_nodes", "without_isolated_nodes"],
        help="Which evaluation section to compare.",
    )
    return parser


def _format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summaries = [summarize_run(Path(run_dir)) for run_dir in args.run_dirs]
    rows = [
        (summary["run_name"], pick_metrics(summary, section=args.section))
        for summary in summaries
    ]

    headers = ["run", "epoch", "validation_nmi", "validation_ami", "validation_ari", "test_nmi", "test_ami", "test_ari"]
    widths = {header: len(header) for header in headers}
    for run_name, metrics in rows:
        widths["run"] = max(widths["run"], len(run_name))
        for key in headers[1:]:
            widths[key] = max(widths[key], len(_format_value(metrics.get(key))))

    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    print(header_line)
    print("  ".join("-" * widths[header] for header in headers))
    for run_name, metrics in rows:
        values = [run_name, metrics.get("epoch"), metrics.get("validation_nmi"), metrics.get("validation_ami"), metrics.get("validation_ari"), metrics.get("test_nmi"), metrics.get("test_ami"), metrics.get("test_ari")]
        print("  ".join(_format_value(value).ljust(widths[header]) for value, header in zip(values, headers)))


if __name__ == "__main__":
    main()
