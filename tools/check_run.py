from __future__ import annotations

import argparse
from pathlib import Path

from run_artifacts import latest_run_dir, summarize_run


REQUIRED_FILES = ["args.txt", "evaluate.txt", "graph_statistics.txt"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the expected files in a DistilBERTGNN run directory.")
    parser.add_argument("run_dir", nargs="?", help="Path to an embeddings_* run directory.")
    parser.add_argument("--root", help="Search this directory for the latest embeddings_* run.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.root:
        run_dir = latest_run_dir(args.root)
    else:
        parser.error("Provide either run_dir or --root.")

    summary = summarize_run(run_dir)
    missing = [file_name for file_name in REQUIRED_FILES if not (run_dir / file_name).exists()]

    print(f"Run: {summary['run_name']}")
    if missing:
        print("Missing files:")
        for file_name in missing:
            print(f"- {file_name}")
    else:
        print("All required files are present.")

    if summary["latest"]:
        print("Latest metrics found in evaluate.txt.")
    else:
        print("No evaluation history found yet.")


if __name__ == "__main__":
    main()
