from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_artifacts import latest_run_dir, summarize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a DistilBERTGNN run directory.")
    parser.add_argument("run_dir", nargs="?", help="Path to an embeddings_* run directory.")
    parser.add_argument("--root", help="Search this directory for the latest embeddings_* run.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
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
    indent = 2 if args.pretty else None
    print(json.dumps(summary, indent=indent, sort_keys=True))


if __name__ == "__main__":
    main()
