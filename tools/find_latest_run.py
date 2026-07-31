from __future__ import annotations

import argparse

from run_artifacts import latest_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print the newest embeddings_* directory under a root path.")
    parser.add_argument("root", help="Directory that contains embeddings_* run folders.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    print(latest_run_dir(args.root))


if __name__ == "__main__":
    main()
