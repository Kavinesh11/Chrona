#!/usr/bin/env python3
"""
CLI launcher for the Chrona Interactive Visualization Dashboard.
Usage:
    python tools/run_dashboard.py [--port 8501] [--root DistilBERTGNN]
"""

from __future__ import annotations

import argparse
import os
import sys
import subprocess
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_APP_PATH = _HERE / "dashboard" / "app.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the interactive Streamlit dashboard for Chrona (DistilBERT-GNN)."
    )
    parser.add_argument(
        "--port", type=int, default=8501, help="Port to run Streamlit server on (default: 8501)."
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host address (default: localhost)."
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not auto-open browser on launch."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not _APP_PATH.exists():
        print(f"Error: Dashboard app entry point not found at {_APP_PATH}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_APP_PATH),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]
    if args.no_browser:
        cmd.append("--server.headless=true")

    print(f"Launching Chrona Interactive Dashboard on http://{args.host}:{args.port}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")

    except Exception as e:
        print(f"Error launching dashboard: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
