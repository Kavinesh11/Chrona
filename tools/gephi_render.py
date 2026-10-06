"""
Drive a locally-running Gephi AI API server (MattArtzAnthro/gephi-ai plugin,
default http://127.0.0.1:8081) to turn a .gexf file into publication-quality
PNGs: import -> degree -> modularity -> size-by-degree -> ForceAtlas2 layout ->
color-by-partition -> export one PNG per partition column.

Requires Gephi Desktop running with the "Gephi AI Server" started (the plugin's
HTTP API). Nothing here modifies the training pipeline — it only visualizes a
graph that export_gephi.py already produced.

Usage:
    python gephi_render.py --gexf block0.gexf --out-dir figs \
        --columns ground_truth_event louvain_community modularity_class
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request


class GephiAPI:
    def __init__(self, base="http://127.0.0.1:8081"):
        self.base = base.rstrip("/")

    def _call(self, method, path, body=None):
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        return self._call("GET", path)

    def post(self, path, body=None):
        return self._call("POST", path, body or {})


def require(resp, label):
    if not resp.get("success", False):
        print(f"[gephi_render] {label} FAILED: {resp.get('error', resp)}", file=sys.stderr)
        sys.exit(1)
    return resp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gexf", required=True, help="Absolute path to the .gexf to import")
    ap.add_argument("--out-dir", default=".", help="Where to write PNGs")
    ap.add_argument("--columns", nargs="+", default=["modularity_class"],
                    help="Partition columns to color by (one PNG each)")
    ap.add_argument("--base", default="http://127.0.0.1:8081")
    ap.add_argument("--resolution", type=float, default=1.0, help="Modularity resolution")
    ap.add_argument("--iterations", type=int, default=600, help="ForceAtlas2 iterations")
    ap.add_argument("--width", type=int, default=2000)
    ap.add_argument("--height", type=int, default=1500)
    ap.add_argument("--min-size", type=float, default=6.0)
    ap.add_argument("--max-size", type=float, default=50.0)
    ap.add_argument("--prefix", default="gephi", help="PNG filename prefix")
    args = ap.parse_args()

    gexf = os.path.abspath(args.gexf)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    g = GephiAPI(args.base)

    health = g.get("/health")
    require(health, "health")
    print(f"Gephi AI API {health.get('version')} (Gephi {health.get('gephi_version')})")

    require(g.post("/project/new", {"name": f"render_{os.path.basename(gexf)}"}), "project/new")
    imp = require(g.post("/import/gexf", {"file": gexf, "mode": "new_workspace"}), "import/gexf")
    stats = g.get("/graph/stats")
    n = stats.get("nodes") or stats.get("node_count")
    e = stats.get("edges") or stats.get("edge_count")
    print(f"imported {os.path.basename(gexf)}: nodes={n} edges={e}")

    deg = require(g.post("/statistics/degree", {}), "statistics/degree")
    print(f"  average degree = {deg.get('average_degree')}")
    mod = require(g.post("/statistics/modularity", {"resolution": args.resolution}), "statistics/modularity")
    print(f"  gephi modularity = {mod.get('modularity')}")

    require(g.post("/appearance/ranking/size", {"column": "degree", "min_size": args.min_size, "max_size": args.max_size}),
            "appearance/ranking/size")

    print(f"  running ForceAtlas2 ({args.iterations} iters, sync)...")
    lay = require(g.post("/layout/run", {
        "algorithm": "forceatlas2",
        "iterations": args.iterations,
        "properties": {"scalingRatio": 12.0, "gravity": 1.0, "adjustSizes": True},
        "sync": True,
    }), "layout/run")
    if lay.get("layout_exploded"):
        print("  [warn] layout exploded; re-running with stronger gravity")
        require(g.post("/layout/run", {
            "algorithm": "forceatlas2", "iterations": args.iterations,
            "properties": {"scalingRatio": 2.0, "gravity": 5.0, "strongGravityMode": True, "adjustSizes": True},
            "sync": True,
        }), "layout/run retry")

    written = []
    for col in args.columns:
        cres = g.post("/appearance/partition/color", {"column": col})
        if not cres.get("success", False):
            print(f"  [skip] color by '{col}': {cres.get('error')}")
            continue
        png = os.path.join(out_dir, f"{args.prefix}_{col}.png")
        require(g.post("/export/png", {"file": png, "width": args.width, "height": args.height}),
                f"export/png ({col})")
        written.append(png)
        print(f"  wrote {png}")

    print("\nDONE. PNGs:")
    for p in written:
        print(" ", p)


if __name__ == "__main__":
    main()
