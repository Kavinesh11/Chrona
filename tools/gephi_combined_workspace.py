"""
Build a COMBINED GEXF with both block 0 and block 5 in ONE graph, then drive
Gephi's AI Server API to lay out, run statistics, color by multiple partitions,
and export publication-quality PNGs — all automated, no manual clicking.

The two blocks are placed in the same workspace as disconnected components;
ForceAtlas2 naturally separates them spatially.  Each node carries a `block`
attribute (0 or 5) plus ground_truth_event, louvain_community, degree, and
degree_centrality.

Usage:
    1. Open Gephi Desktop and start the AI Server plugin (port 8081).
    2. Run this script:
       python tools/gephi_combined_workspace.py

    It will:
      a) Build a merged GEXF → real_analysis_outputs/combined_block0_block5.gexf
      b) Import it into Gephi
      c) Run Degree, Modularity, Eigenvector Centrality, Avg Path Length statistics
      d) Size nodes by degree, run ForceAtlas2 layout
      e) Export PNGs colored by: block, ground_truth_event, louvain_community,
         modularity_class
      f) Save all PNGs to real_analysis_outputs/gephi/

    Screenshots land in real_analysis_outputs/gephi/ — ready for the deck and
    for demonstrating Gephi tool exploration.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn import metrics


# ─── Gephi API client ───────────────────────────────────────────────

class GephiAPI:
    def __init__(self, base="http://127.0.0.1:8081"):
        self.base = base.rstrip("/")

    def _call(self, method, path, body=None):
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        return self._call("GET", path)

    def post(self, path, body=None):
        return self._call("POST", path, body or {})


def require(resp, label):
    if not resp.get("success", False):
        print(f"[FAIL] {label}: {resp.get('error', resp)}", file=sys.stderr)
        sys.exit(1)
    return resp


# ─── Build combined graph ────────────────────────────────────────────

def load_block(data_path: Path, block: int):
    """Load adjacency + labels for one block and return annotated graph."""
    adj = sparse.load_npz(data_path / str(block) / "s_bool_A_tid_tid.npz")
    labels = np.load(data_path / str(block) / "labels.npy")
    G = nx.from_scipy_sparse_array(adj)

    # Louvain baseline
    communities = nx.algorithms.community.louvain_communities(G, seed=42)
    louvain = {}
    for cid, members in enumerate(communities):
        for n in members:
            louvain[n] = cid

    nx.set_node_attributes(G, {n: int(labels[n]) for n in G.nodes()}, "ground_truth_event")
    nx.set_node_attributes(G, {n: int(louvain[n]) for n in G.nodes()}, "louvain_community")
    nx.set_node_attributes(G, dict(G.degree()), "degree")
    nx.set_node_attributes(G, nx.degree_centrality(G), "degree_centrality")
    nx.set_node_attributes(G, {n: block for n in G.nodes()}, "block")

    mod = nx.algorithms.community.modularity(G, communities)
    nmi = metrics.normalized_mutual_info_score(labels, [louvain[n] for n in range(len(labels))])
    print(f"  Block {block}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, "
          f"modularity={mod:.3f}, NMI={nmi:.3f}")
    return G, labels


def build_combined_gexf(data_path: Path, out_path: Path):
    """Merge block 0 and block 5 into one graph with prefixed node IDs.

    Writes GEXF manually with viz:position so the two blocks start far
    apart (block 0 on the left, block 5 on the right).  ForceAtlas2 then
    refines positions *within* each component without merging them.
    """
    import random
    import xml.etree.ElementTree as ET

    print("Loading blocks...")
    G0, _ = load_block(data_path, 0)
    G5, _ = load_block(data_path, 5)

    # Prefix node IDs
    G0 = nx.relabel_nodes(G0, {n: f"b0_{n}" for n in G0.nodes()})
    G5 = nx.relabel_nodes(G5, {n: f"b5_{n}" for n in G5.nodes()})

    combined = nx.compose(G0, G5)

    # Assign initial positions: block 0 centered at (-2000, 0),
    # block 5 centered at (+2000, 0), with random jitter.
    rng = random.Random(42)
    for n in combined.nodes():
        blk = combined.nodes[n]["block"]
        cx = -2000.0 if blk == 0 else 2000.0
        combined.nodes[n]["viz_x"] = cx + rng.gauss(0, 300)
        combined.nodes[n]["viz_y"] = rng.gauss(0, 300)

    # Write GEXF with viz:position namespace
    VIZ = "http://www.gexf.net/1.2draft/viz"
    GEXF = "http://www.gexf.net/1.2draft"
    XSI = "http://www.w3.org/2001/XMLSchema-instance"

    root = ET.Element("gexf", {
        "xmlns": GEXF,
        "xmlns:viz": VIZ,
        "xmlns:xsi": XSI,
        "version": "1.2",
    })
    meta = ET.SubElement(root, "meta", {"lastmodifieddate": "2026-10-07"})
    ET.SubElement(meta, "creator").text = "gephi_combined_workspace.py"

    graph = ET.SubElement(root, "graph", {
        "defaultedgetype": "undirected", "mode": "static",
    })

    # Node attributes
    attrs_el = ET.SubElement(graph, "attributes", {"mode": "static", "class": "node"})
    attr_defs = [
        ("0", "ground_truth_event", "long"),
        ("1", "degree", "long"),
        ("2", "degree_centrality", "double"),
        ("3", "louvain_community", "long"),
        ("4", "block", "long"),
    ]
    for aid, title, atype in attr_defs:
        ET.SubElement(attrs_el, "attribute", {"id": aid, "title": title, "type": atype})

    nodes_el = ET.SubElement(graph, "nodes")
    for n in combined.nodes():
        d = combined.nodes[n]
        node_el = ET.SubElement(nodes_el, "node", {"id": str(n), "label": str(n)})
        avs = ET.SubElement(node_el, "attvalues")
        ET.SubElement(avs, "attvalue", {"for": "0", "value": str(d["ground_truth_event"])})
        ET.SubElement(avs, "attvalue", {"for": "1", "value": str(d["degree"])})
        ET.SubElement(avs, "attvalue", {"for": "2", "value": str(d["degree_centrality"])})
        ET.SubElement(avs, "attvalue", {"for": "3", "value": str(d["louvain_community"])})
        ET.SubElement(avs, "attvalue", {"for": "4", "value": str(d["block"])})
        ET.SubElement(node_el, f"{{{VIZ}}}position", {
            "x": f"{d['viz_x']:.1f}",
            "y": f"{d['viz_y']:.1f}",
            "z": "0.0",
        })

    edges_el = ET.SubElement(graph, "edges")
    for eid, (u, v) in enumerate(combined.edges()):
        ET.SubElement(edges_el, "edge", {
            "id": str(eid), "source": str(u), "target": str(v),
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"Combined GEXF: {out_path}")
    print(f"  Total: {combined.number_of_nodes()} nodes, {combined.number_of_edges()} edges")
    print(f"  Block 0 positioned at x=-2000, Block 5 at x=+2000")
    return combined


# ─── Drive Gephi ─────────────────────────────────────────────────────

def drive_gephi(gexf_path: Path, out_dir: Path, base_url: str, iterations: int):
    """Import combined GEXF, run statistics, layout, and export PNGs."""
    g = GephiAPI(base_url)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Health check
    health = g.get("/health")
    require(health, "health")
    print(f"\nGephi AI API {health.get('version')} (Gephi {health.get('gephi_version')})")

    # 2. New project + import
    print("\n== Importing combined GEXF ==")
    require(g.post("/project/new", {"name": "Combined_Block0_Block5"}), "project/new")
    require(g.post("/import/gexf", {
        "file": str(gexf_path.resolve()),
        "mode": "new_workspace",
    }), "import/gexf")
    stats = g.get("/graph/stats")
    print(f"  Imported: {stats.get('nodes', stats.get('node_count'))} nodes, "
          f"{stats.get('edges', stats.get('edge_count'))} edges")

    # 3. Run statistics (demonstrates tool exploration)
    print("\n== Running statistics (tool exploration) ==")

    deg = require(g.post("/statistics/degree", {}), "degree")
    print(f"  [Stat] Average Degree = {deg.get('average_degree')}")

    mod = require(g.post("/statistics/modularity", {"resolution": 1.0}), "modularity")
    print(f"  [Stat] Modularity = {mod.get('modularity')}")

    # Try eigenvector centrality
    try:
        eig = g.post("/statistics/eigenvector_centrality", {})
        if eig.get("success"):
            print(f"  [Stat] Eigenvector Centrality computed")
    except Exception:
        print("  [Stat] Eigenvector centrality endpoint not available (ok)")

    # Try average path length
    try:
        apl = g.post("/statistics/avg_path_length", {})
        if apl.get("success"):
            print(f"  [Stat] Avg Path Length = {apl.get('avg_path_length', 'computed')}")
    except Exception:
        print("  [Stat] Avg path length endpoint not available (ok)")

    # Try connected components
    try:
        cc = g.post("/statistics/connected_components", {})
        if cc.get("success"):
            print(f"  [Stat] Connected Components = {cc.get('count', 'computed')}")
    except Exception:
        print("  [Stat] Connected components endpoint not available (ok)")

    # 4. Size nodes by degree
    print("\n== Appearance: size by degree ==")
    require(g.post("/appearance/ranking/size", {
        "column": "degree",
        "min_size": 4.0,
        "max_size": 50.0,
    }), "size-by-degree")

    # 5. Layout: ForceAtlas2
    print(f"\n== Layout: ForceAtlas2 ({iterations} iterations) ==")
    lay = require(g.post("/layout/run", {
        "algorithm": "forceatlas2",
        "iterations": iterations,
        "properties": {
            "scalingRatio": 20.0,
            "gravity": 3.0,
            "strongGravityMode": True,
            "adjustSizes": True,
            "barnesHutOptimize": True,
            "linLogMode": False,
        },
        "sync": True,
    }), "forceatlas2")

    if lay.get("layout_exploded"):
        print("  Layout exploded — retrying with stronger gravity...")
        require(g.post("/layout/run", {
            "algorithm": "forceatlas2",
            "iterations": iterations,
            "properties": {
                "scalingRatio": 2.0,
                "gravity": 5.0,
                "strongGravityMode": True,
                "adjustSizes": True,
            },
            "sync": True,
        }), "forceatlas2-retry")
    print("  Layout done")

    # 6. Export PNGs — one per partition
    partitions = [
        ("block",               "combined_by_block"),
        ("ground_truth_event",  "combined_by_ground_truth"),
        ("louvain_community",   "combined_by_louvain"),
        ("modularity_class",    "combined_by_modularity"),
    ]
    width, height = 2400, 1600

    print(f"\n== Exporting PNGs ({width}x{height}) ==")
    written = []
    for col, fname in partitions:
        cres = g.post("/appearance/partition/color", {"column": col})
        if not cres.get("success", False):
            print(f"  [skip] color by '{col}': {cres.get('error')}")
            continue
        png = str((out_dir / f"{fname}.png").resolve())
        require(g.post("/export/png", {
            "file": png,
            "width": width,
            "height": height,
        }), f"export/{fname}")
        written.append(png)
        print(f"  OK {fname}.png")

    # 7. Also export individual block views if filter endpoint exists
    for block_val, block_name in [(0, "block0"), (5, "block5")]:
        for col, suffix in [("ground_truth_event", "ground_truth"), ("louvain_community", "louvain")]:
            try:
                # Color by partition first
                g.post("/appearance/partition/color", {"column": col})
                # Try to filter to just this block
                filt = g.post("/filters/add", {
                    "type": "attributes.equal",
                    "column": "block",
                    "value": str(block_val),
                })
                if filt.get("success"):
                    png = str((out_dir / f"{block_name}_{suffix}_filtered.png").resolve())
                    require(g.post("/export/png", {
                        "file": png, "width": width, "height": height,
                    }), f"export/{block_name}_{suffix}")
                    written.append(png)
                    print(f"  OK {block_name}_{suffix}_filtered.png")
                    g.post("/filters/remove", {})
            except Exception:
                pass

    print(f"\n{'='*60}")
    print(f"DONE — {len(written)} PNGs written to {out_dir}/")
    for p in written:
        print(f"  {p}")
    print(f"{'='*60}")
    return written


# ─── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--data-path", type=Path,
        default=Path(__file__).resolve().parent.parent
        / "DistilBERTGNN" / "incremental_test_100messagesperday",
        help="Root data directory with block subdirs",
    )
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "real_analysis_outputs" / "gephi",
                    help="Where to write PNGs")
    ap.add_argument("--gexf-out", type=Path,
                    default=Path(__file__).resolve().parent.parent
                    / "real_analysis_outputs" / "combined_block0_block5.gexf",
                    help="Where to write combined GEXF")
    ap.add_argument("--base", default="http://127.0.0.1:8081",
                    help="Gephi AI Server URL")
    ap.add_argument("--iterations", type=int, default=800,
                    help="ForceAtlas2 iterations")
    ap.add_argument("--skip-gephi", action="store_true",
                    help="Only build the GEXF, don't drive Gephi API")
    args = ap.parse_args()

    # Step 1: Build combined GEXF
    combined = build_combined_gexf(args.data_path, args.gexf_out)

    if args.skip_gephi:
        print("\n--skip-gephi set. Open the GEXF in Gephi manually.")
        print("Steps in Gephi:")
        print("  1. File > Open > select the combined GEXF")
        print("  2. Statistics panel > run Average Degree, Modularity")
        print("  3. Appearance > Nodes > Ranking > Size > degree")
        print("  4. Layout > ForceAtlas2 > Run")
        print("  5. Appearance > Nodes > Partition > Color > block / ground_truth_event / louvain_community")
        print("  6. File > Export > PNG")
        return

    # Step 2: Drive Gephi
    drive_gephi(args.gexf_out, args.out_dir, args.base, args.iterations)


if __name__ == "__main__":
    main()
