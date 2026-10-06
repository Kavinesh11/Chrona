"""
One-command orchestrator for the features -> graph -> Gephi export chain:

    1. generate_initial_features.py  (skipped if the features file already exists)
    2. custom_message_graph.py       (skipped if the target block's graph already exists)
    3. export_gephi.py               (always run, for each requested block)

Each step is invoked as the real, unmodified script via subprocess — this file
does not reimplement or patch any pipeline logic, it just sequences the
existing ones and skips steps whose output is already present.

Requires the actual QSGNN Twitter benchmark at
<distilbertgnn-dir>/../datasets/Twitter/68841_tweets_multiclasses_filtered_0722_part{1,2}.npy.
If you don't have it yet, generate a synthetic stand-in first to smoke-test the
chain:

    python make_synthetic_dataset.py --out-dir <distilbertgnn-dir>/../datasets/Twitter

Usage (from tools/, or anywhere with --distilbertgnn-dir set):
    python run_gephi_pipeline.py --block 0
    python run_gephi_pipeline.py --all-blocks --max-nodes 800
    python run_gephi_pipeline.py --block 0 --filter_method centrality --force
"""

import argparse
import os
import subprocess
import sys


def run_step(cmd, cwd, label):
    print(f"\n=== {label} ===")
    print(f"$ {' '.join(cmd)}  (cwd={cwd})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        print(f"\n[run_gephi_pipeline] '{label}' failed (exit code {result.returncode}) — stopping.", file=sys.stderr)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    default_distilbertgnn_dir = os.path.normpath(os.path.join(here, "..", "DistilBERTGNN"))

    parser.add_argument("--distilbertgnn-dir", default=default_distilbertgnn_dir,
                         help=f"Path to DistilBERTGNN/ (default: {default_distilbertgnn_dir})")
    parser.add_argument("--data-path", default=None,
                         help="Root data dir (default: <distilbertgnn-dir>/incremental_test_100messagesperday, "
                              "matching args.py's own default)")
    block_group = parser.add_mutually_exclusive_group(required=True)
    block_group.add_argument("--block", type=int, help="Export this single block index")
    block_group.add_argument("--all-blocks", action="store_true", help="Export every block found under data-path")
    parser.add_argument("--max-nodes", type=int, default=None, help="Forwarded to export_gephi.py")
    parser.add_argument("--out-dir", default=".", help="Directory to write the .gexf file(s) into")
    parser.add_argument("--filter_method", default="sentiment", choices=["sentiment", "centrality", "none"],
                         help="Forwarded to custom_message_graph.py (default matches the paper: sentiment)")
    parser.add_argument("--force", action="store_true",
                         help="Re-run generate_initial_features.py / custom_message_graph.py even if their "
                              "outputs already exist")
    args = parser.parse_args()

    distilbertgnn_dir = os.path.abspath(args.distilbertgnn_dir)
    data_path = args.data_path or os.path.join(distilbertgnn_dir, "incremental_test_100messagesperday")
    raw_dataset_dir = os.path.normpath(os.path.join(distilbertgnn_dir, "..", "datasets", "Twitter"))

    part1 = os.path.join(raw_dataset_dir, "68841_tweets_multiclasses_filtered_0722_part1.npy")
    part2 = os.path.join(raw_dataset_dir, "68841_tweets_multiclasses_filtered_0722_part2.npy")
    if not (os.path.exists(part1) and os.path.exists(part2)):
        print(
            f"[run_gephi_pipeline] Raw dataset not found at {raw_dataset_dir}\n"
            f"  Expected: {os.path.basename(part1)} and {os.path.basename(part2)}\n\n"
            f"  This is the QSGNN Twitter benchmark — it is not bundled with this repo.\n"
            f"  Either place it at the path above, or generate a synthetic stand-in to\n"
            f"  smoke-test this pipeline:\n\n"
            f"      python make_synthetic_dataset.py --out-dir {raw_dataset_dir}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    features_path = os.path.join(raw_dataset_dir, "features_distilbert_0709_multiclasses_filtered.npy")
    if args.force or not os.path.exists(features_path):
        run_step([sys.executable, "generate_initial_features.py"], distilbertgnn_dir, "1/3 generate_initial_features.py")
    else:
        print(f"\n=== 1/3 generate_initial_features.py === skipped (found {features_path})")

    block0_graph = os.path.join(data_path, "0", "s_bool_A_tid_tid.npz")
    if args.force or not os.path.exists(block0_graph):
        run_step(
            [sys.executable, "custom_message_graph.py", "--filter_method", args.filter_method],
            distilbertgnn_dir, "2/3 custom_message_graph.py",
        )
    else:
        print(f"\n=== 2/3 custom_message_graph.py === skipped (found {block0_graph})")

    if args.all_blocks:
        blocks = sorted(
            int(name) for name in os.listdir(data_path)
            if name.isdigit() and os.path.exists(os.path.join(data_path, name, "s_bool_A_tid_tid.npz"))
        )
        if not blocks:
            print(f"[run_gephi_pipeline] No block directories with s_bool_A_tid_tid.npz found under {data_path}", file=sys.stderr)
            sys.exit(1)
    else:
        blocks = [args.block]

    os.makedirs(args.out_dir, exist_ok=True)
    export_script = os.path.join(here, "export_gephi.py")
    for b in blocks:
        cmd = [sys.executable, export_script, "--data-path", data_path, "--block", str(b),
               "--out", os.path.join(args.out_dir, f"block{b}.gexf")]
        if args.max_nodes:
            cmd += ["--max-nodes", str(args.max_nodes)]
        run_step(cmd, here, f"3/3 export_gephi.py (block {b})")

    print(f"\nDone — {len(blocks)} block(s) exported to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
