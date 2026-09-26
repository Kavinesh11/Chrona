# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

This repo is a case study (Social Network Analysis) reproducing/extending **DistilBERT-GNN**, a model for
incremental social media event detection. It combines a DistilBERT-based node filter, a GAT-style GNN over a
heterogeneous message graph, and contrastive learning (triplet + global-local pair loss) to cluster tweets into
events over time. See `README.md` for the full case-study write-up (problem, methodology, results) and
`base_paper/` for the source paper this implementation is based on.

## Commands

Run all commands from `DistilBERTGNN/` (paths in the code assume this as CWD).

Install dependencies (repo root has the actual pinned versions; `DistilBERTGNN/requirement.txt` is a legacy/older list):
```bash
pip install -r requirements.txt
```

Full pipeline, in order:
```bash
python generate_initial_features.py      # DistilBERT embeddings + initial message features
python custom_message_graph.py           # builds the incremental heterogeneous/homogeneous message graphs
python main.py                           # trains + evaluates DistilBERTGNN over the incremental blocks
```

`custom_message_graph.py` has a `test` flag inside `construct_incremental_dataset_0922()` — set `test=True` for a
quick run on a small graph, `test=False` for the full dataset.

Key `main.py` / `args.py` flags (all optional, defaults match the paper):
```bash
python main.py --filter_method {sentiment,centrality,none}   # node filtering strategy (default: sentiment)
python main.py --remove_obsolete {0,1,2}                     # message-updating strategy: 0=keep all, 1=keep relevant, 2=keep latest (default: 2)
python main.py --top_k_ratio 0.5                              # fraction of historical nodes kept at maintenance windows
python main.py --window_size 3 --num_heads 4 --out_dim 8 --n_neighbors 800 --patience 5
python main.py --resume_path <embeddings_dir> --resume_point <i> [--resume_current]  # resume a partially completed run
```

There is no test suite or linter configured in this repo.

### Inspecting runs

`main.py` writes each run to `<data_path>/embeddings_<timestamp>/` (contains `args.txt`, `evaluate.txt`,
`graph_statistics.txt`, per-epoch embeddings). Use the helpers in `tools/` (run from `tools/`, or add it to
`PYTHONPATH`) to inspect these without hand-parsing the text logs:

```bash
python find_latest_run.py <root>                 # newest embeddings_* dir under root
python check_run.py <run_dir> | --root <root>    # validates required files are present
python summarize_run.py <run_dir> | --root <root> [--pretty]  # parses evaluate.txt into JSON (epoch -> NMI/AMI/ARI)
python compare_runs.py <run_dir> [<run_dir> ...] [--section with_isolated_nodes|without_isolated_nodes]  # tabular diff across runs
python visualize_run.py metrics <run_dir> | --root <root>                  # NMI/AMI/ARI over time -> metrics_over_time.png
python visualize_run.py embeddings <run_dir> --block <i>                    # PCA scatter of message embeddings -> embedding_clusters_<i>.png
python visualize_run.py graph --data-path <data_path> --block <i>           # node-degree histogram -> degree_distribution_<i>.png
```
`run_artifacts.py` is the shared parsing library behind the run-inspection CLIs (`latest_run_dir`, `summarize_run`,
`pick_metrics`, `load_evaluate_history`) — extend parsing logic there, not in the individual scripts.
`visualize_run.py` produces the figures used in the case-study report/slides (see README's Visualization section);
`embeddings`/`graph` read from an actual run's output (`features_<i>.tsv`/`labels_<i>.tsv`) and the message-graph
data directory (`<data_path>/<i>/s_bool_A_tid_tid.npz`) respectively, so a full `main.py` run must exist first.

## Architecture

Pipeline (see README for the full narrative): raw tweets → preprocessing/NER → heterogeneous info network (HIN) →
homogeneous message graph → DistilBERT node filtering → GAT encoding → contrastive learning → DBSCAN/KMeans
clustering into events.

- `args.py` — single source of truth for all hyperparameters and CLI flags; imported everywhere else as
  `from args import args_define; args = args_define.args`. `data_path` defaults to
  `DistilBERTGNN/incremental_test_100messagesperday` (relative to `args.py`, not CWD).
- `generate_initial_features.py` — computes DistilBERT embeddings and initial per-message features; run first.
- `custom_message_graph.py` — builds the incremental message graphs block-by-block. The graph construction differs
  by `remove_obsolete` mode (0/1 vs 2) — read the file's header comment before changing this.
- `model.py` — the `DistilBERTGNN` model class, with the three lifecycle entry points used by `main.py`:
  `initial_maintain` (initial training and periodic maintenance/retraining), `infer` (prediction on a new block).
- `layers.py` — GAT-style attention layers used by `model.py`.
- `node_filter.py` — implements the two node-filtering strategies selectable via `--filter_method`: DistilBERT
  sentiment-confidence filtering (default) and graph-centrality filtering (degree/closeness/betweenness).
- `utils.py` — `SocialDataset`, embedding extraction/saving, `run_kmeans`/`evaluate` (NMI/AMI/ARI computation),
  `generateMasks` (train/validation/test index generation — logic branches on `remove_obsolete` mode), graph
  statistics.
- `metrics.py` — small `Metric` interface used during training (e.g. `AverageNonzeroTripletsMetric`).
- `main.py` — orchestrates the incremental loop: block 0 is always `initial_maintain`ed first; for each subsequent
  block `i`, `infer` runs prediction, and every `window_size`-th block triggers another `initial_maintain` (model
  maintenance/retraining). Supports resuming via `--resume_path`/`--resume_point`/`--resume_current`.
- `baselines/` — standalone comparison methods (`BiLSTM.py`, `eventx.py`), not wired into `main.py`.
- `tools/` — post-hoc run inspection utilities (see Commands above); independent of the training code, only reads
  the `embeddings_*` output directories.

### Incremental training/inference loop (main.py)

The loop is stateful and order-dependent: `train_i` tracks which block was last used for training, `model` and
`train_indices`/`indices_to_remove` carry over between iterations. Understanding this loop requires reading
`main.py` together with `initial_maintain`/`infer` in `model.py` and `generateMasks` in `utils.py` — the index
bookkeeping (which nodes are train/validation/test at any point) depends on both the current block `i` and the
active `remove_obsolete` strategy.
