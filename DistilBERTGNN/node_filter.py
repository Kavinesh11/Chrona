"""
Node Filtering for DistilBERTGNN

Two strategies (Section 4.4 of the DistilBERTGNN paper):

  4.4.1  Sentiment-based: DistilBERT predicts sentiment; messages with
         confidence score <= threshold are excluded as noise.

  4.4.2  Centrality-based: degree + closeness + betweenness centrality
         are combined; tweet nodes below a percentile threshold are removed.

A unified dispatcher `apply_node_filter()` lets callers choose the strategy
via --filter_method ('sentiment' | 'centrality' | 'none').

Algorithm 1, line 6: "Apply DistilBERT"
"""

import numpy as np
import networkx as nx

try:
    import torch  # noqa: F401
    from transformers import pipeline as _hf_pipeline
    _TORCH_AVAILABLE = True
except (ImportError, NameError):
    _hf_pipeline = None
    _TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# 4.4.1  Sentiment-based node filtering
# ─────────────────────────────────────────────────────────────────────────────

_sentiment_pipeline = None


def get_sentiment_pipeline():
    """Lazily load the DistilBERT sentiment pipeline (once per process)."""
    global _sentiment_pipeline
    if not _TORCH_AVAILABLE:
        print("[WARNING] PyTorch not available – sentiment pipeline disabled.")
        return None
    if _sentiment_pipeline is None:
        print("Loading DistilBERT sentiment model for node filtering...")
        try:
            _sentiment_pipeline = _hf_pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512,
            )
            print("DistilBERT sentiment model loaded.")
        except Exception as e:
            print(f"[WARNING] Could not load DistilBERT sentiment pipeline: {e}")
            print("[WARNING] Sentiment filtering disabled – all nodes will be kept.")
            return None
    return _sentiment_pipeline


def filter_nodes_by_sentiment(df, threshold=0.50, text_col='filtered_words', batch_size=32):
    """
    Keep message nodes with DistilBERT sentiment confidence > threshold.

    Args:
        df            : message DataFrame
        threshold     : confidence cutoff (default 0.50, from paper §4.4.1)
        text_col      : column containing word list or raw text
        batch_size    : inference batch size

    Returns:
        filtered_df, mask (bool ndarray), scores (list[float])
    """
    model = get_sentiment_pipeline()

    if model is None:
        mask = np.ones(len(df), dtype=bool)
        return df.copy(), mask, [1.0] * len(df)

    texts = []
    for val in df[text_col].values:
        if isinstance(val, (list, np.ndarray)):
            texts.append(' '.join(str(w) for w in val))
        else:
            texts.append(str(val))

    scores = []
    for i in range(0, len(texts), batch_size):
        results = model(texts[i: i + batch_size])
        scores.extend([r['score'] for r in results])

    mask = np.array([s > threshold for s in scores])
    filtered_df = df[mask].copy()
    print(f"[Sentiment Filter] kept {mask.sum()}/{len(mask)} nodes "
          f"({(~mask).sum()} removed, threshold={threshold:.2f})")
    return filtered_df, mask, scores


# ─────────────────────────────────────────────────────────────────────────────
# 4.4.2  Centrality-based node filtering
# ─────────────────────────────────────────────────────────────────────────────

def filter_nodes_by_centrality(df, G, threshold_percentile=50.0,
                                tweet_id_col='tweet_id', betweenness_k=None):
    """
    Remove tweet nodes whose combined centrality score falls below a percentile.

    Three measures are computed on the full heterogeneous NetworkX graph G:
      - Degree centrality
      - Closeness centrality
      - Betweenness centrality (approximated when betweenness_k is set)

    Their average forms the combined score. Tweet nodes below
    `threshold_percentile` (default 50 – median) are dropped.

    Args:
        df                    : message DataFrame (tweet_id column must align with G)
        G                     : NetworkX heterogeneous graph (from construct_graph_from_df)
        threshold_percentile  : percentile cutoff (0–100). Default 50.
        tweet_id_col          : DataFrame column for tweet IDs
        betweenness_k         : pivot nodes for approximate betweenness.
                                None = exact. Recommended: min(50, n_tweet_nodes).

    Returns:
        filtered_df, mask (bool ndarray), combined_scores (dict node->score)
    """
    tweet_nodes = [n for n in G.nodes if str(n).startswith('t_')]
    n_tweets = len(tweet_nodes)

    if n_tweets == 0:
        print("[Centrality Filter] No tweet nodes in graph – keeping all rows.")
        return df.copy(), np.ones(len(df), dtype=bool), {}

    print(f"[Centrality Filter] Computing 3 centrality measures for {n_tweets} tweet nodes...")

    # 1. Degree centrality – O(V)
    degree_cent = nx.degree_centrality(G)

    # 2. Closeness centrality – O(V * (V + E))
    closeness_cent = nx.closeness_centrality(G)

    # 3. Betweenness centrality – O(V*E); use k-pivot approximation for speed
    k_sample = betweenness_k if betweenness_k else min(50, n_tweets)
    print(f"[Centrality Filter] Betweenness: using k={k_sample} pivot nodes (approximation).")
    betweenness_cent = nx.betweenness_centrality(G, k=k_sample, normalized=True)

    # Combined score = average of three measures (for tweet nodes only)
    combined_scores = {}
    for node in tweet_nodes:
        d = degree_cent.get(node, 0.0)
        c = closeness_cent.get(node, 0.0)
        b = betweenness_cent.get(node, 0.0)
        combined_scores[node] = (d + c + b) / 3.0

    # Cutoff from requested percentile
    score_values = np.array(list(combined_scores.values()))
    cutoff = float(np.percentile(score_values, threshold_percentile))

    # Build boolean mask aligned to df row order
    mask = np.array(
        [combined_scores.get('t_' + str(tid), 0.0) >= cutoff
         for tid in df[tweet_id_col].values],
        dtype=bool
    )
    filtered_df = df[mask].copy()
    print(f"[Centrality Filter] kept {mask.sum()}/{len(mask)} nodes "
          f"({(~mask).sum()} removed, percentile={threshold_percentile:.1f})")
    return filtered_df, mask, combined_scores


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def apply_node_filter(df, method='sentiment', G=None,
                      sentiment_threshold=0.50,
                      centrality_percentile=50.0,
                      text_col='filtered_words'):
    """
    Dispatch to the selected node-filtering strategy.

    Args:
        method                : 'sentiment' | 'centrality' | 'none'
        G                     : NetworkX graph (required for method='centrality')
        sentiment_threshold   : confidence cutoff for sentiment filter
        centrality_percentile : percentile cutoff for centrality filter
        text_col              : column used for sentiment text

    Returns:
        filtered_df, mask, scores
    """
    if method == 'sentiment':
        return filter_nodes_by_sentiment(df, threshold=sentiment_threshold,
                                         text_col=text_col)
    elif method == 'centrality':
        if G is None:
            raise ValueError("NetworkX graph G must be provided for centrality filtering.")
        return filter_nodes_by_centrality(df, G,
                                          threshold_percentile=centrality_percentile)
    elif method == 'none':
        mask = np.ones(len(df), dtype=bool)
        print(f"[Node Filter] method='none': keeping all {len(df)} nodes.")
        return df.copy(), mask, {}
    else:
        raise ValueError(f"Unknown filter_method '{method}'. "
                         "Choose: 'sentiment', 'centrality', 'none'.")
