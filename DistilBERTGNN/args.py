import argparse
import os

# Always resolve data_path relative to this file's location, regardless of CWD
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA_PATH = os.path.join(_HERE, 'incremental_test_100messagesperday')

class args_define():
    
    parser = argparse.ArgumentParser()
    # Hyper-parameters
    parser.add_argument('--n_epochs', default=15, type=int,
                        help="Number of initial-training/maintenance-training epochs.")
    parser.add_argument('--n_infer_epochs', default=0, type=int,
                        help="Number of inference epochs.")
    parser.add_argument('--window_size', default=3, type=int,
                        help="Maintain the model after predicting window_size blocks.")
    parser.add_argument('--patience', default=5, type=int,
                        help="Early stop if performance did not improve in the last patience epochs.")
    parser.add_argument('--margin', default=3., type=float,
                        help="Margin for computing triplet losses")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument('--batch_size', default=2000, type=int,
                        help="Batch size (number of nodes sampled to compute triplet loss in each batch)")
    parser.add_argument('--n_neighbors', default=800, type=int,
                        help="Number of neighbors sampled for each node.")
    parser.add_argument('--hidden_dim', default=8, type=int,
                        help="Hidden dimension")
    parser.add_argument('--out_dim', default=32, type=int,
                        help="Output dimension of tweet representations")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="Number of heads in each GAT layer")
    parser.add_argument('--use_residual', dest='use_residual', default=True,
                        action='store_false',
                        help="If true, add residual(skip) connections")
    parser.add_argument('--validation_percent', default=0.2, type=float,
                        help="Percentage of validation nodes(tweets)")
    parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False,
                        action='store_true',
                        help="If true, use hardest negative messages to form triplets. Otherwise use random ones")
    parser.add_argument('--use_dgi', dest='use_dgi', default=False,
                        action='store_true',
                        help="If true, add a DGI term to the loss. Otherwise use triplet loss only")
    parser.add_argument('--remove_obsolete', default=2, type=int,
                        help="If 0, adopt the All Message Strategy: keep inseting new message blocks and never remove;\n" +
                             "if 1, adopt the Relevant Message Strategy: remove obsolete training nodes (that are not connected to the new messages arrived during the last window) during maintenance;\n" +
                             "if 2, adopt the Latest Message Strategy: during each prediction, use only the new data to construct message graph, during each maintenance, use only the last message block arrived during the last window for continue training.\n" +
                             "See Section 4.4 of the paper for detailed explanations of these strategies. Note the message graphs for 0/1 and 2 need to be constructed differently, see the head comment of custom_message_graph.py")

    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=False,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--data_path', default=_DEFAULT_DATA_PATH,
                        type=str, help="Path of features, labels and edges")
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--mask_path', default=None,
                        type=str, help="File path that contains the training, validation and test masks")
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--resume_path', default=None,
                        type=str,
                        help="File path that contains the partially performed experiment that needs to be resume.")
    parser.add_argument('--resume_point', default=0, type=int,
                        help="The block model to be loaded.")
    parser.add_argument('--resume_current', dest='resume_current', default=True,
                        action='store_false',
                        help="If true, continue to train the resumed model of the current block(to resume a partally trained initial/mantenance block);\
                            If false, start the next(infer/predict) block from scratch;")
    parser.add_argument('--log_interval', default=10, type=int,
                        help="Log interval")
    # DistilBERTGNN: fraction of historical training nodes to keep at each maintenance window
    # (extract_similar_messages, Algorithm 1 line 14)
    parser.add_argument('--top_k_ratio', default=0.5, type=float,
                        help="Fraction of historical training nodes to keep at maintenance "
                             "windows (cosine similarity selection). Default: 0.5")
    # DistilBERTGNN: node filtering strategy (Section 4.4)
    parser.add_argument('--filter_method', default='sentiment',
                        choices=['sentiment', 'centrality', 'none'],
                        help="Node filtering strategy (§4.4): "
                             "'sentiment' uses DistilBERT confidence score (§4.4.1); "
                             "'centrality' uses degree+closeness+betweenness centrality (§4.4.2); "
                             "'none' keeps all nodes.")
    parser.add_argument('--sentiment_threshold', default=0.50, type=float,
                        help="Confidence cutoff for sentiment-based node filtering (default 0.50).")
    parser.add_argument('--centrality_percentile', default=50.0, type=float,
                        help="Percentile cutoff for centrality-based node filtering "
                             "(default 50 = median). Nodes below this percentile are removed.")

    args = parser.parse_args()
