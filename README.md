# DistilBERT-GNN: Social Media Event Detection

An incremental, adaptable, and contextually aware event detection framework that leverages DistilBERT and Graph Neural Networks (GNNs) for detecting and tracking events as they emerge and evolve on social media platforms.

## Overview

**DistilBERT-GNN** proposes a novel approach to social media event detection that combines:
- **DistilBERT**: For real-time contextual understanding and sentiment-based node filtering
- **Graph Neural Networks (GNNs)**: To capture evolving relationships in social media networks
- **Contrastive Learning**: Using triplet loss and global-local pair loss for scalable training

### Key Contributions

1. **DistilBERT-driven Node Filtering**: Optimizes node selection while mitigating noise in social media data through sentiment analysis
2. **Incremental Learning Framework**: Continuously adapts to new data without forgetting previously learned patterns
3. **State-of-the-art Performance**: Outperforms baseline methods with NMI=0.72, AMI=0.53, ARI=0.24 on Twitter datasets

### Problem Statement

Social media event detection faces three major challenges:
- **Vocabulary Gap**: Language constantly evolves with slang and abbreviations absent from traditional dictionaries
- **Short & Unstructured Tweets**: Limited context makes semantic interpretation difficult
- **Event Involvement**: Dynamic nature requires incremental adaptation rather than static models

### Architecture

The DistilBERT-GNN framework consists of six main phases:

```
a) Homogeneous Message Graph → 
b) Node Filtering (DistilBERT) → 
c) Attention Residual → 
d) Message Representations → 
e) Contrastive Learning (Lt + Lp) → 
f) Social Event Clustering
```

**Figure 1**: DistilBERT-GNN architectural design with components for heterogeneous graph construction, node filtering via DistilBERT, multi-head attention, message representation learning, and contrastive loss optimization.

## Methodology

### 1. Pre-processing Phase
- URL and special character removal
- Stop word elimination and stemming
- Named Entity Recognition (NER) for identifying persons, locations, organizations
- Vector Space Model (VSM) vectorization of message content

### 2. Heterogeneous Social Message Modeling
Constructs Heterogeneous Information Networks (HINs) by:
- Extracting entities and keywords from messages
- Creating nodes for messages, words, entities, and users
- Establishing edges between messages sharing common keywords
- Transforming into homogeneous message graphs

### 3. Knowledge Preserving Incremental Message Embedding
Three-stage lifecycle:
- **Stage I (Pre-training)**: Initial message graph construction and model training
- **Stage II (Detection)**: Real-time event detection via graph updates
- **Stage III (Maintenance)**: Periodic retraining to discard obsolete messages and retain relevant patterns

### 4. Node Filtering (DistilBERT-based)
Employs sentiment analysis via DistilBERT to:
- Predict emotional tone of messages
- Include messages with confidence score > 50%
- Reduce noise and irrelevant content from the graph
- Improve computational efficiency

Alternative: Centrality-based filtering using degree, proximity, and betweenness measures (less preferred).

### 5. Message Clustering
Uses DBSCAN (density-based clustering) to:
- Group message representations into events
- Avoid requiring predefined number of event classes
- Support open-set event detection

### 6. Scalability Through Contrastive Learning
Combines two loss functions:

**Triplet Loss (Lt)**:
```
Lt = max{D(h_mi, h_mi+) - D(h_mi, h_mi-) + α, 0}
```
Enforces hard negative triplets for faster convergence in dynamic streams.

**Global-Local Pair Loss (Lp)**:
```
Lp = (1/N) Σ [log S(h_mi, s) + log(1 - S(h~_mi, s))]
```
Maximizes mutual information between local and global graph representations.

**Total Loss**: L = Lt + Lp

## Workflow Pipeline

To run and evaluate the DistilBERTGNN model, follow these steps from the root directory (`/DistilBERTGNN`):

### 1. Setup and Installation
Ensure you have the required dependencies (such as `torch`, `torch-geometric`, `transformers`, `scikit-learn`, `pandas`, `numpy`, and `spacy`) installed in your environment.

### 2. Generate Initial Features
Run the feature extraction script to compute the DistilBERT embeddings and other initial features for the messages:
```bash
python generate_initial_features.py
```

### 3. Construct the Message Graph
Run the graph construction script to build the heterogeneous incremental message graphs (filtering noisy nodes and creating similarity-based edges):
```bash
python custom_message_graph.py
```
*Note:* To run a quick test with small message graphs, set `test=True` (e.g., when calling `construct_incremental_dataset_0922()`) within the script. To use all messages, set `test=False`.

### 4. Training and Evaluation
Train the main DistilBERTGNN model on the incremental data blocks:
```bash
python main.py
```

## Experimental Results

### Datasets

1. **Twitter Dataset** (Primary):
   - 68,841 messages
   - 503 event categories
   - 16,358,812 edges in graph
   - Collected over 28 days (Oct 10 - Nov 7, 2022)
   - Events: earthquakes, storms, breaking news
   - Split: 70% training, 20% test, 10% validation

2. **MAVEN Dataset**:
   - General domain event detection
   - 10,242 messages over 154 classes

### Performance Metrics

DistilBERT-GNN significantly outperforms baseline methods:

| Metric | Score | vs Word2Vec | vs LDA | vs KPGNN | vs QSGNN |
|--------|-------|------------|--------|----------|----------|
| **NMI** | 0.72 | +28% | +43% | +2.8% | +2.8% |
| **AMI** | 0.53 | +40% | +49% | +1.9% | +3.9% |
| **ARI** | 0.24 | +22% | +23% | +9.1% | +20% |

**Evaluation Metrics**:
- **NMI (Normalized Mutual Information)**: Measures agreement between ground truth and predicted clusters (0-1 scale)
- **AMI (Adjusted Mutual Information)**: Variant of NMI adjusted for chance agreement
- **ARI (Adjusted Rand Index)**: Measures similarity between clusterings adjusted for random chance

### Key Findings

1. **Offline Evaluation**: DistilBERT-GNN consistently achieves highest scores across all metrics
2. **Online Evaluation**: Maintains performance across 22 message blocks with varying data distributions
3. **Execution Time**: 15-40% faster than KPGNN and QSGNN on large message blocks
4. **Convergence**: Reaches optimal NMI within 250 epochs, stabilizes thereafter

### Message Updating Strategies

Three approaches tested:
- **Keep All**: Accumulates all messages (worst performance, memory-intensive)
- **Keep Relevant**: Retains messages connected to recent arrivals (moderate)
- **Keep Latest**: Uses only newest block (best performance, efficient) ✓

### Ablation Studies

1. **Hard Negative Sampling**: Including hard negatives consistently improves NMI scores across all blocks
2. **Window Size**: Smaller windows (1-3) slightly outperform larger ones (0.75 vs 0.74 NMI)
3. **Batch Size**: Stable performance across batch sizes (1000-4000) for NMI and AMI

## Baselines Compared

The model is evaluated against state-of-the-art methods:

### Traditional Methods
- **Word2Vec**: Word embedding vectors (Mikolov et al., 2013)
- **LDA**: Probabilistic topic modeling (Blei et al., 2003)
- **WMD**: Word Mover's Distance (Kusner et al., 2015)

### Deep Learning Methods
- **BERT**: Bidirectional Encoder Representations (Devlin et al., 2018)
- **BiLSTM**: Bidirectional LSTM for sequential learning (Graves & Schmidhuber, 2005)

### Graph-based Methods
- **PP-GCN**: Heterogeneous information network with GCN (Peng et al., 2019)
- **EventX**: Online event extraction (Liu et al., 2020)
- **KPGNN**: Knowledge-preserving GNN (Cao et al., 2021)
- **QSGNN**: Quality-aware self-improving GNN (Ren et al., 2022)

Please refer to the `baselines/` directory for individual execution scripts.

## Running Ablation Studies

To reproduce the ablation studies and variants of the DistilBERTGNN model, you can run `main.py` with the following command-line parameters:

**1. Node Filtering Strategy:**
- **No Filtering:** `python3 main.py --filter_method none`
- **Centrality-based Filtering:** `python3 main.py --filter_method centrality`
- **Sentiment-based Filtering (Ours):** `python3 main.py --filter_method sentiment` (Default)

**2. Historical Maintenance / Similar Message Selection:**
- **Keep all historical messages (No selection):** `python3 main.py --top_k_ratio 1.0`
- **Select top similar memory messages:** `python3 main.py --top_k_ratio 0.5` (Default)

**3. Hyperparameter Sensitivity (Optional Attributes):**
- **Maintenance Window Size:** `--window_size <int>` (Default: 3) — Optimal: 1-3
- **GAT Attention Heads:** `--num_heads <int>` (Default: 4)
- **GNN Output Dimension:** `--out_dim <int>` (Default: 8)
- **Number of Neighborhood Samples:** `--neighbor_samples <int>` (Default: 800) — Stable 600-1000
- **Embedding Dimension:** `--embed_dim <int>` (Default: 64) — Stable 100-500
- **Early Stopping Patience:** `--early_stop <int>` (Default: 5) — Stable 6-14

## Model Parameters

Default configuration used in experiments:

| Parameter | Value |
|-----------|-------|
| Number of GNN Layers | 2 |
| Attention Heads | 4 |
| Embedding Dimension | 64 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Training Epochs | 200 |
| Early Stopping Patience | 5 |
| Window Size | 3 |
| Mini-batch Size | 200 |
| Triplet Margin | 3 |
| Neighborhood Samples | 800 |

## Implementation Details

### Data Flow Pipeline

1. **Raw Tweets** → Pre-processing (URL removal, stemming, NER)
2. **Preprocessed Messages** → Heterogeneous Graph Construction
3. **HIN** → Homogeneous Message Graph Transformation
4. **Message Graph** → DistilBERT Node Filtering
5. **Filtered Nodes** → GNN Encoding with Multi-head Attention
6. **Message Embeddings** → Contrastive Learning (Triplet + Global-Local Loss)
7. **Trained Representations** → DBSCAN Clustering
8. **Event Clusters** → Social Event Detection Output

### Key Algorithms

**Adjacency Matrix Computation**:
```
A[i,j] = min{[Σ_k W_mk · (W_mk)^T][i,j], 1}
```
Creates edges between messages sharing keywords, entities, or users.

**Message Representation Update** (GNN layer l):
```
h^(l+1)_mi = h^l_mi ⊕ Aggregator(Extractor(h^l_mj) ∀mj ∈ N(mi))
```
Aggregates features from neighboring messages using head-wise concatenation.

## Future Work

- Exploration of different modalities (images, videos) in social media
- Real-time deployment on streaming platforms
- Integration with external knowledge bases
- Multi-lingual event detection
- Temporal dynamics modeling for event evolution

## Citation

If you use DistilBERT-GNN in your research, please cite the associated paper.

## References

See the paper for complete references to:
- Transformer models (BERT, DistilBERT)
- Graph Neural Networks (GCN, GAT, GNN)
- Event detection methods (LDA, Topic Detection and Tracking)
- Contrastive learning approaches
- Social media analysis literature

