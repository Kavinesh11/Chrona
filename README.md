# DistilBERTGNN

This repository contains the source code, preprocessing scripts, and model implementation for **DistilBERTGNN**. DistilBERT-GNN: A Powerful Approach to Social Media Event Detection, which incorporates DistilBERT for sentiment-based node filtering and rich semantic feature extraction.

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

## Datasets

The model evaluates performance on two main datasets:
1. **Twitter dataset**: 68,841 manually labeled tweets over 503 event classes.
2. **MAVEN dataset**: A general domain event detection dataset containing 10,242 messages over 154 classes.

For detailed information regarding dataset format and how to load the data, please see the [**data_usage.md**](data_usage.md) file.

## Baselines

This project includes baseline models:
- **Word2vec**: Uses `spaCy` pre-trained vectors.
- **LDA, WMD, BERT, and PP-GCN**: Open-source implementations.
- **EventX** and **BiLSTM**: Available in the `baselines` folder.

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
- **Maintenance Window Size:** `--window_size <int>` (Default: 3)
- **GAT Attention Heads:** `--num_heads <int>` (Default: 4)
- **GNN Output Dimension:** `--out_dim <int>` (Default: 8)

