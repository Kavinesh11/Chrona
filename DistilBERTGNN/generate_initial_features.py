"""
DistilBERTGNN: Generate initial message features.

Replaces the original spaCy-based word-average embeddings (300-dim) with
DistilBERT [CLS] token embeddings (768-dim) as the semantic component.

Each message's final feature vector = [DistilBERT CLS (768-dim) | temporal (2-dim)] = 770-dim.

See Section 4.3 and Figure 1(b) of the DistilBERT-GNN paper.
"""

import numpy as np
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from datetime import datetime

load_path = '../datasets/Twitter/'
save_path = '../datasets/Twitter/'

# Load dataset
p_part1 = load_path + '68841_tweets_multiclasses_filtered_0722_part1.npy'
p_part2 = load_path + '68841_tweets_multiclasses_filtered_0722_part2.npy'
df_np_part1 = np.load(p_part1, allow_pickle=True)
df_np_part2 = np.load(p_part2, allow_pickle=True)
df_np = np.concatenate((df_np_part1, df_np_part2), axis=0)
print("Loaded data.")
df = pd.DataFrame(data=df_np, columns=[
    "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
    "place_type", "place_full_name", "place_country_code", "hashtags",
    "user_mentions", "image_urls", "entities",
    "words", "filtered_words", "sampled_words"
])
print("Data converted to dataframe.")
print(df.shape)
print(df.head(5))


def distilbert_cls_features(df, text_col='filtered_words', batch_size=32,
                             model_name='distilbert-base-uncased'):
    """
    Compute DistilBERT [CLS] token embeddings for each message.

    Args:
        df: DataFrame with a column of word lists
        text_col: column containing pre-tokenised word lists
        batch_size: inference batch size
        model_name: HuggingFace model identifier

    Returns:
        np.ndarray of shape (N, 768)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertModel.from_pretrained(model_name).to(device)
    model.eval()
    print("DistilBERT model loaded.")

    # Build text strings from word lists
    texts = []
    for val in df[text_col].values:
        if isinstance(val, (list, np.ndarray)):
            texts.append(' '.join(str(w) for w in val))
        else:
            texts.append(str(val))

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i: i + batch_size]
            encoding = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # [CLS] token is the first token's hidden state
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

            if (i // batch_size) % 10 == 0:
                print(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} messages...")

    return np.vstack(all_embeddings)


def extract_time_feature(t_str):
    """Encode one timestamp as a 2-d OLE date vector."""
    t = datetime.fromisoformat(str(t_str))
    OLE_TIME_ZERO = datetime(1899, 12, 30)
    delta = t - OLE_TIME_ZERO
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]


def df_to_t_features(df):
    """Encode timestamps of all messages in the dataframe."""
    t_features = np.asarray([extract_time_feature(t_str) for t_str in df['created_at']])
    return t_features


# Generate DistilBERT semantic features (768-dim)
print("\nGenerating DistilBERT [CLS] features (768-dim)...")
d_features = distilbert_cls_features(df)
print(f"DistilBERT features generated: {d_features.shape}")

# Generate temporal features (2-dim)
t_features = df_to_t_features(df)
print(f"Temporal features generated: {t_features.shape}")

# Concatenate: final feature dim = 770
combined_features = np.concatenate((d_features, t_features), axis=1)
print(f"Combined features shape: {combined_features.shape}")

# Save
out_path = save_path + 'features_distilbert_0709_multiclasses_filtered.npy'
np.save(out_path, combined_features)
print(f"Initial features saved to: {out_path}")

# Verify
loaded = np.load(out_path)
print(f"Verified loaded shape: {loaded.shape}")
