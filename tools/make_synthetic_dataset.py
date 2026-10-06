"""
Generates a small, clearly-synthetic stand-in for the QSGNN Twitter benchmark
(68841_tweets_multiclasses_filtered_0722_part1/2.npy) that generate_initial_features.py
and custom_message_graph.py expect at <distilbertgnn-dir>/../datasets/Twitter/.

Use this when the real benchmark isn't available and you want to smoke-test the
real, unmodified pipeline (features -> graph -> export_gephi.py) end-to-end, or
to demo tools/export_gephi.py without downloading the actual dataset.

NOT real data. Six synthetic "events" each get their own small vocabulary of
entities/keywords and a recurring pool of user ids, so the resulting message
graph has genuine event-correlated community structure (for Louvain etc. to
find) rather than pure noise — but any numbers computed downstream from this
are a pipeline smoke test, not a research result.

Usage (from anywhere):
    python make_synthetic_dataset.py --out-dir ../DistilBERTGNN/../datasets/Twitter
    python make_synthetic_dataset.py --out-dir /path/to/datasets/Twitter --n-events 8 --n-dates 15
"""

import argparse
import os
import random
from datetime import datetime, timedelta

import numpy as np

COLUMNS = [
    "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
    "place_type", "place_full_name", "place_country_code", "hashtags",
    "user_mentions", "image_urls", "entities",
    "words", "filtered_words", "sampled_words",
]

FILLER_WORDS = ["breaking", "update", "just", "happened", "near", "reports", "say", "watch", "live", "now"]


def build_dataset(n_events, n_dates, tweets_per_date, seed):
    rng = random.Random(seed)
    base_date = datetime(2022, 10, 10, 6, 0, 0)

    events = []
    for e in range(n_events):
        events.append({
            "name": f"event{e}",
            "keywords": [f"kw_e{e}_{j}" for j in range(8)],
            "entities": [f"ent_e{e}_{j}" for j in range(4)],
            "users": [10000 + e * 100 + j for j in range(20)],
        })

    rows = []
    tweet_id = 1
    for d in range(n_dates):
        date_dt = base_date + timedelta(days=d)
        for _ in range(tweets_per_date):
            ev = rng.choice(events)
            created_at = date_dt + timedelta(minutes=rng.randint(0, 23 * 60))

            user_id = rng.choice(ev["users"]) if rng.random() < 0.85 else rng.randint(90000, 99999)
            mentions = []
            if rng.random() < 0.3:
                mentions.append(rng.choice(ev["users"]))

            tweet_entities = rng.sample(ev["entities"], k=rng.randint(1, min(3, len(ev["entities"]))))
            sampled_words = rng.sample(ev["keywords"], k=rng.randint(2, 4))
            filler = rng.sample(FILLER_WORDS, k=rng.randint(3, 5))
            all_words = filler + sampled_words
            filtered_words = filler[:2] + sampled_words
            hashtags = ["#" + w for w in rng.sample(ev["keywords"], k=min(2, len(ev["keywords"])))]

            rows.append([
                ev["name"].replace("event", ""), tweet_id,
                f"synthetic tweet {tweet_id} about {ev['name']}",
                user_id, created_at, "unknown", "unknown", "unknown", "unknown",
                hashtags, mentions, [], tweet_entities, all_words, filtered_words, sampled_words,
            ])
            tweet_id += 1

    data = np.array(rows, dtype=object)
    assert data.shape[1] == len(COLUMNS)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True, help="Directory to write the two part files into "
                                                           "(this is the '../datasets/Twitter/' the real scripts expect)")
    parser.add_argument("--n-events", type=int, default=6)
    parser.add_argument("--n-dates", type=int, default=12, help=">=9 so at least a couple of incremental blocks get built")
    parser.add_argument("--tweets-per-date", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = build_dataset(args.n_events, args.n_dates, args.tweets_per_date, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    split = data.shape[0] // 2
    part1_path = os.path.join(args.out_dir, "68841_tweets_multiclasses_filtered_0722_part1.npy")
    part2_path = os.path.join(args.out_dir, "68841_tweets_multiclasses_filtered_0722_part2.npy")
    np.save(part1_path, data[:split], allow_pickle=True)
    np.save(part2_path, data[split:], allow_pickle=True)

    print(f"Synthetic dataset written: {data.shape[0]} tweets, {args.n_events} events, {args.n_dates} dates")
    print(f"  {part1_path}")
    print(f"  {part2_path}")
    print("NOT real data — for smoke-testing generate_initial_features.py / "
          "custom_message_graph.py / export_gephi.py only.")


if __name__ == "__main__":
    main()
