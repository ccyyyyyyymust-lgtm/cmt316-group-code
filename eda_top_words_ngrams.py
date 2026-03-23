from pathlib import Path
import pandas as pd
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

DATA_DIR = Path(r"X:\CMT316\CW2\DATASET")
OUTPUT_DIR = Path(r"X:\CMT316\CW2\outputs\tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

STOPWORDS = set(ENGLISH_STOP_WORDS)
CUSTOM_REMOVE = {
    "user", "amp", "rt",
    "@user",
    "tomorrow", "today", "day", "night",
    "monday", "friday", "saturday", "sunday",
    "1st", "2nd", "3rd",
    "just", "going", "time", "new"
}
STOPWORDS = STOPWORDS.union(CUSTOM_REMOVE)

def load_split(text_file, label_file, split_name):
    with open(DATA_DIR / text_file, "r", encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f]

    with open(DATA_DIR / label_file, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f]

    df = pd.DataFrame({
        "text": texts,
        "label_id": labels
    })
    df["label_name"] = df["label_id"].map(LABEL_MAP)
    df["split"] = split_name
    return df

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\\u[0-9a-fA-F]{4}", " ", text)
    text = re.sub(r"&amp;", " ", text)
    text = re.sub(r"[^a-z0-9#@'\s]", " ", text)
    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
    return tokens

def get_top_ngrams(texts, n=1, top_k=20):
    counter = Counter()
    for text in texts:
        tokens = clean_and_tokenize(text)
        if len(tokens) < n:
            continue
        ngrams = zip(*[tokens[i:] for i in range(n)])
        counter.update([" ".join(ng) for ng in ngrams])
    return counter.most_common(top_k)

train_df = load_split("train_text.txt", "train_labels.txt", "train")

results = []

# overall
for n in [1, 2, 3]:
    top_items = get_top_ngrams(train_df["text"], n=n, top_k=20)
    for rank, (ngram, count) in enumerate(top_items, start=1):
        results.append({
            "group": "overall",
            "ngram_n": n,
            "rank": rank,
            "ngram": ngram,
            "count": count
        })

# by label
for label in ["negative", "neutral", "positive"]:
    texts = train_df[train_df["label_name"] == label]["text"]
    for n in [1, 2, 3]:
        top_items = get_top_ngrams(texts, n=n, top_k=20)
        for rank, (ngram, count) in enumerate(top_items, start=1):
            results.append({
                "group": label,
                "ngram_n": n,
                "rank": rank,
                "ngram": ngram,
                "count": count
            })

result_df = pd.DataFrame(results)
result_df.to_csv(OUTPUT_DIR / "top_words_ngrams_train.csv", index=False, encoding="utf-8-sig")

print(result_df.head(60).to_string(index=False))
print("\nSaved:")
print(OUTPUT_DIR / "top_words_ngrams_train.csv")