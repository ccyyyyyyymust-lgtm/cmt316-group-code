from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"X:\CMT316\CW2\DATASET")

label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

def load_split(text_file, label_file, split_name):
    with open(DATA_DIR / text_file, "r", encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f]

    with open(DATA_DIR / label_file, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f]

    assert len(texts) == len(labels), f"{split_name}: text/label count mismatch"

    df = pd.DataFrame({
        "text": texts,
        "label_id": labels
    })
    df["label_name"] = df["label_id"].map(label_map)
    df["split"] = split_name
    return df

train_df = load_split("train_text.txt", "train_labels.txt", "train")
val_df   = load_split("val_text.txt", "val_labels.txt", "val")
test_df  = load_split("test_text.txt", "test_labels.txt", "test")

print("Train size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

print("\nTrain sample:")
print(train_df.head())
#step4
print("\nTrain size:", len(train_df))
print("Val size:", len(val_df))
print("Test size:", len(test_df))

print("\nTrain label distribution:")
print(train_df["label_name"].value_counts())

print("\nVal label distribution:")
print(val_df["label_name"].value_counts())

print("\nTest label distribution:")
print(test_df["label_name"].value_counts())
#step5
for df in [train_df, val_df, test_df]:
    df["char_len"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))

print("\nTrain character length stats:")
print(train_df["char_len"].describe())

print("\nTrain word count stats:")
print(train_df["word_count"].describe())

print("\nAverage word count by label (train):")
print(train_df.groupby("label_name")["word_count"].mean())
#step6
print("\nAverage word count by label (train):")
print(train_df.groupby("label_name")["word_count"].mean())

import matplotlib.pyplot as plt

label_counts = train_df["label_name"].value_counts()

plt.figure(figsize=(6, 4))
label_counts.plot(kind="bar")
plt.title("Train Label Distribution")
plt.xlabel("Sentiment Label")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(r"X:\CMT316\CW2\outputs\figures\train_label_distribution.png")
plt.show()

plt.figure(figsize=(6, 4))
train_df["word_count"].hist(bins=30)
plt.title("Train Tweet Word Count Distribution")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(r"X:\CMT316\CW2\outputs\figures\train_word_count_distribution.png")
plt.show()

avg_word_by_label = train_df.groupby("label_name")["word_count"].mean()

plt.figure(figsize=(6, 4))
avg_word_by_label.plot(kind="bar")
plt.title("Average Word Count by Sentiment Label (Train)")
plt.xlabel("Sentiment Label")
plt.ylabel("Average Word Count")
plt.tight_layout()
plt.savefig(r"X:\CMT316\CW2\outputs\figures\avg_word_count_by_label.png")
plt.show()
#step7
def has_mention(text):
    return "@user" in text

def has_hashtag(text):
    return "#" in text

def has_url(text):
    return "http" in text or "https" in text

for df in [train_df, val_df, test_df]:
    df["has_mention"] = df["text"].apply(has_mention)
    df["has_hashtag"] = df["text"].apply(has_hashtag)
    df["has_url"] = df["text"].apply(has_url)

print("\nTrain mention proportion:")
print(train_df["has_mention"].mean())

print("\nTrain hashtag proportion:")
print(train_df["has_hashtag"].mean())

print("\nTrain url proportion:")
print(train_df["has_url"].mean())

print("\nTweet-specific feature proportions by label (train):")
print(train_df.groupby("label_name")[["has_mention", "has_hashtag", "has_url"]].mean())
#step8
print("\nEmpty texts in train:")
print((train_df["text"].str.strip() == "").sum())

print("\nDuplicate texts in train:")
print(train_df["text"].duplicated().sum())

dup_df = train_df[train_df["text"].duplicated(keep=False)].sort_values("text")
conflict_df = dup_df.groupby("text")["label_id"].nunique()
conflict_texts = conflict_df[conflict_df > 1]

print("\nDuplicated texts with conflicting labels:")
print(len(conflict_texts))