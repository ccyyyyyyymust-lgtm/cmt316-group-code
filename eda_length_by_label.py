from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(r"X:\CMT316\CW2\DATASET")
OUTPUT_TABLE_DIR = Path(r"X:\CMT316\CW2\outputs\tables")
OUTPUT_FIG_DIR = Path(r"X:\CMT316\CW2\outputs\figures")
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIG_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

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

train_df = load_split("train_text.txt", "train_labels.txt", "train")

train_df["char_len"] = train_df["text"].apply(len)
train_df["word_count"] = train_df["text"].apply(lambda x: len(x.split()))

length_stats = train_df.groupby("label_name")[["char_len", "word_count"]].describe()
length_stats.to_csv(OUTPUT_TABLE_DIR / "train_length_stats_by_label.csv", encoding="utf-8-sig")

print(length_stats)

# boxplot: word_count by label
plt.figure(figsize=(8, 5))
train_df.boxplot(column="word_count", by="label_name")
plt.title("Word Count Distribution by Label (Train)")
plt.suptitle("")
plt.xlabel("Label")
plt.ylabel("Word Count")
plt.tight_layout()
plt.savefig(OUTPUT_FIG_DIR / "train_word_count_boxplot_by_label.png")
plt.close()

labels = ["negative", "neutral", "positive"]

plt.figure(figsize=(8, 5))
for label in labels:
    subset = train_df[train_df["label_name"] == label]
    plt.hist(subset["word_count"], bins=30, alpha=0.5, label=label)

plt.title("Word Count Histogram by Label (Train)")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_FIG_DIR / "train_word_count_hist_by_label.png")
plt.close()

# char_len boxplot
plt.figure(figsize=(8, 5))
train_df.boxplot(column="char_len", by="label_name")
plt.title("Character Length Distribution by Label (Train)")
plt.suptitle("")
plt.xlabel("Label")
plt.ylabel("Character Length")
plt.tight_layout()
plt.savefig(OUTPUT_FIG_DIR / "train_char_length_boxplot_by_label.png")
plt.close()

print("\nSaved:")
print(OUTPUT_TABLE_DIR / "train_length_stats_by_label.csv")
print(OUTPUT_FIG_DIR / "train_word_count_boxplot_by_label.png")
print(OUTPUT_FIG_DIR / "train_word_count_hist_by_label.png")
print(OUTPUT_FIG_DIR / "train_char_length_boxplot_by_label.png")