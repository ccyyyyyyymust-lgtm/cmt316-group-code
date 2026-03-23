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
val_df = load_split("val_text.txt", "val_labels.txt", "val")
test_df = load_split("test_text.txt", "test_labels.txt", "test")

all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

split_size_df = all_df.groupby("split").size().reset_index(name="count")
split_size_df["percentage"] = split_size_df["count"] / split_size_df["count"].sum() * 100
split_size_df.to_csv(OUTPUT_TABLE_DIR / "split_sizes.csv", index=False, encoding="utf-8-sig")

label_count_df = (
    all_df.groupby(["split", "label_name"])
    .size()
    .reset_index(name="count")
)

label_count_df["split_total"] = label_count_df.groupby("split")["count"].transform("sum")
label_count_df["percentage_within_split"] = label_count_df["count"] / label_count_df["split_total"] * 100
label_count_df.to_csv(OUTPUT_TABLE_DIR / "split_label_distribution.csv", index=False, encoding="utf-8-sig")

print("Split sizes:")
print(split_size_df.to_string(index=False))

print("\nSplit label distribution:")
print(label_count_df.to_string(index=False))

pivot_pct = label_count_df.pivot(index="split", columns="label_name", values="percentage_within_split")
pivot_pct = pivot_pct[["negative", "neutral", "positive"]]
pivot_pct.to_csv(OUTPUT_TABLE_DIR / "split_label_distribution_pivot.csv", encoding="utf-8-sig")

pivot_pct.plot(kind="bar", figsize=(8, 5))
plt.title("Label Percentage Within Each Split")
plt.xlabel("Split")
plt.ylabel("Percentage (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_FIG_DIR / "split_label_percentage_comparison.png")
plt.close()

print("\nSaved:")
print(OUTPUT_TABLE_DIR / "split_sizes.csv")
print(OUTPUT_TABLE_DIR / "split_label_distribution.csv")
print(OUTPUT_TABLE_DIR / "split_label_distribution_pivot.csv")
print(OUTPUT_FIG_DIR / "split_label_percentage_comparison.png")