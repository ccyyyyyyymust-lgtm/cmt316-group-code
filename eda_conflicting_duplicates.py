from pathlib import Path
import pandas as pd

DATA_DIR = Path(r"X:\CMT316\CW2\DATASET")
OUTPUT_DIR = Path(r"X:\CMT316\CW2\outputs\tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    assert len(texts) == len(labels), f"{split_name}: text/label count mismatch"

    df = pd.DataFrame({
        "text": texts,
        "label_id": labels
    })
    df["label_name"] = df["label_id"].map(LABEL_MAP)
    df["split"] = split_name
    df["row_id"] = range(len(df))
    return df

train_df = load_split("train_text.txt", "train_labels.txt", "train")

dup_df = train_df[train_df["text"].duplicated(keep=False)].copy()
label_counts = dup_df.groupby("text")["label_id"].nunique()
conflict_texts = label_counts[label_counts > 1].index
conflict_df = dup_df[dup_df["text"].isin(conflict_texts)].copy()

conflict_df = conflict_df.sort_values(["text", "label_id", "row_id"])

# 保存完整表
conflict_df.to_csv(
    OUTPUT_DIR / "conflicting_duplicates_detailed.csv",
    index=False,
    encoding="utf-8-sig"
)

# 生成简化展示表：每条文本有哪些标签
summary_rows = []
for text, group in conflict_df.groupby("text"):
    summary_rows.append({
        "text": text,
        "row_ids": "; ".join(map(str, group["row_id"].tolist())),
        "label_ids": "; ".join(map(str, group["label_id"].tolist())),
        "label_names": "; ".join(group["label_name"].tolist()),
        "num_versions": len(group)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(
    OUTPUT_DIR / "conflicting_duplicates_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Conflicting duplicate texts:")
print(summary_df.to_string(index=False))

print("\nSaved:")
print(OUTPUT_DIR / "conflicting_duplicates_detailed.csv")
print(OUTPUT_DIR / "conflicting_duplicates_summary.csv")