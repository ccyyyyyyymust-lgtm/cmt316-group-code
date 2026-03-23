from pathlib import Path
import pandas as pd
import random

DATA_DIR = Path(r"X:\CMT316\CW2\DATASET")
OUTPUT_DIR = Path(r"X:\CMT316\CW2\outputs\tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


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


def sample_random_examples(df, n=15):
    n = min(n, len(df))
    return df.sample(n=n, random_state=RANDOM_SEED).copy()


def sample_by_label(df, n_per_label=8):
    samples = []
    for label_id, label_name in LABEL_MAP.items():
        sub = df[df["label_id"] == label_id]
        n = min(n_per_label, len(sub))
        sampled = sub.sample(n=n, random_state=RANDOM_SEED).copy()
        samples.append(sampled)
    return pd.concat(samples, ignore_index=True)


def find_conflicting_duplicates(df):
    dup_df = df[df["text"].duplicated(keep=False)].copy()
    if dup_df.empty:
        return pd.DataFrame(columns=df.columns)

    label_counts = dup_df.groupby("text")["label_id"].nunique()
    conflict_texts = label_counts[label_counts > 1].index

    conflict_df = dup_df[dup_df["text"].isin(conflict_texts)].copy()
    return conflict_df.sort_values(["text", "label_id", "row_id"])


def sample_short_texts(df, n=10):
    temp = df.copy()
    temp["word_count"] = temp["text"].apply(lambda x: len(x.split()))
    short_df = temp.sort_values(["word_count", "row_id"]).head(n).copy()
    return short_df


def sample_long_texts(df, n=10):
    temp = df.copy()
    temp["word_count"] = temp["text"].apply(lambda x: len(x.split()))
    long_df = temp.sort_values(["word_count", "row_id"], ascending=[False, True]).head(n).copy()
    return long_df


def save_and_print(df, filename, title):
    out_path = OUTPUT_DIR / filename
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n=== {title} ===")
    print(df[["split", "row_id", "label_id", "label_name", "text"]].head(20).to_string(index=False))
    print(f"\nSaved to: {out_path}")


def main():
    train_df = load_split("train_text.txt", "train_labels.txt", "train")
    val_df = load_split("val_text.txt", "val_labels.txt", "val")
    test_df = load_split("test_text.txt", "test_labels.txt", "test")

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    random_sample = sample_random_examples(train_df, n=15)
    save_and_print(random_sample, "audit_random_train_sample.csv", "Random train sample")

    by_label_sample = sample_by_label(train_df, n_per_label=8)
    save_and_print(by_label_sample, "audit_balanced_train_sample.csv", "Balanced sample by label")

    conflict_df = find_conflicting_duplicates(train_df)
    if conflict_df.empty:
        print("\nNo duplicated texts with conflicting labels found in train split.")
    else:
        save_and_print(conflict_df, "audit_conflicting_duplicates_train.csv", "Conflicting duplicate texts in train")

    short_sample = sample_short_texts(train_df, n=10)
    save_and_print(short_sample, "audit_short_texts_train.csv", "Shortest train texts")

    long_sample = sample_long_texts(train_df, n=10)
    save_and_print(long_sample, "audit_long_texts_train.csv", "Longest train texts")

    audit_pool = pd.concat(
        [random_sample, by_label_sample, short_sample, long_sample],
        ignore_index=True
    ).drop_duplicates(subset=["split", "row_id"]).copy()

    audit_pool["human_check"] = ""
    audit_pool["notes"] = ""

    audit_pool_path = OUTPUT_DIR / "audit_pool_for_manual_check.csv"
    audit_pool.to_csv(audit_pool_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved manual audit pool to: {audit_pool_path}")

    print("\nDone. You can open the CSV files and manually inspect whether the labels look reasonable.")


if __name__ == "__main__":
    main()