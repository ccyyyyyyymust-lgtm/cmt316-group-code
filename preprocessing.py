import os
import re
import html
import pandas as pd
import matplotlib.pyplot as plt
 
try:
    import contractions
    HAS_CONTRACTIONS = True
except ImportError:
    HAS_CONTRACTIONS = False
    print("[WARN] 'contractions' not installed. Run: pip install contractions")
 
try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False
    print("[WARN] 'emoji' not installed. Run: pip install emoji")
 
 
# ── CONFIGURATION ────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "DATASET")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "preprocessed")
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
REMOVE_STOPWORDS = False  # OFF by default — negation words matter for sentiment
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
 
 
# ── STOPWORD LIST (negations deliberately kept) ─────────────────────────
STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because", "as",
    "until", "while", "of", "at", "by", "for", "with", "about",
    "between", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "both", "each", "few",
    "more", "most", "other", "some", "such", "own", "same",
    "than", "too", "very", "s", "t", "just", "should", "now",
    "d", "ll", "m", "o", "re", "ve", "y",
    "will", "would", "could", "shall", "may", "might",
}
 
 
# ── PREPROCESSING FUNCTIONS ─────────────────────────────────────────────
 
def lowercase(text):
    return text.lower()
 
 
def replace_user_mentions(text):
    return re.sub(r"@\w+", "@USER", text)
 
 
def replace_urls(text):
    text = re.sub(r"https?://\S+", "HTTPURL", text)
    text = re.sub(r"www\.\S+", "HTTPURL", text)
    return text
 
 
def expand_contractions_fn(text):
    if HAS_CONTRACTIONS:
        return contractions.fix(text)
    return text
 
 
def convert_emojis(text):
    if HAS_EMOJI:
        return emoji.demojize(text, delimiters=(" ", " "))
    return text
 
 
def decode_html_entities(text):
    return html.unescape(text)
 
 
def clean_hashtags(text):
    def _split_camel(match):
        tag = match.group(1)
        return re.sub(r"([a-z])([A-Z])", r"\1 \2", tag)
    return re.sub(r"#(\w+)", _split_camel, text)
 
 
def remove_special_chars(text):
    return re.sub(r"[^a-zA-Z0-9\s]", " ", text)
 
 
def collapse_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()
 
 
def remove_stopwords_fn(text):
    tokens = text.split()
    return " ".join(w for w in tokens if w not in STOPWORDS)
 
 
def preprocess_tweet(text):
    """Apply full preprocessing pipeline to one tweet."""
    text = lowercase(text)
    text = replace_user_mentions(text)
    text = replace_urls(text)
    text = expand_contractions_fn(text)
    text = convert_emojis(text)
    text = decode_html_entities(text)
    text = clean_hashtags(text)
    text = remove_special_chars(text)
    text = collapse_whitespace(text)
    if REMOVE_STOPWORDS:
        text = remove_stopwords_fn(text)
    return text
 
 
# ── DATA LOADING ─────────────────────────────────────────────────────────
 
def load_split(text_file, label_file, split_name):
    """Load one split (train/val/test) from .txt files."""
    text_path = os.path.join(DATA_DIR, text_file)
    label_path = os.path.join(DATA_DIR, label_file)
 
    with open(text_path, "r", encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f]
 
    with open(label_path, "r", encoding="utf-8") as f:
        labels = [int(line.strip()) for line in f if line.strip()]
 
    assert len(texts) == len(labels), \
        f"Mismatch in {split_name}: {len(texts)} texts vs {len(labels)} labels"
 
    df = pd.DataFrame({"text": texts, "label_id": labels})
    df["label_name"] = df["label_id"].map(LABEL_MAP)
    df["split"] = split_name
    return df
 
 
# ── STATISTICS ───────────────────────────────────────────────────────────
 
def compute_stats(df, col):
    wc = df[col].apply(lambda x: len(str(x).split()))
    cl = df[col].apply(lambda x: len(str(x)))
    return {
        "avg_words": round(wc.mean(), 2),
        "med_words": round(wc.median(), 2),
        "avg_chars": round(cl.mean(), 2),
        "med_chars": round(cl.median(), 2),
        "empty":     int((df[col].str.strip() == "").sum()),
    }
 
 
def print_report(df, split_name):
    """Print before vs after statistics for one split."""
    before = compute_stats(df, "text")
    after = compute_stats(df, "clean_text")
 
    print(f"\n{'='*55}")
    print(f"  {split_name.upper()} ({len(df)} samples)")
    print(f"{'='*55}")
    print(f"  Label distribution:")
    for lid in sorted(LABEL_MAP):
        count = (df["label_id"] == lid).sum()
        pct = count / len(df) * 100
        print(f"    {LABEL_MAP[lid]:>10s}: {count:>6d} ({pct:.1f}%)")
 
    print(f"\n  {'Metric':<22s}  {'Before':>8s}  {'After':>8s}")
    print(f"  {'-'*45}")
    print(f"  {'Avg word count':<22s}  {before['avg_words']:>8.2f}  {after['avg_words']:>8.2f}")
    print(f"  {'Median word count':<22s}  {before['med_words']:>8.2f}  {after['med_words']:>8.2f}")
    print(f"  {'Avg char length':<22s}  {before['avg_chars']:>8.2f}  {after['avg_chars']:>8.2f}")
    print(f"  {'Median char length':<22s}  {before['med_chars']:>8.2f}  {after['med_chars']:>8.2f}")
    print(f"  {'Empty after cleaning':<22s}  {before['empty']:>8d}  {after['empty']:>8d}")
 
    print(f"\n  Sample before -> after:")
    sample = df.sample(n=min(5, len(df)), random_state=42)
    for _, row in sample.iterrows():
        print(f"    BEFORE: {row['text'][:100]}")
        print(f"    AFTER:  {row['clean_text'][:100]}")
        print()
 
 
# ── PLOTTING ─────────────────────────────────────────────────────────────
 
def plot_word_counts(df, split_name):
    """Plot word count distributions before vs after preprocessing."""
    df["wc_before"] = df["text"].apply(lambda x: len(x.split()))
    df["wc_after"] = df["clean_text"].apply(lambda x: len(x.split()))
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    labels_list = ["negative", "neutral", "positive"]
 
    for label in labels_list:
        subset = df[df["label_name"] == label]
        axes[0].hist(subset["wc_before"], bins=30, alpha=0.5, label=label)
        axes[1].hist(subset["wc_after"], bins=30, alpha=0.5, label=label)
 
    axes[0].set_title(f"Word Count BEFORE Preprocessing ({split_name})")
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Frequency")
    axes[0].legend()
 
    axes[1].set_title(f"Word Count AFTER Preprocessing ({split_name})")
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Frequency")
    axes[1].legend()
 
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, f"word_count_{split_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {fig_path}")
 
 
# ── MAIN ─────────────────────────────────────────────────────────────────
 
def main():
    splits = {
        "train": ("train_text.txt", "train_labels.txt"),
        "val":   ("val_text.txt",   "val_labels.txt"),
        "test":  ("test_text.txt",  "test_labels.txt"),
    }
 
    for split_name, (text_f, label_f) in splits.items():
        print(f"Processing {split_name}...")
 
        df = load_split(text_f, label_f, split_name)
        df["clean_text"] = df["text"].apply(preprocess_tweet)
 
        print_report(df, split_name)
        plot_word_counts(df, split_name)
 
        out_path = os.path.join(OUTPUT_DIR, f"{split_name}_preprocessed.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  CSV saved: {out_path}\n")
 
    print("Done! All preprocessed files saved to outputs/preprocessed/")
 
 
if __name__ == "__main__":
    main()
