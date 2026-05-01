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
    print("[WARN] contractions not installed. Run: pip3 install contractions")

try:
    import emoji
    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False
    print("[WARN] emoji not installed. Run: pip3 install emoji")

print("All imports loaded")
DATA_DIR = "."
OUTPUT_DIR = "outputs/preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REMOVE_STOPWORDS = False
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}

# Verify files exist
for f in ["train_text.txt", "train_labels.txt", "val_text.txt",
          "val_labels.txt", "test_text.txt", "test_labels.txt"]:
    path = os.path.join(DATA_DIR, f)
    status = "Found" if os.path.exists(path) else "MISSING"
    print(f"  {status}: {path}")
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
print(f"Stopword list: {len(STOPWORDS)} words (negations excluded)")
def lowercase(text):
    return text.lower()


def replace_user_mentions(text):
    """Replace @user mentions with a plain lowercase placeholder.
    Using usermention (no @ or uppercase) so it survives
    special-character removal and lowercasing without changing form."""
    return re.sub(r"@\w+", "usermention", text)


def replace_urls(text):
    """Replace URLs with a plain lowercase placeholder."""
    text = re.sub(r"https?://\S+", "httpurl", text)
    text = re.sub(r"www\.\S+", "httpurl", text)
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


def decode_unicode_escapes(text):
    """Decode unicode escape sequences like \\u002c \\u2019 etc.
    These appear in some tweets and would otherwise become noise."""
    def _replace_escape(match):
        try:
            return chr(int(match.group(1), 16))
        except ValueError:
            return match.group(0)
    return re.sub(r"\\u([0-9a-fA-F]{4})", _replace_escape, text)


def clean_hashtags(text):
    """Remove # but keep the word, splitting CamelCase.
    Must run BEFORE lowercase so CamelCase boundaries are visible.
    E.g. #HappyPrimeDay -> Happy Prime Day"""
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
    """Apply full preprocessing pipeline to one tweet.
    Order matters:
      - Unicode escapes decoded first (raw noise removal)
      - Hashtag splitting BEFORE lowercase (needs CamelCase intact)
      - Mentions/URLs replaced with lowercase tokens that survive cleaning
    """
    text = decode_unicode_escapes(text)
    text = clean_hashtags(text)
    text = lowercase(text)
    text = replace_user_mentions(text)
    text = replace_urls(text)
    text = expand_contractions_fn(text)
    text = convert_emojis(text)
    text = decode_html_entities(text)
    text = remove_special_chars(text)
    text = collapse_whitespace(text)
    if REMOVE_STOPWORDS:
        text = remove_stopwords_fn(text)
    return text

print("All preprocessing functions defined")
def load_split(text_file, label_file, split_name):
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

train_df = load_split("train_text.txt", "train_labels.txt", "train")
val_df   = load_split("val_text.txt",   "val_labels.txt",   "val")
test_df  = load_split("test_text.txt",  "test_labels.txt",  "test")

print(f"Train: {len(train_df)} samples")
print(f"Val:   {len(val_df)} samples")
print(f"Test:  {len(test_df)} samples")
print(f"\nLabel distribution (train):")
print(train_df["label_name"].value_counts())
train_df["clean_text"] = train_df["text"].apply(preprocess_tweet)
val_df["clean_text"]   = val_df["text"].apply(preprocess_tweet)
test_df["clean_text"]  = test_df["text"].apply(preprocess_tweet)

print("Preprocessing complete!")
sample = train_df.sample(n=min(8, len(train_df)), random_state=42)
for _, row in sample.iterrows():
    print(f"LABEL:  {row['label_name']}")
    print(f"BEFORE: {row['text'][:120]}")
    print(f"AFTER:  {row['clean_text'][:120]}")
    print("-" * 80)

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

for name, df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
    before = compute_stats(df, "text")
    after  = compute_stats(df, "clean_text")
    print(f"\n{'='*55}")
    print(f"  {name} ({len(df)} samples)")
    print(f"{'='*55}")
    print(f"  {'Metric':<22s}  {'Before':>8s}  {'After':>8s}")
    print(f"  {'-'*45}")
    print(f"  {'Avg word count':<22s}  {before['avg_words']:>8.2f}  {after['avg_words']:>8.2f}")
    print(f"  {'Median word count':<22s}  {before['med_words']:>8.2f}  {after['med_words']:>8.2f}")
    print(f"  {'Avg char length':<22s}  {before['avg_chars']:>8.2f}  {after['avg_chars']:>8.2f}")
    print(f"  {'Median char length':<22s}  {before['med_chars']:>8.2f}  {after['med_chars']:>8.2f}")
    print(f"  {'Empty after cleaning':<22s}  {before['empty']:>8d}  {after['empty']:>8d}")
    train_df["wc_before"] = train_df["text"].apply(lambda x: len(x.split()))
train_df["wc_after"]  = train_df["clean_text"].apply(lambda x: len(x.split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label in ["negative", "neutral", "positive"]:
    subset = train_df[train_df["label_name"] == label]
    axes[0].hist(subset["wc_before"], bins=30, alpha=0.5, label=label)
    axes[1].hist(subset["wc_after"], bins=30, alpha=0.5, label=label)

axes[0].set_title("Word Count BEFORE Preprocessing")
axes[0].set_xlabel("Word Count")
axes[0].set_ylabel("Frequency")
axes[0].legend()
axes[1].set_title("Word Count AFTER Preprocessing")
axes[1].set_xlabel("Word Count")
axes[1].set_ylabel("Frequency")
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "word_count_before_after.png"), dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved")
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    out_path = os.path.join(OUTPUT_DIR, f"{name}_preprocessed.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {out_path}")

print("\nDone! All preprocessed files are in outputs/preprocessed/")
