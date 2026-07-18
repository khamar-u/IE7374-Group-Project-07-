"""
Text preprocessing module for the adverse event summarizer pipeline.

Produces two processed text columns from raw patient reviews:
  - review_clean    : minimally cleaned natural language, used for transformer models
                      (DistilBERT and BART expect this format)
  - review_processed: fully tokenized, stop-word filtered, and lemmatized text,
                      used for classical NLP methods (LDA, word frequency, n-grams)

Also handles:
  - HTML entity decoding
  - Deduplication
  - Missing condition label handling
  - Date parsing
  - Drug name normalization
  - Train / validation / test split (70/15/15) on the focused subset

Role 2 deliverable — Yosephine Tong
"""

import html
import re
import os
import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, save_dataframe, print_section

# Download required NLTK data on first run.
# nltk.download() is idempotent — skips silently if already present.
for resource in ["punkt_tab", "stopwords", "wordnet", "omw-1.4", "vader_lexicon"]:
    nltk.download(resource, quiet=True)

# Words that carry negative meaning and must NOT be removed as stop words.
# Removing "not" or "never" from "not helpful" would flip the sentiment.
NEGATION_WORDS = {
    "no", "not", "never", "neither", "nor", "none",
    "nobody", "nothing", "nowhere", "without", "n't"
}

_lemmatizer = WordNetLemmatizer()
_stop_words  = (set(stopwords.words("english")) - NEGATION_WORDS)


# ── Low-level text cleaning ───────────────────────────────────────────────────

def _decode_html(text):
    """
    Decode HTML entities (&#039; -> ', &amp; -> &) and strip the surrounding
    double-quotes that were added when the data was scraped from drugs.com.
    """
    text = html.unescape(str(text))
    text = text.strip().strip('"')
    return text.strip()


def _remove_noise(text):
    """Remove leftover HTML tags and non-printable characters."""
    text = re.sub(r"<[^>]+>", " ", text)          # strip any residual HTML tags
    text = re.sub(r"[^\x20-\x7E]", " ", text)     # remove non-ASCII printable chars
    text = re.sub(r"\s+", " ", text)              # collapse multiple spaces
    return text.strip()


# ── Column-level transformations ─────────────────────────────────────────────

def make_review_clean(text):
    """
    Minimal cleaning for transformer input.

    Decodes HTML entities, strips surrounding quotes, removes leftover tags,
    and collapses whitespace. Preserves natural language structure so that
    DistilBERT and BART can read it without losing context.
    """
    return _remove_noise(_decode_html(text))


def make_review_processed(text):
    """
    Full preprocessing for classical NLP (LDA, n-grams, word frequency).

    Applies make_review_clean, then lowercases, tokenizes, removes stop words
    (keeping negation words), and lemmatizes. Returns a single joined string.
    """
    cleaned = make_review_clean(text)
    tokens  = word_tokenize(cleaned.lower())
    tokens  = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok.isalpha() and (tok not in _stop_words or tok in NEGATION_WORDS)
    ]
    return " ".join(tokens)


# ── Dataframe-level cleaning ──────────────────────────────────────────────────

def handle_missing_conditions(df):
    """
    Fill missing condition labels with 'Unknown' instead of dropping the rows.
    The review text is still valid for adverse event detection even without a label.
    """
    n_missing = df["condition"].isna().sum()
    if n_missing > 0:
        df["condition"] = df["condition"].fillna("Unknown")
        print(f"  Filled {n_missing:,} missing condition labels with 'Unknown'")
    return df


def remove_duplicates(df):
    """Drop rows with identical review text. Keep the first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=["review"], keep="first").copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Removed {dropped:,} duplicate reviews")
    return df


def normalize_drug_names(df):
    """Title-case drug names to fix inconsistencies (LEVOTHYROXINE -> Levothyroxine)."""
    df["drugName"] = df["drugName"].str.strip().str.title()
    return df


def parse_dates(df):
    """Parse the date column from string to datetime."""
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y", errors="coerce")
    return df


# ── Main preprocessing pipeline ──────────────────────────────────────────────

def preprocess_dataframe(df):
    """
    Run the full cleaning and preprocessing pipeline on the merged dataframe.

    Steps:
      1. Fill missing condition labels
      2. Remove duplicate reviews
      3. Normalize drug name casing
      4. Parse dates
      5. Produce review_clean  (for transformers)
      6. Produce review_processed (for classical NLP)

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with review_clean and review_processed columns added.
    """
    print_section("Preprocessing dataframe")

    df = handle_missing_conditions(df)
    df = remove_duplicates(df)
    df = normalize_drug_names(df)
    df = parse_dates(df)

    print("  Generating review_clean  (transformer input)...")
    tqdm.pandas(desc="  review_clean")
    df["review_clean"] = df["review"].progress_apply(make_review_clean)

    # Drop reviews that are too short to carry meaningful signal
    config = load_config()
    min_len = config["preprocessing"]["min_review_length"]
    df["review_word_count"] = df["review_clean"].apply(lambda x: len(x.split()))
    before = len(df)
    df = df[df["review_word_count"] >= min_len].copy()
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped:,} reviews shorter than {min_len} words")

    print("  Generating review_processed (classical NLP — this takes ~10 min)...")
    tqdm.pandas(desc="  review_processed")
    df["review_processed"] = df["review"].progress_apply(make_review_processed)

    print(f"  Done. Final shape: {df.shape}")
    return df


# ── Train / validation / test split ──────────────────────────────────────────

def split_focused_subset(df_focused, config_path="configs/model_config.yaml"):
    """
    Split the focused (Depression + Anxiety) subset into train, validation,
    and test sets using a 70 / 15 / 15 ratio.

    The test set is held out and should not be used for any model tuning.

    Parameters
    ----------
    df_focused : pd.DataFrame
        Filtered dataframe containing only the focus conditions.

    Returns
    -------
    tuple of pd.DataFrame
        (df_train, df_val, df_test)
    """
    config    = load_config(config_path)
    ratios    = config["preprocessing"]["split_ratios"]
    seed      = config["preprocessing"]["random_seed"]
    val_ratio = ratios["val"] / (ratios["val"] + ratios["test"])

    df_train, df_temp = train_test_split(
        df_focused, test_size=(1 - ratios["train"]), random_state=seed, shuffle=True
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=(1 - val_ratio), random_state=seed
    )

    print_section("Train / val / test split")
    print(f"  Train : {len(df_train):,} rows ({ratios['train']*100:.0f}%)")
    print(f"  Val   : {len(df_val):,}  rows ({ratios['val']*100:.0f}%)")
    print(f"  Test  : {len(df_test):,}  rows ({ratios['test']*100:.0f}%)")
    print("  (Test set held out — do not use for model tuning)")

    return df_train, df_val, df_test


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import load_raw_data, get_focused_subset

    config = load_config()

    df       = load_raw_data()
    df_clean = preprocess_dataframe(df)

    df_focused = get_focused_subset(df_clean)
    df_train, df_val, df_test = split_focused_subset(df_focused)

    save_dataframe(df_clean,   config["data"]["clean_output"])
    save_dataframe(df_focused, config["data"]["focused_output"])

    print("\nPreprocessing complete.")
    print(f"  df_clean   -> {config['data']['clean_output']}")
    print(f"  df_focused -> {config['data']['focused_output']}")
