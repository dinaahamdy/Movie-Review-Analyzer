"""
Create one leakage-safe, shared train/validation/test split for SVM, LSTM and ELMo.
Run from the project root:
    python training/prepare_data.py

Inputs:
    dataset/IMDB Dataset.csv

Outputs:
    dataset/experiments/train.csv
    dataset/experiments/validation.csv
    dataset/experiments/test.csv

Each output contains:
    raw_review, clean_review, sentiment, label
"""
from pathlib import Path
import re
import string
import html

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dataset" / "IMDB Dataset.csv"
OUTPUT_DIR = ROOT / "dataset" / "experiments"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.10  # fraction of the full dataset


def ensure_nltk_data():
    resources = {
        "tokenizers": ["punkt", "punkt_tab"],
        "corpora": ["stopwords", "wordnet", "omw-1.4"],
        "taggers": ["averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"],
    }
    for category, names in resources.items():
        for name in names:
            try:
                nltk.data.find(f"{category}/{name}")
            except LookupError:
                print(f"Downloading NLTK resource: {name}")
                nltk.download(name, quiet=False)


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def clean_for_classical_and_lstm(text, stop_words, lemmatizer):
    text = BeautifulSoup(html.unescape(str(text)), "html.parser").get_text(" ")
    text = text.lower()
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)
    text = re.sub(r"n't\b", " not", text)
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    cleaned = []
    for word, tag in tagged:
        if word in stop_words:
            continue
        cleaned.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))

    negations = {
        "not", "no", "never", "neither", "nor", "cannot", "cant", "wont",
        "dont", "didnt", "isnt", "arent", "wasnt", "werent", "hasnt",
        "havent", "hadnt", "wouldnt", "shouldnt", "couldnt"
    }
    result = []
    i = 0
    while i < len(cleaned):
        word = cleaned[i]
        if word in negations and i + 1 < len(cleaned):
            result.append(f"{word}_{cleaned[i + 1]}")
            i += 2
        else:
            result.append(word)
            i += 1
    return " ".join(result)


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")

    ensure_nltk_data()
    print(f"Loading {SOURCE} ...")
    df = pd.read_csv(SOURCE)
    required = {"review", "sentiment"}
    if not required.issubset(df.columns):
        raise ValueError(f"Expected columns {required}; found {set(df.columns)}")

    df = df.dropna(subset=["review", "sentiment"]).copy()
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    df = df[df["sentiment"].isin(["positive", "negative"])].copy()
    df["raw_review"] = df["review"].astype(str)
    df = df.drop(columns=["review"]).drop_duplicates().reset_index(drop=True)

    stop_words = set(stopwords.words("english"))
    stop_words -= {"no", "nor", "not", "never", "neither"}
    lemmatizer = WordNetLemmatizer()

    print(f"Preprocessing {len(df):,} reviews. This may take a while...")
    cleaned = []
    for i, review in enumerate(df["raw_review"], start=1):
        cleaned.append(clean_for_classical_and_lstm(review, stop_words, lemmatizer))
        if i % 5000 == 0:
            print(f"  Processed {i:,}/{len(df):,}")

    df["clean_review"] = cleaned
    df = df[df["clean_review"].str.strip().ne("")].copy()
    # Match the original preprocessing behavior: remove duplicate rows after normalization.
    df = df.drop_duplicates(subset=["clean_review", "sentiment"]).reset_index(drop=True)
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1}).astype("int8")

    # 70% train, 10% validation, 20% test; stratified at both stages.
    train_df, temp_df = train_test_split(
        df, test_size=TEST_SIZE + VALIDATION_SIZE,
        random_state=RANDOM_STATE, stratify=df["label"]
    )
    relative_test = TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE)
    validation_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        random_state=RANDOM_STATE, stratify=temp_df["label"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    columns = ["raw_review", "clean_review", "sentiment", "label"]
    for name, part in [
        ("train.csv", train_df),
        ("validation.csv", validation_df),
        ("test.csv", test_df),
    ]:
        path = OUTPUT_DIR / name
        part[columns].to_csv(path, index=False, encoding="utf-8")
        print(f"Saved {name}: {len(part):,} rows -> {path}")

    print("\nSplit summary:")
    for name, part in [("Train", train_df), ("Validation", validation_df), ("Test", test_df)]:
        print(f"\n{name}: {len(part):,}")
        print(part["sentiment"].value_counts().to_string())

    print("\nDone. Use these exact three files for every model experiment.")


if __name__ == "__main__":
    main()
