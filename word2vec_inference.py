
from pathlib import Path
import html
import re
import string

import joblib
import nltk
import numpy as np
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer


# ============================================================
# Paths and model configuration
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "model" / "comparison" / "word2vec"

WORD2VEC_PATH = MODEL_DIR / "word2vec_embeddings.model"
CLASSIFIER_PATH = MODEL_DIR / "word2vec_linearsvc.joblib"
CONFIG_PATH = MODEL_DIR / "word2vec_config.json"

VECTOR_SIZE = 100


# ============================================================
# NLTK resources
# ============================================================

def ensure_nltk_data():
    resources = {
        "tokenizers": ["punkt", "punkt_tab"],
        "corpora": ["stopwords", "wordnet", "omw-1.4"],
        "taggers": [
            "averaged_perceptron_tagger",
            "averaged_perceptron_tagger_eng",
        ],
    }

    for category, names in resources.items():
        for name in names:
            try:
                nltk.data.find(f"{category}/{name}")
            except LookupError:
                if not nltk.download(name, quiet=True):
                    raise RuntimeError(
                        f"Could not download NLTK resource: {name}"
                    )


ensure_nltk_data()


# ============================================================
# Load trained models
# ============================================================

if not WORD2VEC_PATH.exists():
    raise FileNotFoundError(f"Word2Vec model not found: {WORD2VEC_PATH}")

if not CLASSIFIER_PATH.exists():
    raise FileNotFoundError(f"LinearSVC model not found: {CLASSIFIER_PATH}")

word2vec_model = Word2Vec.load(str(WORD2VEC_PATH))
classifier = joblib.load(CLASSIFIER_PATH)

if word2vec_model.vector_size != VECTOR_SIZE:
    raise ValueError("Unexpected Word2Vec vector size.")

if classifier.n_features_in_ != VECTOR_SIZE:
    raise ValueError("Classifier feature count does not match Word2Vec.")

stop_words = set(stopwords.words("english"))
stop_words -= {"no", "nor", "not", "never", "neither"}

lemmatizer = WordNetLemmatizer()


# ============================================================
# POS mapping (same as training)
# ============================================================

def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


# ============================================================
# Text preprocessing (same as prepare_data.py)
# ============================================================

def clean_text(text):
    text = BeautifulSoup(
        html.unescape(str(text)),
        "html.parser"
    ).get_text(" ")

    text = text.lower()

    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)
    text = re.sub(r"n't\b", " not", text)

    text = text.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )

    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)

    cleaned = []

    for word, tag in tagged:
        if word in stop_words:
            continue

        cleaned.append(
            lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        )

    negations = {
        "not", "no", "never", "neither", "nor", "cannot", "cant",
        "wont", "dont", "didnt", "isnt", "arent", "wasnt", "werent",
        "hasnt", "havent", "hadnt", "wouldnt", "shouldnt", "couldnt"
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


# ============================================================
# Word2Vec document vector
# ============================================================

def get_sentence_vector(cleaned_text):
    tokens = cleaned_text.split()

    vectors = [
        word2vec_model.wv[word]
        for word in tokens
        if word in word2vec_model.wv
    ]

    if not vectors:
        return np.zeros(VECTOR_SIZE, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


# ============================================================
# Sentiment prediction
# ============================================================

def predict_sentiment(review):
    cleaned = clean_text(review)

    vector = get_sentence_vector(cleaned)
    vector = vector.reshape(1, -1)

    prediction = int(classifier.predict(vector)[0])
    decision_score = float(classifier.decision_function(vector)[0])

    sentiment = "Positive" if prediction == 1 else "Negative"

    # This is a normalized decision score, NOT a calibrated probability.
    confidence = (
        100.0 / (1.0 + np.exp(-min(abs(decision_score), 50.0)))
    )

    return {
        "sentiment": sentiment,
        "confidence": round(float(confidence), 2),
        "decision_score": decision_score,
        "cleaned_text": cleaned,
    }


if __name__ == "__main__":
    print("Word2Vec + LinearSVC inference engine")
    print("Vocabulary size:", len(word2vec_model.wv))
    print("Classifier:", type(classifier).__name__)
    print("Classifier C:", classifier.C)

    test_reviews = [
        "This movie was absolutely wonderful. I loved every minute!",
        "This was a terrible movie. I hated it.",
        "The movie was not good at all.",
        "I did not hate this movie. It was surprisingly enjoyable.",
    ]

    for review in test_reviews:
        result = predict_sentiment(review)

        print("\nReview:", review)
        print("Cleaned:", result["cleaned_text"])
        print("Sentiment:", result["sentiment"])
        print("Confidence score:", result["confidence"])
        print("Decision score:", result["decision_score"])
