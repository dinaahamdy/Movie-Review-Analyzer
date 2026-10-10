import os
import re
import string
import joblib
import nltk

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


# ============================================================
# Auto-download NLTK data
# ============================================================
def ensure_nltk_data():
    resources = {
        'tokenizers': ['punkt', 'punkt_tab'],
        'corpora': ['stopwords', 'wordnet', 'omw-1.4'],
        'taggers': ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng'],
    }
    for category, names in resources.items():
        for name in names:
            try:
                nltk.data.find(f'{category}/{name}')
            except LookupError:
                nltk.download(name, quiet=True)

ensure_nltk_data()


# ============================================================
# Initialize tools (نفس اللي في preprocessing.py)
# ============================================================
stop_words = set(stopwords.words("english"))
stop_words -= {"no", "nor", "not", "never", "neither"}

lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def remove_html(text):
    return BeautifulSoup(str(text), "html.parser").get_text(" ")


def normalize_contractions(text):
    text = text.lower()
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)
    text = re.sub(r"n't\b", " not", text)
    return text


def remove_punctuation(text):
    return text.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )

def handle_negation(text):
    """
    Attach negation words to the following word.
    Example: "not enjoy" -> "not_enjoy"
    """
    negation_words = {
        "not", "no", "never", "neither", "nor",
        "cannot", "cant", "wont", "dont", "didnt",
        "isnt", "arent", "wasnt", "werent",
        "hasnt", "havent", "hadnt",
        "wouldnt", "shouldnt", "couldnt"
    }

    tokens = text.split()
    result = []
    i = 0

    while i < len(tokens):
        word = tokens[i]
        if word in negation_words and i + 1 < len(tokens):
            combined = f"{word}_{tokens[i + 1]}"
            result.append(combined)
            i += 2
        else:
            result.append(word)
            i += 1

    return " ".join(result)

def clean_text(text):
    """نفس دالة preprocess_text في preprocessing.py"""
    text = remove_html(text).strip()
    if not text:
        return ""

    text = normalize_contractions(text)
    text = remove_punctuation(text)

    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    cleaned_tokens = []
    for word, tag in tagged_tokens:
        if word in stop_words:
            continue
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        cleaned_tokens.append(lemma)

    text = " ".join(cleaned_tokens)
    text = handle_negation(text)   # ← ضيفي السطر ده
    return text


# ============================================================
# Load model & vectorizer
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model', 'sentiment_model.pkl')
vectorizer_path = os.path.join(script_dir, 'model', 'tfidf_vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)


# ============================================================
# Predict function
# ============================================================
def predict_sentiment(review):
    cleaned = clean_text(review)

    if not cleaned:
        return "Unknown", 0.0, cleaned

    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probabilities = model.predict_proba(vectorized)[0]
    confidence = float(max(probabilities) * 100)

    sentiment = 'Positive 😊' if prediction == 1 else 'Negative 😞'

    return sentiment, confidence, cleaned


# ============================================================
# Test
# ============================================================
if __name__ == '__main__':
    test_reviews = [
        "The movie was absolutely amazing. I loved every minute of it.",
        "Terrible film. Boring story and bad acting.",
        "It was okay, not great but not terrible.",
        "I did not enjoy this movie at all.",
        "The plot was not good, but the acting was great.",
        "One of the best movies I have ever seen!",
        "Waste of time. I want my money back."
    ]

    print("\n" + "=" * 70)
    print("Testing the Movie Review Analyzer")
    print("=" * 70)

    for review in test_reviews:
        sentiment, confidence, cleaned = predict_sentiment(review)
        print(f"\nReview:      {review}")
        print(f"Cleaned:     {cleaned}")
        print(f"Prediction:  {sentiment}")
        print(f"Confidence:  {confidence:.2f}%")
        print("-" * 70)