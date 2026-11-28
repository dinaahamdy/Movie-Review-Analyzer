import os
import re
import string
import joblib
import numpy as np
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from gensim.models import FastText

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
# Initialize tools
# ============================================================
stop_words = set(stopwords.words("english"))
stop_words -= {"no", "nor", "not", "never", "neither"}
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    elif tag.startswith("V"): return wordnet.VERB
    elif tag.startswith("R"): return wordnet.ADV
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
    return text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

def handle_negation(text):
    negation_words = {
        "not", "no", "never", "neither", "nor",
        "cannot", "cant", "wont", "dont", "didnt",
        "isnt", "arent", "wasnt", "werent",
        "hasnt", "havent", "hadnt",
        "wouldnt", "shouldnt", "couldnt"
    }

    tokens = text.split()
    
    # بنحدد نوع كل كلمة (POS Tagging)
    tagged_tokens = pos_tag(tokens)
    
    result = []
    i = 0

    while i < len(tagged_tokens):
        word, tag = tagged_tokens[i]
        
        if word in negation_words and i + 1 < len(tagged_tokens):
            next_word, next_tag = tagged_tokens[i + 1]
            
            # لو الكلمة اللي بعد النفي هي صفة (JJ) أو فعل (VB) أو ظرف (RB)
            # زي: not + favorite (JJ) أو not + love (VB)
            if next_tag.startswith('J') or next_tag.startswith('V') or next_tag.startswith('R'):
                # نلزقهم ببعض عشان الموديل يقراها ككلمة واحدة معناها سلبي
                result.append(f"not_{next_word}") 
            else:
                # لو اسم، بنسيب النفي لوحده
                result.append(word)
                result.append(next_word)
            i += 2
        else:
            result.append(word)
            i += 1

    return " ".join(result)

def clean_text(text):
    text = remove_html(text).strip()
    if not text: return ""
    text = normalize_contractions(text)
    text = remove_punctuation(text)
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    cleaned_tokens = []
    for word, tag in tagged_tokens:
        if word in stop_words: continue
        lemma = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        cleaned_tokens.append(lemma)
    text = " ".join(cleaned_tokens)
    return handle_negation(text)

# ============================================================
# Load models
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'model', 'sentiment_model.pkl')
fasttext_path = os.path.join(script_dir, 'model', 'fasttext_model.model')

model = joblib.load(model_path)
fasttext_model = FastText.load(fasttext_path)

# ============================================================
# Predict function
# ============================================================
def get_sentence_vector(text):
    tokens = text.split()
    vectors = [fasttext_model.wv[word] for word in tokens if word in fasttext_model.wv]
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

def predict_sentiment(review):
    cleaned = clean_text(review)
    if not cleaned:
        return "Unknown", 0.0, cleaned

    vectorized = get_sentence_vector(cleaned).reshape(1, -1)
    
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
        "I did not enjoy this movie at all.",
        "I don't like the plot, it was boring."
    ]
    print("\n" + "=" * 70)
    print("Testing Movie Review Analyzer (FastText + SVM)")
    print("=" * 70)

    for review in test_reviews:
        sentiment, confidence, cleaned = predict_sentiment(review)
        print(f"\nReview:      {review}")
        print(f"Cleaned:     {cleaned}")
        print(f"Prediction:  {sentiment}")
        print(f"Confidence:  {confidence:.2f}%")
        print("-" * 70)