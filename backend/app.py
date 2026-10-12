import os
import re
import string

import joblib
import nltk
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# ============================================================
# 1. NLTK data (auto-download if missing)
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
# 2. Flask app
# ============================================================
app = Flask(__name__)
CORS(app)


# ============================================================
# 3. Load model + vectorizer  (project_root/model/)
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

model_path = os.path.join(project_root, 'model', 'sentiment_model.pkl')
vectorizer_path = os.path.join(project_root, 'model', 'tfidf_vectorizer.pkl')

print("Loading model and vectorizer...")
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
print("Model loaded successfully!")


# ============================================================
# 4. Text cleaning — IDENTICAL to preprocessing.py / predict.py
#    (the model was trained on text cleaned this exact way)
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
        str.maketrans(string.punctuation, " " * len(string.punctuation))
    )


NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor",
    "cannot", "cant", "wont", "dont", "didnt",
    "isnt", "arent", "wasnt", "werent",
    "hasnt", "havent", "hadnt",
    "wouldnt", "shouldnt", "couldnt"
}


def handle_negation(text):
    """'not enjoy' -> 'not_enjoy'"""
    tokens = text.split()
    result = []
    i = 0
    while i < len(tokens):
        word = tokens[i]
        if word in NEGATION_WORDS and i + 1 < len(tokens):
            result.append(f"{word}_{tokens[i + 1]}")
            i += 2
        else:
            result.append(word)
            i += 1
    return " ".join(result)


def clean_text(text):
    text = remove_html(text).strip()
    if not text:
        return ""

    text = normalize_contractions(text)
    text = remove_punctuation(text)

    tagged_tokens = pos_tag(word_tokenize(text))

    cleaned_tokens = []
    for word, tag in tagged_tokens:
        if word in stop_words:
            continue
        cleaned_tokens.append(lemmatizer.lemmatize(word, get_wordnet_pos(tag)))

    return handle_negation(" ".join(cleaned_tokens))


# ============================================================
# 5. Routes
# ============================================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Movie Review Analyzer API is running!',
        'endpoints': {
            'POST /predict': 'Analyze a movie review',
            'GET /health': 'Check server status'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'vectorizer_loaded': vectorizer is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Request : { "review": "The movie was amazing!" }
    Response: { "sentiment": "Positive", "confidence": 94.5, "cleaned_text": "..." }
    """
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        review = str(data.get('review') or '').strip()
        if not review:
            return jsonify({'error': 'Field "review" is required and cannot be empty'}), 400

        cleaned = clean_text(review)
        if not cleaned:
            return jsonify({
                'error': 'Review is empty after cleaning. Please provide more meaningful text.'
            }), 400

        vectorized = vectorizer.transform([cleaned])
        prediction = int(model.predict(vectorized)[0])
        probabilities = model.predict_proba(vectorized)[0]
        confidence = float(max(probabilities) * 100)

        # Training labels: positive -> 1, negative -> 0
        sentiment = 'Positive' if prediction == 1 else 'Negative'

        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence, 2),
            'cleaned_text': cleaned
        }), 200

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


# ============================================================
# 6. Run
# ============================================================
if __name__ == '__main__':
    print("\nMovie Review Analyzer API")
    print("Server running at: http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
