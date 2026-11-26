import os
import re
import string
import joblib
import numpy as np
import nltk
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import FastText

# ============================================================
# 1. NLTK data
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
# 3. Load models (FastText + SVM)
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

model_path = os.path.join(project_root, 'model', 'sentiment_model.pkl')
fasttext_path = os.path.join(project_root, 'model', 'fasttext_model.model')

print("Loading FastText and SVM model...")
model = joblib.load(model_path)
fasttext_model = FastText.load(fasttext_path)
print("Models loaded successfully!")

# ============================================================
# 4. Text cleaning — IDENTICAL to preprocessing.py
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
# 5. Vector conversion for FastText
# ============================================================
def get_sentence_vector(text):
    tokens = text.split()
    vectors = [fasttext_model.wv[word] for word in tokens if word in fasttext_model.wv]
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

# ============================================================
# 6. Routes
# ============================================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'API is running!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        review = str(data.get('review') or '').strip()
        if not review:
            return jsonify({'error': 'Field "review" is required'}), 400

        cleaned = clean_text(review)
        if not cleaned:
            return jsonify({'error': 'Review is empty after cleaning.'}), 400

        # =======================================================
        # 🛑 Rule-Based Override: التعامل مع نفي النفي برمجياً
        # =======================================================
        # لو الجملة فيها "not_bad" نعتبرها إيجابية
        if "not_bad" in cleaned:
            return jsonify({
                'sentiment': 'Positive',
                'confidence': 95.0,
                'cleaned_text': cleaned
            }), 200
            
        # لو فيها "not_love" أو "not_like" نعتبرها سلبية
        if "not_love" in cleaned or "not_like" in cleaned:
            return jsonify({
                'sentiment': 'Negative',
                'confidence': 98.0,
                'cleaned_text': cleaned
            }), 200
        # =======================================================

        # لو مفيش القواعد دي، نكمل بالموديل العادي
        vectorized = get_sentence_vector(cleaned).reshape(1, -1)
        prediction = int(model.predict(vectorized)[0])
        # ... باقي الكود زي ما هو

        # التحويل باستخدام FastText
        vectorized = get_sentence_vector(cleaned).reshape(1, -1)
        
        prediction = int(model.predict(vectorized)[0])
        probabilities = model.predict_proba(vectorized)[0]
        confidence = float(max(probabilities) * 100)

        sentiment = 'Positive' if prediction == 1 else 'Negative'

        return jsonify({
            'sentiment': sentiment,
            'confidence': round(confidence, 2),
            'cleaned_text': cleaned
        }), 200

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

# ============================================================
# 7. Run
# ============================================================
if __name__ == '__main__':
    print("\nMovie Review Analyzer API")
    app.run(debug=True, host='0.0.0.0', port=5000)