
import pandas as pd
import nltk
import re
import string

from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# ============================================================
# Auto-download all required NLTK data
# ============================================================
def ensure_nltk_data():
    """Ensure all NLTK resources are available."""
    resources = {
        'tokenizers': ['punkt', 'punkt_tab'],
        'corpora': ['stopwords', 'wordnet', 'omw-1.4'],
        'taggers': [
            'averaged_perceptron_tagger',
            'averaged_perceptron_tagger_eng'
        ],
    }
    for category, names in resources.items():
        for name in names:
            try:
                nltk.data.find(f'{category}/{name}')
            except LookupError:
                print(f"Downloading NLTK resource: {name}")
                nltk.download(name, quiet=True)

ensure_nltk_data()
# 1. Load dataset
df = pd.read_csv("./dataset/IMDB Dataset.csv")


# 2. Check data
def check_data(df):
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts(dropna=False))

    print("\nReview length:")
    print(df["review"].dropna().str.len().describe())

check_data(df)


# 3. Cleaning
df_clean = df.dropna(
    subset=["review", "sentiment"]
).copy()

df_clean = df_clean.drop_duplicates()


def remove_html(text):
    return BeautifulSoup(
        str(text), "html.parser"
    ).get_text(" ")


df_clean["review"] = df_clean["review"].apply(remove_html)


# Remove reviews that are empty after HTML removal
df_clean["review"] = df_clean["review"].str.strip()

df_clean = df_clean[
    df_clean["review"].ne("")
].copy()


# 4. Initialize preprocessing tools once
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


# 5. Normalize contractions before punctuation removal
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


# 6. Tokenization, POS tagging, stopword removal,
#    and lemmatization
def preprocess_tokens(text):
    text = normalize_contractions(text)
    text = remove_punctuation(text)

    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    cleaned_tokens = []

    for word, tag in tagged_tokens:
        if word in stop_words:
            continue

        lemma = lemmatizer.lemmatize(
            word, get_wordnet_pos(tag)
        )

        cleaned_tokens.append(lemma)

    return cleaned_tokens

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
            # نلزق كلمة النفي بالكلمة اللي بعدها
            combined = f"{word}_{tokens[i + 1]}"
            result.append(combined)
            i += 2
        else:
            result.append(word)
            i += 1

    return " ".join(result)

def preprocess_text(text):
    tokens = preprocess_tokens(text)
    text = " ".join(tokens)
    text = handle_negation(text)   # ← ضيفي السطر ده
    return text

# 7. Apply preprocessing
df_clean["review"] = df_clean["review"].apply(
    preprocess_text
)


# 8. Remove empty processed reviews
df_clean["review"] = df_clean["review"].str.strip()

df_clean = df_clean[
    df_clean["review"].ne("")
].copy()

# Remove duplicates created by text normalization
df_clean = df_clean.drop_duplicates()


# 9. Final checks
check_data(df_clean)

print("\nEmpty reviews:", df_clean["review"].eq("").sum())

# 10. Save processed dataset
df_clean.to_csv(
    "dataset/movies_preprocessed.csv",
    index=False,
    encoding="utf-8"
)

print("\nPreprocessing completed!")
