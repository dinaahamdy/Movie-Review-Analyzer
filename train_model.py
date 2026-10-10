import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)


# ============================================================
# 1. Paths
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

# اسم الملف زي ما الشخص الأول حفظه
dataset_path = os.path.join(script_dir, 'dataset', 'movies_preprocessed.csv')
model_dir = os.path.join(script_dir, 'model')
os.makedirs(model_dir, exist_ok=True)


# ============================================================
# 2. Load data
# ============================================================
print("Loading dataset...")
df = pd.read_csv(dataset_path)
print(f"Total reviews: {len(df)}")

df = df.dropna(subset=['review', 'sentiment'])

# Encode
df['sentiment_encoded'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df = df.dropna(subset=['sentiment_encoded'])
df['sentiment_encoded'] = df['sentiment_encoded'].astype(int)

print(f"Sentiment distribution:\n{df['sentiment_encoded'].value_counts()}")


# ============================================================
# 3. Features & Labels
# ============================================================
X = df['review']                    # ← اسم العمود زي ما هو
y = df['sentiment_encoded']


# ============================================================
# 4. Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")


# ============================================================
# 5. TF-IDF
# ============================================================
print("\nApplying TF-IDF...")
vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"Train matrix: {X_train_tfidf.shape}")
print(f"Test matrix:  {X_test_tfidf.shape}")


# ============================================================
# 6. Train
# ============================================================
print("\nTraining Logistic Regression...")
model = LogisticRegression(
    max_iter=1000, C=1.0,
    solver='liblinear', random_state=42
)
model.fit(X_train_tfidf, y_train)
print("Done!")


# ============================================================
# 7. Evaluate
# ============================================================
y_pred = model.predict(X_test_tfidf)

print(f"\n========== Evaluation ==========")
print(f"Accuracy :  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:  {precision_score(y_test, y_pred):.4f}")
print(f"Recall   :  {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score :  {f1_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))


# ============================================================
# 8. Save
# ============================================================
joblib.dump(model, os.path.join(model_dir, 'sentiment_model.pkl'))
joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))

print("\n✅ Model and vectorizer saved!")
print(f"Vectorizer features: {len(vectorizer.get_feature_names_out())}")
print(f"Model expects:       {model.n_features_in_}")