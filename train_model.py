import pandas as pd
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from gensim.models import FastText

# ============================================================
# 1. Paths
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
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

# إضافة أمثلة متنوعة لتحسين النفي
negative_examples = [
    'not_like movie', 'not_like act', 'not_like plot', 'not_like film',
    'not_like end', 'not_like stori', 'not_like direct', 'not_like charact',
    'not_like scene', 'not_like dialog', 'not_like music', 'not_like pace',
    'not_like humor', 'not_like twist', 'not_like cinematographi', 'not_like perform',
    'not_like script', 'not_like cast', 'not_like visual', 'not_like effect',
    'not_like soundtrack', 'not_like screenwrit', 'not_like product', 'not_like edit',
    'not_like pac',
    'not_favorite movie', 'not_favorite film', 'not_favorite at all'
]

repeated_negatives = negative_examples * 50

synthetic_data = pd.DataFrame({
    'review': repeated_negatives,
    'sentiment': ['negative'] * len(repeated_negatives)
})
print(f"Synthetic reviews: {len(synthetic_data)}")
df = pd.concat([df, synthetic_data], ignore_index=True)
print(f"After adding synthetic data: {len(df)}")

# Encode
df['sentiment_encoded'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df = df.dropna(subset=['sentiment_encoded'])
df['sentiment_encoded'] = df['sentiment_encoded'].astype(int)
print(f"Sentiment distribution:\n{df['sentiment_encoded'].value_counts()}")

# ============================================================
# 3. Features & Labels
# ============================================================
X = df['review']
y = df['sentiment_encoded']

# ============================================================
# 4. Split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# ============================================================
# 5. FastText (Word Embeddings) - feature extractor ثابت
# ============================================================
print("\nTraining FastText Model... (This might take a minute)")
X_train_tokens = [text.split() for text in X_train]

fasttext_model = FastText(
    sentences=X_train_tokens,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    sg=1
)
print("FastText training done!")

def get_sentence_vector(text):
    tokens = text.split()
    vectors = [fasttext_model.wv[word] for word in tokens if word in fasttext_model.wv]
    if len(vectors) == 0:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

print("Converting text to vectors...")
X_train_vectors = np.array([get_sentence_vector(text) for text in X_train])
X_test_vectors = np.array([get_sentence_vector(text) for text in X_test])

# ============================================================
# 6. مقارنة 4 موديلات
# ============================================================
models_to_compare = {
    "LinearSVC": CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=5000, random_state=42), cv=3
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=2000, random_state=42
    ),
    "GaussianNB": GaussianNB(),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
}

results = {}

for name, clf in models_to_compare.items():
    print(f"\n{'='*50}")
    print(f"=== تدريب {name} ===")
    print(f"{'='*50}")
    clf.fit(X_train_vectors, y_train)
    y_pred = clf.predict(X_test_vectors)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        "model": clf,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

    print(f"Accuracy :  {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall   :  {rec:.4f}")
    print(f"F1-Score :  {f1:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

# ============================================================
# 7. جدول المقارنة النهائي
# ============================================================
print("\n" + "=" * 70)
print("جدول المقارنة بين الموديلات الأربعة")
print("=" * 70)
print(f"{'Model':<22} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<22} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
          f"{r['recall']:<12.4f} {r['f1']:<12.4f}")
print("=" * 70)

# ============================================================
# 8. اختيار الأحسن بناءً على F1
# ============================================================
best_name = max(results, key=lambda k: results[k]["f1"])
best_model = results[best_name]["model"]
print(f"\n✅ أحسن موديل: {best_name}")
print(f"   Accuracy: {results[best_name]['accuracy']:.4f}")
print(f"   F1-Score: {results[best_name]['f1']:.4f}")

# ============================================================
# 9. Save Models
# ============================================================
joblib.dump(best_model, os.path.join(model_dir, 'sentiment_model.pkl'))
fasttext_model.save(os.path.join(model_dir, 'fasttext_model.model'))

# حفظ اسم الموديل الأحسن في ملف صغير (اختياري للتوثيق)
with open(os.path.join(model_dir, 'best_model_name.txt'), 'w', encoding='utf-8') as f:
    f.write(best_name)

print("\n✅ Models saved successfully!")
print(f"   - sentiment_model.pkl  ({best_name})")
print(f"   - fasttext_model.model")