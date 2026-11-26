"""
Train a fresh SVM comparison baseline on the shared split.
This does NOT overwrite model/sentiment_model.pkl or model/tfidf_vectorizer.pkl.

Run from project root:
    python training/train_svm.py
"""
from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "experiments"
MODEL_DIR = ROOT / "model" / "comparison"
RESULTS_DIR = ROOT / "results"


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    x_train = train["clean_review"].fillna("").astype(str)
    y_train = train["label"].astype(int)
    x_test = test["clean_review"].fillna("").astype(str)
    y_test = test["label"].astype(int)

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    print("Fitting TF-IDF on training data only...")
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    print("Training calibrated LinearSVC...")
    start = time.perf_counter()
    base = LinearSVC(C=1.0, max_iter=5000, random_state=42)
    model = CalibratedClassifierCV(base, cv=3)
    model.fit(x_train_vec, y_train)
    training_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(x_test_vec)
    probabilities = model.predict_proba(x_test_vec)[:, 1]
    inference_seconds = time.perf_counter() - start

    metrics = {
        "model": "SVM (TF-IDF + calibrated LinearSVC)",
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "training_seconds": round(training_seconds, 3),
        "inference_seconds_total": round(inference_seconds, 3),
        "test_samples": int(len(y_test)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "class_order": ["Negative", "Positive"],
        "split": {"train": int(len(train)), "test": int(len(test))},
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "svm_model.joblib")
    joblib.dump(vectorizer, MODEL_DIR / "svm_tfidf_vectorizer.joblib")

    results_path = RESULTS_DIR / "comparison_results.json"
    existing = {}
    if results_path.exists():
        try:
            existing = json.loads(results_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("Warning: existing comparison_results.json is invalid; creating a new results file.")
    existing.setdefault("protocol", {
        "dataset": "IMDB Dataset.csv",
        "split": "70% train / 10% validation / 20% test",
        "random_state": 42,
        "test_set_shared": True,
    })
    existing.setdefault("models", {})
    existing["models"]["svm"] = metrics
    results_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    print("\nSVM comparison baseline:")
    print(classification_report(y_test, predictions, target_names=["Negative", "Positive"]))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model artifacts in: {MODEL_DIR}")
    print(f"Saved metrics in: {results_path}")
    print("\nThe existing production model files were not changed.")


if __name__ == "__main__":
    main()
