"""
ELMo + Logistic Regression experiment for Movie-Review-Analyzer.

Run from project root while .venv_elmo is activated:
    python training/train_elmo.py

Uses the existing train/validation/test split created by prepare_data.py.
Does NOT overwrite the production SVM model.
"""
from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "experiments"
CACHE_DIR = DATA_DIR / "elmo_embeddings"
MODEL_DIR = ROOT / "model" / "comparison"
RESULTS_FILE = ROOT / "results" / "comparison_results.json"

ELMO_URL = "https://tfhub.dev/google/elmo/3"
BATCH_SIZE = 16
RANDOM_STATE = 42
C_VALUES = [0.1, 1.0, 10.0]

def find_column(df, candidates, description):
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find {description} column. Available columns: {list(df.columns)}"
    )

def load_split(name):
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python training\\prepare_data.py"
        )
    df = pd.read_csv(path)
    text_col = find_column(df, ["raw_review", "review", "clean_review"], "review text")
    label_col = find_column(df, ["label", "sentiment"], "sentiment label")
    texts = df[text_col].fillna("").astype(str).tolist()
    labels = df[label_col].tolist()
    return texts, labels

def get_elmo_encoder():
    print(f"Loading pretrained ELMo from {ELMO_URL} ...")
    module = hub.load(ELMO_URL)
    # The default signature expects the argument named "text".
    return module.signatures["default"]

def encode_texts(encoder, texts, cache_path, split_name):
    if cache_path.exists():
        print(f"Loading cached {split_name} embeddings: {cache_path.name}")
        return np.load(cache_path, mmap_mode=None)

    print(f"Extracting {split_name} ELMo embeddings ({len(texts):,} reviews)...")
    all_vectors = []
    start = time.perf_counter()

    for start_idx in range(0, len(texts), BATCH_SIZE):
        batch = texts[start_idx:start_idx + BATCH_SIZE]
        tensor = tf.constant(batch, dtype=tf.string)
        output = encoder(text=tensor)
        # Pooled sentence representation: one 1024-dimensional vector per review.
        vectors = output["default"].numpy().astype(np.float32)
        all_vectors.append(vectors)

        done = min(start_idx + len(batch), len(texts))
        if done % 1024 < BATCH_SIZE or done == len(texts):
            print(f"  {done:,}/{len(texts):,} completed")

    matrix = np.concatenate(all_vectors, axis=0)
    np.save(cache_path, matrix)
    elapsed = time.perf_counter() - start
    print(f"Saved {split_name} embeddings: {matrix.shape} ({elapsed:.1f}s)")
    return matrix

def to_binary_labels(y_train, y_val, y_test):
    # Handles either text labels (positive/negative) or numeric 0/1 labels.
    encoder = LabelEncoder()
    train = encoder.fit_transform(np.asarray(y_train).astype(str))
    val = encoder.transform(np.asarray(y_val).astype(str))
    test = encoder.transform(np.asarray(y_test).astype(str))
    if len(encoder.classes_) != 2:
        raise ValueError(f"Expected binary sentiment labels, got {encoder.classes_}")
    print("Label mapping:", {str(c): int(i) for i, c in enumerate(encoder.classes_)})
    return train, val, test, encoder

def save_metrics(metrics):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if RESULTS_FILE.exists():
        try:
            data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}

    if not isinstance(data, dict):
        data = {"previous_results": data}

    data["ELMo"] = metrics
    RESULTS_FILE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_texts, y_train_raw = load_split("train")
    val_texts, y_val_raw = load_split("validation")
    test_texts, y_test_raw = load_split("test")

    y_train, y_val, y_test, label_encoder = to_binary_labels(
        y_train_raw, y_val_raw, y_test_raw
    )

    encoder = get_elmo_encoder()
    total_train_start = time.perf_counter()

    X_train = encode_texts(
        encoder, train_texts, CACHE_DIR / "train_elmo.npy", "train"
    )
    X_val = encode_texts(
        encoder, val_texts, CACHE_DIR / "validation_elmo.npy", "validation"
    )

    best_model = None
    best_c = None
    best_val_f1 = -1.0

    print("\nSelecting Logistic Regression C using validation F1...")
    for c in C_VALUES:
        clf = LogisticRegression(
            C=c, max_iter=1000, solver="liblinear",
            random_state=RANDOM_STATE
        )
        clf.fit(X_train, y_train)
        val_pred = clf.predict(X_val)
        score = f1_score(y_val, val_pred, zero_division=0)
        print(f"  C={c:<5} validation F1={score:.5f}")
        if score > best_val_f1:
            best_val_f1 = score
            best_model = clf
            best_c = c

    # Refit the selected configuration on train + validation for final testing.
    X_train_val = np.concatenate([X_train, X_val], axis=0)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    final_model = LogisticRegression(
        C=best_c, max_iter=1000, solver="liblinear",
        random_state=RANDOM_STATE
    )
    final_model.fit(X_train_val, y_train_val)

    training_time = time.perf_counter() - total_train_start

    # Test embeddings are computed once and cached. This time is included in
    # end-to-end inference time to account for the ELMo feature extraction.
    inference_start = time.perf_counter()
    X_test = encode_texts(
        encoder, test_texts, CACHE_DIR / "test_elmo.npy", "test"
    )
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]
    inference_time = time.perf_counter() - inference_start

    metrics = {
        "model": "ELMo + Logistic Regression",
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "training_time_seconds": round(float(training_time), 3),
        "inference_time_seconds": round(float(inference_time), 3),
        "inference_time_per_review_seconds": round(
            float(inference_time / len(test_texts)), 8
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "class_order": [str(x) for x in label_encoder.classes_],
        "train_samples": len(train_texts),
        "validation_samples": len(val_texts),
        "test_samples": len(test_texts),
        "embedding": "ELMo pooled sentence vector (1024 dimensions)",
        "classifier": "LogisticRegression(liblinear)",
        "selected_C": float(best_c),
        "validation_f1_for_selected_C": float(best_val_f1),
        "batch_size": BATCH_SIZE,
        "elmo_url": ELMO_URL,
        "note": "Training time includes ELMo embedding extraction for train/validation and classifier fitting. Inference time includes ELMo extraction and prediction for the test set."
    }

    joblib.dump(final_model, MODEL_DIR / "elmo_classifier.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "elmo_label_encoder.joblib")
    (MODEL_DIR / "elmo_config.json").write_text(
        json.dumps({"elmo_url": ELMO_URL, "embedding_dim": 1024,
                    "selected_C": best_c, "batch_size": BATCH_SIZE},
                   indent=2),
        encoding="utf-8"
    )
    save_metrics(metrics)

    print("\n========== ELMo TEST RESULTS ==========")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key}: {metrics[key]:.4%}")
    print(f"Training time: {metrics['training_time_seconds']:.2f} seconds")
    print(f"End-to-end inference time: {metrics['inference_time_seconds']:.2f} seconds")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Selected C: {best_c}")
    print("\nSaved classifier: model/comparison/elmo_classifier.joblib")
    print("Updated results: results/comparison_results.json")
    print("Original production model files were not changed.")

if __name__ == "__main__":
    main()
