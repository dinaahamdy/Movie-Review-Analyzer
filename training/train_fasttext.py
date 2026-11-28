
"""
FastText + Logistic Regression experiment for Movie-Review-Analyzer.

Run from project root:
    python training/train_fasttext.py

Uses the existing train/validation/test split.
Trains FastText on training data only.
Tunes Logistic Regression using validation F1.
Refits the classifier on train + validation.
Evaluates on test only once.
Does NOT overwrite the production model.
"""

from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from gensim.models import FastText

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. Paths and configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "dataset" / "experiments"
OUTPUT_DIR = PROJECT_DIR / "model" / "comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
VAL_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"

RANDOM_STATE = 42

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 2
EPOCHS = 10
WORKERS = 4

C_VALUES = [0.1, 1.0, 10.0]

# ============================================================
# 2. Load and validate the existing splits
# ============================================================

def load_split(path, split_name):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required_columns = {"clean_review", "label"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"{split_name} is missing columns: {missing}"
        )

    df = df.dropna(subset=["clean_review", "label"]).copy()

    df["clean_review"] = df["clean_review"].astype(str)
    df["label"] = df["label"].astype(int)

    df = df[df["clean_review"].str.strip() != ""]

    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError(f"{split_name} contains invalid labels.")

    if df.empty:
        raise ValueError(f"{split_name} is empty.")

    return (
        df["clean_review"].tolist(),
        df["label"].to_numpy(dtype=np.int32),
    )


print("Loading existing dataset splits...")

X_train, y_train = load_split(TRAIN_PATH, "Training split")
X_val, y_val = load_split(VAL_PATH, "Validation split")
X_test, y_test = load_split(TEST_PATH, "Test split")

print(f"Train      : {len(X_train)}")
print(f"Validation : {len(X_val)}")
print(f"Test       : {len(X_test)}")

# ============================================================
# 3. Train FastText on training data only
# ============================================================

print("\nTraining FastText embeddings...")

train_tokens = [text.split() for text in X_train]

embedding_start = time.perf_counter()

fasttext_model = FastText(
    sentences=train_tokens,
    vector_size=VECTOR_SIZE,
    window=WINDOW,
    min_count=MIN_COUNT,
    workers=WORKERS,
    sg=1,
    seed=RANDOM_STATE,
    epochs=EPOCHS,
    min_n=3,
    max_n=6,
)

embedding_training_seconds = (
    time.perf_counter() - embedding_start
)

print("FastText training completed.")
print(
    f"Embedding training time: "
    f"{embedding_training_seconds:.2f} seconds"
)

# ============================================================
# 4. Convert reviews into sentence embeddings
# ============================================================

def get_sentence_vector(text):
    tokens = text.split()

    if not tokens:
        return np.zeros(VECTOR_SIZE, dtype=np.float32)

    vectors = [
        fasttext_model.wv[token]
        for token in tokens
    ]

    if not vectors:
        return np.zeros(VECTOR_SIZE, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


def transform_texts(texts):
    return np.vstack([
        get_sentence_vector(text)
        for text in texts
    ])


print("\nConverting reviews into FastText vectors...")

vectorization_start = time.perf_counter()

X_train_vectors = transform_texts(X_train)
X_val_vectors = transform_texts(X_val)
X_test_vectors = transform_texts(X_test)

vectorization_seconds = (
    time.perf_counter() - vectorization_start
)

print("Vectorization completed.")
print(f"Vector shape: {X_train_vectors.shape}")

# ============================================================
# 5. Tune Logistic Regression using validation data
# ============================================================

print("\nTuning Logistic Regression...")

best_c = None
best_val_f1 = -1.0
best_val_model = None

tuning_start = time.perf_counter()

for c_value in C_VALUES:
    clf = LogisticRegression(
        C=c_value,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    clf.fit(X_train_vectors, y_train)

    val_predictions = clf.predict(X_val_vectors)

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    print(f"C={c_value:<5} | Validation F1={val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_c = c_value
        best_val_model = clf

tuning_seconds = time.perf_counter() - tuning_start

print(f"\nBest C: {best_c}")
print(f"Best validation F1: {best_val_f1:.4f}")

# ============================================================
# 6. Refit classifier on train + validation
# ============================================================

print("\nRefitting Logistic Regression on train + validation...")

X_train_val_vectors = np.vstack([
    X_train_vectors,
    X_val_vectors,
])

y_train_val = np.concatenate([
    y_train,
    y_val,
])

final_model = LogisticRegression(
    C=best_c,
    max_iter=2000,
    random_state=RANDOM_STATE,
)

final_fit_start = time.perf_counter()

final_model.fit(X_train_val_vectors, y_train_val)

final_fit_seconds = (
    time.perf_counter() - final_fit_start
)

# ============================================================
# 7. Final evaluation on test data
# ============================================================

print("\nEvaluating on the test set...")

inference_start = time.perf_counter()

test_predictions = final_model.predict(X_test_vectors)
test_probabilities = final_model.predict_proba(X_test_vectors)[:, 1]

classifier_prediction_seconds = (
    time.perf_counter() - inference_start
)

test_metrics = {
    "accuracy": float(
        accuracy_score(y_test, test_predictions)
    ),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "f1": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),
    "confusion_matrix": confusion_matrix(
        y_test,
        test_predictions,
    ).tolist(),
    "classification_report": classification_report(
        y_test,
        test_predictions,
        target_names=["Negative", "Positive"],
        output_dict=True,
        zero_division=0,
    ),
}

print("\n========== TEST RESULTS ==========")

for metric in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]:
    print(f"{metric.upper():<12}: {test_metrics[metric]:.4f}")

print("\nConfusion Matrix:")
print(np.array(test_metrics["confusion_matrix"]))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Negative", "Positive"],
        zero_division=0,
    )
)

# ============================================================
# 8. Save model artifacts
# ============================================================

print("\nSaving FastText experiment artifacts...")

fasttext_path = OUTPUT_DIR / "fasttext_embeddings.model"
classifier_path = OUTPUT_DIR / "fasttext_classifier.joblib"
config_path = OUTPUT_DIR / "fasttext_config.json"
metrics_path = OUTPUT_DIR / "fasttext_metrics.json"

fasttext_model.save(str(fasttext_path))
joblib.dump(final_model, classifier_path)

config = {
    "model_name": "FastText + Logistic Regression",
    "text_column": "clean_review",
    "label_column": "label",
    "vector_size": VECTOR_SIZE,
    "window": WINDOW,
    "min_count": MIN_COUNT,
    "epochs": EPOCHS,
    "sg": 1,
    "min_n": 3,
    "max_n": 6,
    "best_C": best_c,
    "random_state": RANDOM_STATE,
    "embedding_training_data": "train_only",
    "classifier_training_data": "train_plus_validation",
}

with open(config_path, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=4, ensure_ascii=False)

metrics = {
    "model_name": "FastText + Logistic Regression",
    "validation": {
        "best_f1": float(best_val_f1),
        "best_C": float(best_c),
    },
    "test": test_metrics,
    "timing": {
        "embedding_training_seconds": float(
            embedding_training_seconds
        ),
        "vectorization_seconds": float(
            vectorization_seconds
        ),
        "classifier_tuning_seconds": float(
            tuning_seconds
        ),
        "classifier_final_fit_seconds": float(
            final_fit_seconds
        ),
        "classifier_prediction_seconds": float(
            classifier_prediction_seconds
        ),
    },
    "dataset_sizes": {
        "train": len(X_train),
        "validation": len(X_val),
        "test": len(X_test),
    },
}

with open(metrics_path, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4, ensure_ascii=False)

print("\nAll FastText artifacts saved successfully!")
print(f"- {fasttext_path}")
print(f"- {classifier_path}")
print(f"- {config_path}")
print(f"- {metrics_path}")

print("\nFastText experiment completed.")
