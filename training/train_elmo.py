
"""
ELMo + Logistic Regression experiment for Movie-Review-Analyzer.

Run from project root while .venv_elmo is activated:
    python training/train_elmo.py

Uses the existing train/validation/test split.
Does NOT overwrite the production model.
"""

from pathlib import Path
import hashlib
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import LabelEncoder


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "dataset" / "experiments"
CACHE_DIR = DATA_DIR / "elmo_embeddings_clean_v1"

MODEL_DIR = ROOT / "model" / "comparison"
RESULTS_FILE = ROOT / "results" / "comparison_results.json"

ELMO_URL = "https://tfhub.dev/google/elmo/3"

BATCH_SIZE = 16
EMBEDDING_DIM = 1024

RANDOM_STATE = 42
C_VALUES = [0.1, 1.0, 10.0]


# ============================================================
# 2. DATA LOADING
# ============================================================

def find_column(df, candidates, description):

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        f"Could not find {description} column. "
        f"Available columns: {list(df.columns)}"
    )


def load_split(name):

    path = DATA_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. "
            "Run python training\\prepare_data.py"
        )

    df = pd.read_csv(path)

    text_col = find_column(
        df,
        ["clean_review"],
        "clean review text"
    )

    label_col = find_column(
        df,
        ["label"],
        "sentiment label"
    )

    texts = (
        df[text_col]
        .fillna("")
        .astype(str)
        .tolist()
    )

    labels = df[label_col].tolist()

    return texts, labels


# ============================================================
# 3. ELMO ENCODER
# ============================================================

def get_elmo_encoder():

    print(f"Loading pretrained ELMo from {ELMO_URL} ...")

    module = hub.load(ELMO_URL)

    return module.signatures["default"]


# ============================================================
# 4. CACHE VALIDATION
# ============================================================

def calculate_text_hash(texts):

    content = json.dumps(
        texts,
        ensure_ascii=False,
        separators=(",", ":")
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def encode_texts(
    encoder,
    texts,
    cache_path,
    split_name,
    use_cache=True
):

    expected_shape = (
        len(texts),
        EMBEDDING_DIM
    )

    metadata_path = cache_path.with_suffix(".meta.json")

    current_hash = calculate_text_hash(texts)

    # --------------------------------------------------------
    # Check existing cache
    # --------------------------------------------------------

    if use_cache and cache_path.exists() and metadata_path.exists():

        print(f"Checking cached {split_name} embeddings...")

        try:

            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )

            cache_is_compatible = (
                metadata.get("text_hash") == current_hash
                and metadata.get("shape") == list(expected_shape)
                and metadata.get("elmo_url") == ELMO_URL
                and metadata.get("embedding_dim") == EMBEDDING_DIM
            )

            if cache_is_compatible:

                cached = np.load(
                    cache_path,
                    mmap_mode="r"
                )

                if (
                    cached.shape == expected_shape
                    and np.isfinite(cached).all()
                ):

                    print(
                        f"Valid cache found: {cache_path.name}"
                    )

                    return cached

            print(
                f"Cache is outdated or invalid for {split_name}. "
                "Re-extracting embeddings..."
            )

        except (ValueError, OSError, json.JSONDecodeError) as error:

            print(
                f"Cache validation failed: {error}. "
                "Re-extracting embeddings..."
            )

    elif not use_cache:

        print(
            f"Cache bypassed for {split_name}: "
            "measuring fresh embedding extraction."
        )

    # --------------------------------------------------------
    # Extract embeddings
    # --------------------------------------------------------

    print(
        f"Extracting {split_name} ELMo embeddings "
        f"({len(texts):,} reviews)..."
    )

    start = time.perf_counter()

    matrix = np.empty(
        expected_shape,
        dtype=np.float32
    )

    for start_idx in range(0, len(texts), BATCH_SIZE):

        batch = texts[
            start_idx:start_idx + BATCH_SIZE
        ]

        tensor = tf.constant(
            batch,
            dtype=tf.string
        )

        output = encoder(text=tensor)

        vectors = (
            output["default"]
            .numpy()
            .astype(np.float32)
        )

        end_idx = start_idx + len(batch)

        matrix[start_idx:end_idx] = vectors

        done = end_idx

        if done % 1024 < BATCH_SIZE or done == len(texts):

            print(
                f"  {done:,}/{len(texts):,} completed"
            )
    matrix = np.concatenate(
        all_vectors,
        axis=0
    )

    # --------------------------------------------------------
    # Validate generated embeddings
    # --------------------------------------------------------

    if matrix.shape != expected_shape:

        raise ValueError(
            f"Unexpected ELMo shape for {split_name}: "
            f"{matrix.shape}, expected {expected_shape}"
        )

    if not np.isfinite(matrix).all():

        raise ValueError(
            f"ELMo generated invalid values for {split_name}"
        )

    # --------------------------------------------------------
    # Save cache and metadata
    # --------------------------------------------------------

    np.save(cache_path, matrix)

    metadata = {
        "split": split_name,
        "text_hash": current_hash,
        "shape": list(matrix.shape),
        "embedding_dim": EMBEDDING_DIM,
        "elmo_url": ELMO_URL
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8"
    )

    elapsed = time.perf_counter() - start

    print(
        f"Saved {split_name} embeddings: "
        f"{matrix.shape} ({elapsed:.1f}s)"
    )

    return matrix


# ============================================================
# 5. LABEL ENCODING
# ============================================================

def to_binary_labels(
    y_train,
    y_val,
    y_test
):

    encoder = LabelEncoder()

    train = encoder.fit_transform(
        np.asarray(y_train).astype(str)
    )

    val = encoder.transform(
        np.asarray(y_val).astype(str)
    )

    test = encoder.transform(
        np.asarray(y_test).astype(str)
    )

    if len(encoder.classes_) != 2:

        raise ValueError(
            f"Expected binary sentiment labels, "
            f"got {encoder.classes_}"
        )

    print(
        "Label mapping:",
        {
            str(c): int(i)
            for i, c in enumerate(encoder.classes_)
        }
    )

    return train, val, test, encoder


# ============================================================
# 6. SAVE RESULTS
# ============================================================


def save_metrics(metrics):

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if RESULTS_FILE.exists():
        try:
            with open(RESULTS_FILE, encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}
    else:
        existing = {}

    if isinstance(existing, list):
        existing = {"models": existing}

    if not isinstance(existing, dict):
        existing = {}

    # Preserve the existing protocol, models, SVM, and LSTM results.
    # Add or replace only the ELMo entry.
    existing["ELMo"] = metrics

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            existing,
            f,
            ensure_ascii=False,
            indent=2
        )

# ============================================================
# 7. MAIN EXPERIMENT
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load existing dataset splits
    # --------------------------------------------------------

    print("\nLoading existing dataset splits...")

    train_texts, y_train_raw = load_split("train")

    val_texts, y_val_raw = load_split("validation")

    test_texts, y_test_raw = load_split("test")

    print(f"Train      : {len(train_texts):,}")
    print(f"Validation : {len(val_texts):,}")
    print(f"Test       : {len(test_texts):,}")

    y_train, y_val, y_test, label_encoder = (
        to_binary_labels(
            y_train_raw,
            y_val_raw,
            y_test_raw
        )
    )

    # --------------------------------------------------------
    # Load pretrained ELMo
    # --------------------------------------------------------

    encoder = get_elmo_encoder()

    # --------------------------------------------------------
    # Extract train and validation embeddings
    # --------------------------------------------------------

    embedding_train_start = time.perf_counter()

    X_train = encode_texts(
        encoder,
        train_texts,
        CACHE_DIR / "train_elmo.npy",
        "train"
    )

    X_val = encode_texts(
        encoder,
        val_texts,
        CACHE_DIR / "validation_elmo.npy",
        "validation"
    )

    embedding_train_validation_time = (
        time.perf_counter() - embedding_train_start
    )

    # --------------------------------------------------------
    # Tune Logistic Regression using validation F1
    # --------------------------------------------------------

    print(
        "\nSelecting Logistic Regression C "
        "using validation F1..."
    )

    best_c = None
    best_val_f1 = -1.0

    tuning_start = time.perf_counter()

    for c in C_VALUES:

        clf = LogisticRegression(
            C=c,
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_STATE
        )

        clf.fit(
            X_train,
            y_train
        )

        val_pred = clf.predict(X_val)

        score = f1_score(
            y_val,
            val_pred,
            zero_division=0
        )

        print(
            f"  C={c:<5} validation F1={score:.5f}"
        )

        if score > best_val_f1:

            best_val_f1 = score
            best_c = c

    tuning_time = (
        time.perf_counter() - tuning_start
    )

    print(f"\nBest C: {best_c}")
    print(f"Best validation F1: {best_val_f1:.5f}")

    # --------------------------------------------------------
    # Final fit using train + validation
    # --------------------------------------------------------

    print(
        "\nRefitting Logistic Regression "
        "on train + validation..."
    )

    X_train_val = np.concatenate(
        [X_train, X_val],
        axis=0
    )

    y_train_val = np.concatenate(
        [y_train, y_val],
        axis=0
    )

    final_model = LogisticRegression(
        C=best_c,
        max_iter=1000,
        solver="liblinear",
        random_state=RANDOM_STATE
    )

    final_fit_start = time.perf_counter()

    final_model.fit(
        X_train_val,
        y_train_val
    )

    final_fit_time = (
        time.perf_counter() - final_fit_start
    )

    training_time = (
        embedding_train_validation_time
        + tuning_time
        + final_fit_time
    )

    # --------------------------------------------------------
    # Test: fresh ELMo extraction for fair inference timing
    # --------------------------------------------------------

    print("\nExtracting test embeddings...")

    embedding_start = time.perf_counter()

    X_test = encode_texts(
        encoder,
        test_texts,
        CACHE_DIR / "test_elmo.npy",
        "test",
        use_cache=False
    )

    embedding_extraction_time = (
        time.perf_counter() - embedding_start
    )

    # --------------------------------------------------------
    # Classifier-only prediction timing
    # --------------------------------------------------------

    prediction_start = time.perf_counter()

    y_pred = final_model.predict(X_test)

    y_prob = final_model.predict_proba(X_test)[:, 1]

    classifier_prediction_time = (
        time.perf_counter() - prediction_start
    )

    # --------------------------------------------------------
    # End-to-end inference time
    # --------------------------------------------------------

    inference_time = (
        embedding_extraction_time
        + classifier_prediction_time
    )

    # --------------------------------------------------------
    # Final evaluation on test set
    # --------------------------------------------------------

    test_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    test_precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    test_recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    test_f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    test_roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    test_confusion_matrix = confusion_matrix(
        y_test,
        y_pred
    )

    test_classification_report = classification_report(
        y_test,
        y_pred,
        target_names=[
            str(x) for x in label_encoder.classes_
        ],
        output_dict=True,
        zero_division=0
    )

    # --------------------------------------------------------
    # Standardized metrics structure
    # --------------------------------------------------------


    metrics = {

    "model": "ELMo + Logistic Regression",

    "accuracy": float(test_accuracy),
    "precision": float(test_precision),
    "recall": float(test_recall),
    "f1": float(test_f1),
    "roc_auc": float(test_roc_auc),

    "training_seconds": round(
        float(training_time),
        3
    ),

    "inference_seconds_total": round(
        float(inference_time),
        3
    ),

    "test_samples": int(len(y_test)),

    "confusion_matrix": (
        test_confusion_matrix.tolist()
    ),

    "class_order": [
        str(label_encoder.classes_[0]),
        str(label_encoder.classes_[1])
    ],

    "split": {
        "train": len(train_texts),
        "validation": len(val_texts),
        "test": len(test_texts)
    },

    "training_config": {

        "embedding": "ELMo",
        "embedding_dim": EMBEDDING_DIM,
        "elmo_url": ELMO_URL,

        "classifier": "LogisticRegression",
        "selected_C": float(best_c),
        "C_values": C_VALUES,

        "batch_size": BATCH_SIZE,
        "random_state": RANDOM_STATE,

        "validation_best_f1": float(best_val_f1),

        "embedding_extraction_train_validation_seconds": round(
            float(embedding_train_validation_time),
            3
        ),

        "classifier_tuning_seconds": round(
            float(tuning_time),
            3
        ),

        "classifier_final_fit_seconds": round(
            float(final_fit_time),
            3
        ),

        "test_embedding_extraction_seconds": round(
            float(embedding_extraction_time),
            3
        ),

        "classifier_prediction_seconds": round(
            float(classifier_prediction_time),
            6
        ),

        "text_column": "clean_review",
        "classifier_training_data": "train_plus_validation"
    }
}

    # --------------------------------------------------------
    # Save model artifacts
    # --------------------------------------------------------

    joblib.dump(
        final_model,
        MODEL_DIR / "elmo_classifier.joblib"
    )

    joblib.dump(
        label_encoder,
        MODEL_DIR / "elmo_label_encoder.joblib"
    )

    config = {
        "model_name": "ELMo + Logistic Regression",
        "elmo_url": ELMO_URL,
        "embedding_dim": EMBEDDING_DIM,
        "selected_C": float(best_c),
        "batch_size": BATCH_SIZE,
        "random_state": RANDOM_STATE,
        "text_column": "clean_review",
        "classifier_training_data": "train_plus_validation"
    }

    (MODEL_DIR / "elmo_config.json").write_text(
        json.dumps(
            config,
            indent=2
        ),
        encoding="utf-8"
    )

    save_metrics(metrics)

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print("\n========== ELMo TEST RESULTS ==========")

    print(f"Accuracy  : {test_accuracy:.4%}")
    print(f"Precision : {test_precision:.4%}")
    print(f"Recall    : {test_recall:.4%}")
    print(f"F1-score  : {test_f1:.4%}")
    print(f"ROC-AUC   : {test_roc_auc:.4%}")

    print("\nConfusion Matrix:")
    print(test_confusion_matrix)

    print("\n========== TIMING ==========")

    print(
        "Train/Validation embedding extraction: "
        f"{embedding_train_validation_time:.2f} seconds"
    )

    print(
        f"Classifier tuning: {tuning_time:.2f} seconds"
    )

    print(
        f"Final classifier fit: {final_fit_time:.2f} seconds"
    )

    print(
        f"Total training pipeline: {training_time:.2f} seconds"
    )

    print(
        "Test embedding extraction: "
        f"{embedding_extraction_time:.2f} seconds"
    )

    print(
        "Classifier prediction: "
        f"{classifier_prediction_time:.6f} seconds"
    )

    print(
        "End-to-end inference: "
        f"{inference_time:.2f} seconds"
    )

    print(f"\nSelected C: {best_c}")

    print("\nSaved classifier:")
    print("model/comparison/elmo_classifier.joblib")

    print("\nSaved label encoder:")
    print("model/comparison/elmo_label_encoder.joblib")

    print("\nSaved configuration:")
    print("model/comparison/elmo_config.json")

    print("\nUpdated results:")
    print("results/comparison_results.json")

    print(
        "\nOriginal production model files were not changed."
    )


if __name__ == "__main__":
    main()
