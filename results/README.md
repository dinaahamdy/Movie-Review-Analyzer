# Model comparison output

The presentation page (`frontend/model-comparison.html`) requests `GET /comparison-results`.
The Flask endpoint reads `results/comparison_results.json`.

The training experiments should write a JSON object with this shape (use actual measured values; do not fill it with estimates):

```json
{
  "generated_at": "2026-11-26T12:00:00",
  "dataset": {
    "name": "IMDb Movie Reviews",
    "total_reviews": 50000,
    "train_samples": 40000,
    "test_samples": 10000,
    "split": "80/20 stratified"
  },
  "models": [
    {
      "name": "TF-IDF + Linear SVM",
      "accuracy": 0.90,
      "precision": 0.90,
      "recall": 0.90,
      "f1_score": 0.90,
      "training_seconds": 12.3,
      "confusion_matrix": [[4500, 500], [500, 4500]],
      "roc_curve": {"fpr": [0.0, 0.1, 1.0], "tpr": [0.0, 0.8, 1.0]}
    }
  ]
}
```

The values above are schema examples only, not project results. Replace them with measured experiment output.
Metrics can be stored as fractions (0–1) or percentages (0–100). The frontend normalizes both.
Confusion-matrix order is `[[TN, FP], [FN, TP]]`.
