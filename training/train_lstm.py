from pathlib import Path
from collections import Counter
import json, random, time
import numpy as np
import pandas as pd
import joblib
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "dataset" / "experiments"
OUT = ROOT / "model" / "comparison"
RESULTS = ROOT / "results" / "comparison_results.json"

SEED, MAX_VOCAB, MAX_LEN = 42, 30000, 250
EMBED_DIM, HIDDEN_DIM, BATCH_SIZE, EPOCHS = 128, 128, 128, 6
PATIENCE, LR = 2, 1e-3
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cpu")
print("Device:", device, "| PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())

train = pd.read_csv(DATA / "train.csv")
valid = pd.read_csv(DATA / "validation.csv")
test = pd.read_csv(DATA / "test.csv")
for name, df in [("train", train), ("validation", valid), ("test", test)]:
    if not {"clean_review", "label"}.issubset(df.columns):
        raise ValueError(f"{name}.csv must contain clean_review and label columns")

# Vocabulary is built from training data only.
counts = Counter()
for review in train.clean_review.fillna("").astype(str):
    counts.update(review.split())
vocab = {"<PAD>": 0, "<UNK>": 1}
for word, _ in counts.most_common(MAX_VOCAB - 2):
    vocab[word] = len(vocab)

def encode(review):
    ids = [vocab.get(w, 1) for w in str(review).split()[:MAX_LEN]] or [1]
    length = len(ids)
    return ids + [0] * (MAX_LEN - length), length

class Reviews(Dataset):
    def __init__(self, frame):
        items = [encode(x) for x in frame.clean_review.fillna("").astype(str)]
        self.x = torch.tensor([x[0] for x in items], dtype=torch.long)
        self.lengths = torch.tensor([x[1] for x in items], dtype=torch.long)
        self.y = torch.tensor(frame.label.astype(int).to_numpy(), dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.lengths[i], self.y[i]

train_ds, val_ds, test_ds = Reviews(train), Reviews(valid), Reviews(test)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(HIDDEN_DIM, 1)
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.fc(self.dropout(hidden[-1])).squeeze(1)

model = SentimentLSTM(len(vocab)).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def validation_loss():
    model.eval(); total = 0.0; n = 0
    with torch.no_grad():
        for x, lengths, y in val_loader:
            logits = model(x.to(device), lengths.to(device))
            loss = loss_fn(logits, y.to(device))
            total += loss.item() * len(y); n += len(y)
    return total / n

best_loss, best_state, no_improve = float("inf"), None, 0
start = time.perf_counter()
for epoch in range(1, EPOCHS + 1):
    model.train(); total = 0.0; n = 0
    for batch_i, (x, lengths, y) in enumerate(train_loader, 1):
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x, lengths), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * len(y); n += len(y)
        if batch_i % 100 == 0:
            print(f"Epoch {epoch}: batch {batch_i}/{len(train_loader)}")
    val_loss = validation_loss()
    print(f"Epoch {epoch}/{EPOCHS} | train_loss={total/n:.4f} | val_loss={val_loss:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("Early stopping.")
            break
training_seconds = time.perf_counter() - start
if best_state is None: raise RuntimeError("No best model state was produced.")
model.load_state_dict(best_state); model.eval()

probs, labels = [], []
start = time.perf_counter()
with torch.no_grad():
    for x, lengths, y in test_loader:
        p = torch.sigmoid(model(x.to(device), lengths.to(device)))
        probs.extend(p.cpu().numpy().tolist())
        labels.extend(y.numpy().astype(int).tolist())
inference_seconds = time.perf_counter() - start
preds = (np.array(probs) >= 0.5).astype(int)
labels = np.array(labels)
cm = confusion_matrix(labels, preds, labels=[0, 1])
print("\\nLSTM test report:")
print(classification_report(labels, preds, target_names=["Negative", "Positive"], zero_division=0))

metrics = {
    "model": "LSTM (Embedding + PyTorch LSTM)",
    "accuracy": float(accuracy_score(labels, preds)),
    "precision": float(precision_score(labels, preds, zero_division=0)),
    "recall": float(recall_score(labels, preds, zero_division=0)),
    "f1": float(f1_score(labels, preds, zero_division=0)),
    "roc_auc": float(roc_auc_score(labels, probs)),
    "training_seconds": round(training_seconds, 3),
    "inference_seconds_total": round(inference_seconds, 3),
    "test_samples": int(len(labels)),
    "confusion_matrix": cm.tolist(),
    "class_order": ["Negative", "Positive"],
    "split": {"train": len(train), "validation": len(valid), "test": len(test)},
    "training_config": {"max_vocab": MAX_VOCAB, "actual_vocab_size": len(vocab), "max_len": MAX_LEN,
                        "embedding_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM, "batch_size": BATCH_SIZE,
                        "epochs_configured": EPOCHS, "best_validation_loss": float(best_loss), "device": "cpu"}
}

OUT.mkdir(parents=True, exist_ok=True)
torch.save({"model_state_dict": model.state_dict(), "vocab_size": len(vocab),
            "embedding_dim": EMBED_DIM, "hidden_dim": HIDDEN_DIM, "max_len": MAX_LEN, "vocab": vocab},
           OUT / "lstm_model.pt")
joblib.dump(vocab, OUT / "lstm_vocab.joblib")

# Preserve the existing SVM results and add/replace only the LSTM entry.
if RESULTS.exists():
    try:
        with open(RESULTS, encoding="utf-8") as f: existing = json.load(f)
    except (json.JSONDecodeError, OSError): existing = {}
else: existing = {}
if isinstance(existing, list): existing = {"models": existing}
if not isinstance(existing, dict): existing = {}
if isinstance(existing.get("models"), list):
    existing["models"] = [m for m in existing["models"] if not (isinstance(m, dict) and str(m.get("model", "")).startswith("LSTM"))]
    existing["models"].append(metrics)
else:
    existing["LSTM"] = metrics
RESULTS.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS, "w", encoding="utf-8") as f: json.dump(existing, f, ensure_ascii=False, indent=2)

print("\\nMetrics:")
print(json.dumps(metrics, ensure_ascii=False, indent=2))
print("\\nSaved:", OUT / "lstm_model.pt")
print("Updated:", RESULTS)
print("Original production model files were not changed.")
