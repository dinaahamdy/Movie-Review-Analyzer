import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # للعمل بدون GUI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from gensim.models import FastText

# ============================================================
# 0. الإعدادات والمسارات
# ============================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'dataset', 'movies_preprocessed.csv')
model_dir = os.path.join(script_dir, 'model')
reports_dir = os.path.join(script_dir, 'reports')
os.makedirs(reports_dir, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.bbox'] = 'tight'

COLORS = {
    'LinearSVC': '#1f77b4',
    'LogisticRegression': '#ff7f0e',
    'GaussianNB': '#2ca02c',
    'RandomForest': '#d62728',
}

# ============================================================
# 1. تحميل الداتا وتجهيزها (نفس منطق train_model.py)
# ============================================================
print("Loading dataset...")
df = pd.read_csv(dataset_path)
df = df.dropna(subset=['review', 'sentiment'])

# نفس الـ synthetic data بالظبط عشان الـ split يطلع متطابق
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
df = pd.concat([df, synthetic_data], ignore_index=True)

df['sentiment_encoded'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df = df.dropna(subset=['sentiment_encoded'])
df['sentiment_encoded'] = df['sentiment_encoded'].astype(int)

X = df['review']
y = df['sentiment_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 2. تحميل FastText وتجهيز الـ vectors
# ============================================================
print("Loading FastText model...")
fasttext_model = FastText.load(os.path.join(model_dir, 'fasttext_model.model'))

def get_sentence_vector(text):
    tokens = text.split()
    vectors = [fasttext_model.wv[w] for w in tokens if w in fasttext_model.wv]
    if not vectors:
        return np.zeros(100)
    return np.mean(vectors, axis=0)

print("Converting text to vectors...")
X_train_vec = np.array([get_sentence_vector(t) for t in X_train])
X_test_vec = np.array([get_sentence_vector(t) for t in X_test])

# ============================================================
# 3. إعادة تدريب الموديلات الأربعة (خفيف، عشان الرسومات)
# ============================================================
print("Training 4 models for visualization...")

models = {
    "LinearSVC": CalibratedClassifierCV(
        LinearSVC(C=1.0, max_iter=5000, random_state=42), cv=3
    ),
    "LogisticRegression": LogisticRegression(max_iter=2000, random_state=42),
    "GaussianNB": GaussianNB(),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
}

predictions = {}
probabilities = {}

for name, clf in models.items():
    print(f"  Training {name}...")
    clf.fit(X_train_vec, y_train)
    predictions[name] = clf.predict(X_test_vec)
    probabilities[name] = clf.predict_proba(X_test_vec)[:, 1]

# ============================================================
# 4. الرسمة 1: توزيع المشاعر + طول المراجعات
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# 4a: Bar chart للتوزيع
counts = df['sentiment_encoded'].value_counts().sort_index()
axes[0].bar(['Negative', 'Positive'], counts.values,
            color=['#d62728', '#2ca02c'], edgecolor='black')
for i, v in enumerate(counts.values):
    axes[0].text(i, v + max(counts.values) * 0.01, f'{v:,}',
                 ha='center', fontweight='bold')
axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Reviews')

# 4b: Pie chart
axes[1].pie(counts.values, labels=['Negative', 'Positive'],
            colors=['#d62728', '#2ca02c'], autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Class Balance', fontsize=13, fontweight='bold')

# 4c: طول المراجعات بعد الـ preprocessing
review_lengths = df['review'].str.split().str.len()
axes[2].hist(review_lengths, bins=50, color='#1f77b4',
             edgecolor='black', alpha=0.7)
axes[2].axvline(review_lengths.median(), color='red', linestyle='--',
                label=f'Median = {review_lengths.median():.0f} words')
axes[2].set_title('Review Length Distribution (after preprocessing)',
                  fontsize=13, fontweight='bold')
axes[2].set_xlabel('Number of Words')
axes[2].set_ylabel('Frequency')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(reports_dir, '01_data_distribution.png'))
plt.close()
print("  ✅ 01_data_distribution.png")

# ============================================================
# 5. الرسمة 2: Confusion Matrices (2x2 subplots)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, y_pred) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=axes[idx], cbar=False,
                annot_kws={'size': 14, 'fontweight': 'bold'})
    axes[idx].set_title(f'{name}', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.suptitle('Confusion Matrices — All Models', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, '02_confusion_matrices.png'))
plt.close()
print("  ✅ 02_confusion_matrices.png")

# ============================================================
# 6. الرسمة 3: مقارنة المقاييس (Grouped Bar Chart)
# ============================================================
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics_data = {}
for name, y_pred in predictions.items():
    metrics_data[name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
    }

metrics_df = pd.DataFrame(metrics_data).T
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(metric_names))
width = 0.2

for i, model_name in enumerate(metrics_df.index):
    values = metrics_df.loc[model_name, metric_names].values
    ax.bar(x + i * width, values, width, label=model_name,
           color=COLORS[model_name], edgecolor='black')

ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metric_names, fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Comparison — All Metrics', fontsize=14, fontweight='bold')
ax.set_ylim([0.7, 1.0])
ax.legend(loc='lower right', fontsize=11)
ax.grid(axis='y', alpha=0.3)

# إضافة الأرقام فوق كل bar
for i, model_name in enumerate(metrics_df.index):
    values = metrics_df.loc[model_name, metric_names].values
    for j, v in enumerate(values):
        ax.text(j + i * width, v + 0.005, f'{v:.3f}',
                ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(reports_dir, '03_metrics_comparison.png'))
plt.close()
print("  ✅ 03_metrics_comparison.png")

# ============================================================
# 7. الرسمة 4: ROC Curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))

for name, y_prob in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=COLORS[name], lw=2,
            label=f'{name} (AUC = {roc_auc:.3f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(reports_dir, '04_roc_curves.png'))
plt.close()
print("  ✅ 04_roc_curves.png")

# ============================================================
# 8. الرسمة 5: Precision-Recall Curves
# ============================================================
fig, ax = plt.subplots(figsize=(8, 7))

for name, y_prob in probabilities.items():
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    ax.plot(recall, precision, color=COLORS[name], lw=2,
            label=f'{name} (AP = {ap:.3f})')

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves — All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.grid(alpha=0.3)
ax.set_xlim([0, 1.0])
ax.set_ylim([0.7, 1.02])

plt.tight_layout()
plt.savefig(os.path.join(reports_dir, '05_precision_recall_curves.png'))
plt.close()
print("  ✅ 05_precision_recall_curves.png")

# ============================================================
# 9. الرسمة 6: Learning Curves (للموديلين الخطيين)
# ============================================================
# نستخدم عينة لتسريع الحساب
sample_size = min(8000, len(X_train_vec))
rng = np.random.RandomState(42)
idx = rng.choice(len(X_train_vec), sample_size, replace=False)
X_train_sample = X_train_vec[idx]
y_train_sample = y_train.iloc[idx].values

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

learning_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "LinearSVC": LinearSVC(C=1.0, max_iter=3000, random_state=42),
}

for ax, (name, clf) in zip(axes, learning_models.items()):
    print(f"  Computing learning curve for {name}...")
    train_sizes, train_scores, val_scores = learning_curve(
        clf, X_train_sample, y_train_sample,
        train_sizes=np.linspace(0.1, 1.0, 6),
        cv=3, scoring='f1', n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    ax.plot(train_sizes, train_mean, 'o-', color='#1f77b4',
            label='Training score')
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.15, color='#1f77b4')
    ax.plot(train_sizes, val_mean, 'o-', color='#d62728',
            label='Cross-validation score')
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.15, color='#d62728')

    ax.set_xlabel('Training Set Size', fontsize=11)
    ax.set_ylabel('F1-Score', fontsize=11)
    ax.set_title(f'Learning Curve — {name}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(reports_dir, '06_learning_curves.png'))
plt.close()
print("  ✅ 06_learning_curves.png")

# ============================================================
# 10. الرسمة 7: Feature Importance (LogisticRegression)
# ============================================================
print("  Computing feature importance...")
# ندرّب LR عشان نستخرج الـ coefficients (أسهل وأوضح من LinearSVC)
lr = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_train_vec, y_train)
coefs = lr.coef_[0]

# نربط كل بعد (dimension) بأكثر كلمة قريبة له في FastText
# (لأن الـ vectors مش مرتبطة بكلمات محددة، بنعمل تقريب)
vocab = list(fasttext_model.wv.index_to_key)[:5000]
vocab_vectors = np.array([fasttext_model.wv[w] for w in vocab])

# لكل بعد، نلاقي الكلمة الأقرب ليها (أعلى cosine similarity)
# نطبع أعلى وأقل 15 بعد
top_dims = np.argsort(coefs)[-15:][::-1]
bottom_dims = np.argsort(coefs)[:15]

# نستخدم cosine similarity لإيجاد الكلمة الممثلة لكل بعد
def find_nearest_words(dims, n_words=1):
    dim_vecs = np.zeros((len(dims), vocab_vectors.shape[1]))
    for i, d in enumerate(dims):
        dim_vecs[i, d] = 1.0
    # cosine similarity
    sims = dim_vecs @ vocab_vectors.T
    top_words = []
    for s in sims:
        idx = np.argsort(s)[-n_words:][::-1]
        top_words.append([vocab[j] for j in idx])
    return top_words

top_words = find_nearest_words(top_dims)
bottom_words = find_nearest_words(bottom_dims)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Positive indicators
axes[0].barh(range(len(top_dims)),
             coefs[top_dims],
             color='#2ca02c', edgecolor='black')
axes[0].set_yticks(range(len(top_dims)))
axes[0].set_yticklabels([w[0] for w in top_words], fontsize=9)
axes[0].set_title('Top 15 Features → Positive Sentiment',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('Coefficient Value')
axes[0].invert_yaxis()

# Negative indicators
axes[1].barh(range(len(bottom_dims)),
             coefs[bottom_dims],
             color='#d62728', edgecolor='black')
axes[1].set_yticks(range(len(bottom_dims)))
axes[1].set_yticklabels([w[0] for w in bottom_words], fontsize=9)
axes[1].set_title('Top 15 Features → Negative Sentiment',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('Coefficient Value')
axes[1].invert_yaxis()

plt.suptitle('Feature Importance (LogisticRegression coefficients)\n'
             '⚠️ Note: labels are nearest words in FastText space, not exact features',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, '07_feature_importance.png'))
plt.close()
print("  ✅ 07_feature_importance.png")

# ============================================================
# 11. ملخص نهائي في ملف نصي
# ============================================================
summary_path = os.path.join(reports_dir, 'summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("MODEL COMPARISON SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(metrics_df.round(4).to_string())
    f.write("\n\n")
    f.write("Best model by F1: " + metrics_df['F1'].idxmax() + "\n")
    f.write("Best model by Accuracy: " + metrics_df['Accuracy'].idxmax() + "\n")

print(f"\n✅ All visualizations saved to: {reports_dir}/")
print(f"✅ Summary saved to: {summary_path}")
print("\nGenerated files:")
for f_name in sorted(os.listdir(reports_dir)):
    print(f"   - {f_name}")