# clustered_cv_trainer.py (5-fold CV over semantic clusters)

import pandas as pd
import numpy as np
import sentencepiece as spm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# === CONFIG ===
data_path = r"/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Intent_Data_Acquisition/Day2_cleaned_dataset.csv"
tokenizer_model_path = r"/home/ranjit/Desktop/Decision_Making_Model/tokenizer.model"
sim_threshold = 0.90
k_folds = 5

# === LOAD TOKENIZER ===
sp = spm.SentencePieceProcessor(model_file=tokenizer_model_path)

# === LOAD DATA ===
df = pd.read_csv(data_path, sep='|', names=["utterance", "intent"], header=None, dtype=str)
df = df.dropna(subset=["utterance", "intent"]).reset_index(drop=True)

# Filter out labels that occur only once (rare/typo classes)
label_counts = df['intent'].value_counts()
valid_labels = label_counts[label_counts > 1].index
df = df[df['intent'].isin(valid_labels)].reset_index(drop=True)


# === TOKENIZE ===
def encode_to_tokens(text):
    try:
        return " ".join(sp.encode(text, out_type=str))
    except:
        return ""

X_raw = df['utterance'].apply(encode_to_tokens)
y = df['intent']

# === VECTORIZING ===
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vect = vectorizer.fit_transform(X_raw)

# === SEMANTIC CLUSTERING ===
sim_matrix = cosine_similarity(X_vect)
assigned = set()
clusters = []
for i in range(len(df)):
    if i in assigned:
        continue
    cluster = [i]
    for j in range(i + 1, len(df)):
        if j not in assigned and sim_matrix[i, j] >= sim_threshold:
            cluster.append(j)
            assigned.add(j)
    assigned.add(i)
    clusters.append(cluster)
print(f"Formed {len(clusters)} semantic clusters")

# === CREATE FOLDS MANUALLY ===
np.random.seed(42)
np.random.shuffle(clusters)
fold_size = len(clusters) // k_folds
folds = []
for fold_idx in range(k_folds):
    start = fold_idx * fold_size
    end = (fold_idx + 1) * fold_size if fold_idx < k_folds - 1 else len(clusters)
    folds.append(clusters[start:end])

# === RUN CV ===
all_scores = []
for i, test_clusters in enumerate(folds):
    print(f"\n=== Fold {i + 1} ===")
    train_clusters = [cl for idx, cl in enumerate(folds) if idx != i]
    # Flatten the lists properly
    train_idx = [ix for cluster in train_clusters for cl in cluster for ix in cl]
    test_idx = [ix for cl in test_clusters for ix in cl]

    X_train, X_test = X_vect[np.array(train_idx)], X_vect[np.array(test_idx)]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Fold {i + 1} Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0)
)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    all_scores.append(acc)

# === SUMMARY ===
print("\n=== Cross-Validation Summary ===")
print("Fold Accuracies:", [round(s, 4) for s in all_scores])
print(f"Mean Accuracy: {np.mean(all_scores):.4f}")
