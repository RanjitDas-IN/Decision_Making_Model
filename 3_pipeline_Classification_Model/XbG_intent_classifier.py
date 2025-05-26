import os
import time
import joblib
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Paths and constants
MODELS_DIR = "XgB_models"
CSV_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Data_Acquisition/Day2_cleaned_dataset.csv"
TOKENS_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens.pt"
MODEL_PATH = os.path.join(MODELS_DIR, "svm_rbf_tuned.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
RANDOM_STATE = 42
TEST_SIZE = 0.2
batch_size = 300
NUM_CLASSES = 10


# 1. Load tokens
print("Loading precomputed tokens...")
device = torch.device('cpu')
tokens = torch.load(TOKENS_PATH, map_location=device)
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# 2. Load DataFrame and encode labels
df = pd.read_csv(CSV_PATH, sep='|')
targets = df['intent'].tolist()
label_encoder = LabelEncoder().fit(targets)
y = label_encoder.transform(targets)
intent_names = label_encoder.classes_

# 3. Extract RoBERTa [CLS] embeddings
print("Extracting embeddings from RoBERTa...")
roberta = RobertaModel.from_pretrained('roberta-base')
roberta.eval()
for param in roberta.parameters():
    param.requires_grad = False

emb_list = []
with torch.no_grad():
    for i in tqdm(range(0, input_ids.size(0), batch_size), desc="Embedding Batches"):
        batch_ids = input_ids[i:i + batch_size].to(device)
        batch_mask = attention_mask[i:i + batch_size].to(device)
        out = roberta(input_ids=batch_ids, attention_mask=batch_mask)
        emb_list.append(out.last_hidden_state[:, 0, :].cpu())

X = torch.cat(emb_list, 0).numpy()

# ✅ Sanity check
assert len(X) == len(y), f"Mismatch: {len(X)} embeddings vs {len(y)} labels"
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 5. Initialize XGBClassifier with best-found hyperparameters
xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=NUM_CLASSES,
    eval_metric='mlogloss',
    random_state=RANDOM_STATE,
    colsample_bytree=1.0,
    learning_rate=0.2,
    max_depth=5,
    n_estimators=100,
    subsample=0.8
)

# Compute balanced weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Fit using weights
xgb.fit(X_train, y_train, sample_weight=sample_weights)

# 7. Evaluate on test set
y_pred = xgb.predict(X_test)


test_acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=intent_names))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


with open("XgB_save_eval_report_path.txt", 'w', encoding='utf-8') as f:
    f.write(f"Accuracy,,{test_acc:.4f}\n\n")
    f.write("Classification Report:\n")
    report=classification_report(y_test, y_pred, target_names=intent_names)
    f.write(report)
    f.write("\n\nConfusion Matrix:\n")
    cmx=confusion_matrix(y_test, y_pred)
    f.write(str(cmx))


# 8. Save LabelEncoder and model
joblib.dump(label_encoder, LABEL_ENCODER_PATH)
joblib.dump(xgb, MODEL_PATH)
print(f"Saved LabelEncoder and trained XGBoost model to disk.")