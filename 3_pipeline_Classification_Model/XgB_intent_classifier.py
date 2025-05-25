import os
import time
import joblib
import pandas as pd
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Paths and constants
CSV_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Data_Acquisition/Day2_cleaned_dataset.csv"
TOKENS_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens.pt"
MODEL_OUTPUT_PATH = 'xgb_intent_model.joblib'
RANDOM_STATE = 42
TEST_SIZE = 0.2
BATCH_SIZE = 32
NUM_CLASSES = 10


# class ThrottledXGBClassifier(XGBClassifier):
#     def fit(self, *args, **kwargs):
#         result = super().fit(*args, **kwargs)
#         print("✅ Sleeping for 30 seconds after this fold...")
#         time.sleep(30)
#         return result


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

embeddings = []
with torch.no_grad():
    for i in range(0, input_ids.size(0), BATCH_SIZE):
        batch_ids = input_ids[i:i + BATCH_SIZE].to(device)
        batch_mask = attention_mask[i:i + BATCH_SIZE].to(device)
        outputs = roberta(input_ids=batch_ids, attention_mask=batch_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_emb)

X = np.vstack(embeddings)
print(f"Embeddings shape: {X.shape}")

# ✅ Sanity check
assert len(X) == len(y), f"Mismatch: {len(X)} embeddings vs {len(y)} labels"
# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# 5. Define XGBClassifier and parameter grid
xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=10,
    eval_metric='mlogloss',
    random_state=42
)


param_grid = {
    'max_depth': [3, 5, 7],
    'n_estimators': [20, 50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# 6. Grid search
print("Starting GridSearchCV...")
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='f1_weighted',
    cv=3,
    verbose=2,
    n_jobs=10
)
grid_search.fit(X_train, y_train)

# 7. Best parameters and score
print("\nBest parameters found:", grid_search.best_params_)
print("Best cross-validation F1 score:", grid_search.best_score_)

# 8. Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

test_acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {test_acc:.4f}")
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

# 9. Save final model
joblib.dump(best_model, MODEL_OUTPUT_PATH)
print(f"Saved trained XGBoost model to {MODEL_OUTPUT_PATH}")
