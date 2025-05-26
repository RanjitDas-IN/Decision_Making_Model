import os
import time
import joblib
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaModel
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# Paths and constants
CSV_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Data_Acquisition/Day2_cleaned_dataset.csv"
TOKENS_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens.pt"
MODEL_OUTPUT_PATH = 'hyper_tuning_xgb_intent_model.joblib'
RANDOM_STATE = 42
TEST_SIZE = 0.2
BATCH_SIZE = 300
NUM_CLASSES = 10

# Persistent progress tracking
progress_file = "progress_tracker.txt"

# Global shared objects
roberta = None
input_ids = None
attention_mask = None
X = []
y = None
label_encoder = None

# Load once
def initialize():
    global input_ids, attention_mask, y, label_encoder, roberta
    print("Initializing once...")
    device = torch.device('cpu')
    tokens = torch.load(TOKENS_PATH, map_location=device)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    df = pd.read_csv(CSV_PATH, sep='|')
    targets = df['intent'].tolist()
    label_encoder = LabelEncoder().fit(targets)
    y = label_encoder.transform(targets)

    roberta = RobertaModel.from_pretrained('roberta-base')
    roberta.eval()
    for param in roberta.parameters():
        param.requires_grad = False


def get_last_processed_batch():
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            return int(f.read().strip())
    return 0

def save_last_processed_batch(batch_index):
    with open(progress_file, 'w') as f:
        f.write(str(batch_index))

def main_batch(batch_index):
    print(f"Processing batch {batch_index}...")
    device = torch.device('cpu')
    start = batch_index * BATCH_SIZE
    end = min((batch_index + 1) * BATCH_SIZE, input_ids.size(0))
    if start >= end:
        print("All batches processed. Proceeding to training phase.")
        return False  # Signal to stop further batching

    batch_ids = input_ids[start:end].to(device)
    batch_mask = attention_mask[start:end].to(device)
    with torch.no_grad():
        out = roberta(input_ids=batch_ids, attention_mask=batch_mask)
        X_batch = out.last_hidden_state[:, 0, :].cpu().numpy()
        X.extend(X_batch)

    save_last_processed_batch(batch_index + 1)
    return True

def train_model():
    global X, y, label_encoder
    X = np.array(X)

    assert len(X) == len(y), f"Mismatch: {len(X)} embeddings vs {len(y)} labels"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=NUM_CLASSES,
        eval_metric='mlogloss',
        random_state=42
    )

    param_grid = {
        'max_depth': [3, 5],
        'n_estimators': [20, 50],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    print("Starting GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='f1_weighted',
        cv=3,
        verbose=2,
        n_jobs=10
    )

    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    grid_search.fit(X_train, y_train, sample_weight=sample_weights)

    print("\nBest parameters found:", grid_search.best_params_)
    print("Best cross-validation F1 score:", grid_search.best_score_)

    y_pred = grid_search.best_estimator_.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    with open("HyperTune_XgB_eval_report_path.txt", 'w', encoding='utf-8') as f:
        f.write(f"Accuracy,,{test_acc:.4f}\n\n")
        f.write("Classification Report:\n")
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))

    joblib.dump(grid_search.best_estimator_, MODEL_OUTPUT_PATH)
    print(f"Saved trained XGBoost model to {MODEL_OUTPUT_PATH}")



def run_cycle():
    active_duration = 5 * 60
    sleep_duration = 90
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        if elapsed >= active_duration:
            print("\n5 minutes done. Sleeping for 90 seconds...\n")
            time.sleep(sleep_duration)
            break

        batch_index = get_last_processed_batch()
        continue_work = main_batch(batch_index)
        if not continue_work:
            print("All batches completed. Training model now.")
            train_model()
            exit()  # End after training

        time.sleep(1)  # Prevent 100% CPU usage


if __name__ == "__main__":
    initialize()
    while True:
        run_cycle()
