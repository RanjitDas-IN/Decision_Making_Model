import os
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import RobertaModel, RobertaTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Paths and directories
MODELS_DIR = "Experimenting_SVM_models"
os.makedirs(MODELS_DIR, exist_ok=True)
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "svm_rbf_tuned.joblib")
TOKENS_PATH = (r"/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens.pt")
CSV_PATH = (r"/home/ranjit/Desktop/Decision_Making_Model/Pipeline_1_Data_Acquisition/Day2_cleaned_dataset.csv")


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def train_model(X_train, y_train, C=4, gamma=0.001):
    svm = SVC(kernel='rbf', C=C, gamma=gamma)
    svm.fit(X_train, y_train)
    return svm


def evaluate_model(model, X_test, y_test, intent_names=None):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=intent_names))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (raw):")
    print(cm)

    save_eval_report_path = os.path.join(MODELS_DIR, 'eval_report.csv')
    report_dict = classification_report(y_test, y_pred, target_names=intent_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    confusion_df = pd.DataFrame(cm)

    with open(save_eval_report_path, 'w', encoding='utf-8') as f:
        f.write(f"Accuracy,,{acc:.4f}\n\n")
        f.write("Classification Report:\n")
        report_df.to_csv(f)
        f.write("\n\nConfusion Matrix:\n")
        confusion_df.to_csv(f)

    return cm


def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    plt.figure(dpi=300)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


def main(X, y, intent_names=None):
    # 1. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 2. Scale features
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved scaler to {SCALER_PATH}")

    # 3. Train SVM with fixed hyperparameters
    best_model = train_model(X_train_scaled, y_train, C=4, gamma=0.001)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved SVM model with fixed params to {MODEL_PATH}")

    # 4. Evaluate
    cm = evaluate_model(best_model, X_test_scaled, y_test, intent_names)
    plot_confusion_matrix(cm, intent_names)


if __name__ == '__main__':
    device = torch.device("cpu")
    tokens = torch.load(TOKENS_PATH, map_location=device)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    df = pd.read_csv(CSV_PATH, sep='|')
    targets = df['intent'].tolist()
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder().fit(targets)
    y = label_encoder.transform(targets)
    intent_names = label_encoder.classes_

    print("Loading RobertaModel and extracting CLS embeddings...")
    roberta = RobertaModel.from_pretrained('roberta-base').to(device)
    roberta.eval()
    for param in roberta.parameters():
        param.requires_grad = False

    batch_size = 32
    emb_list = []
    with torch.no_grad():
        for i in tqdm(range(0, input_ids.size(0), batch_size), desc="Embedding Batches"):
            batch_ids = input_ids[i:i + batch_size].to(device)
            batch_mask = attention_mask[i:i + batch_size].to(device)
            out = roberta(input_ids=batch_ids, attention_mask=batch_mask)
            emb_list.append(out.last_hidden_state[:, 0, :].cpu())
    X = torch.cat(emb_list, 0).numpy()

    main(X, y, intent_names)
