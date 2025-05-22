import os
import torch
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from transformers import RobertaModel, RobertaTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Context: X and y are preloaded RoBERTa CLS embeddings and encoded intent labels
# Paths and directories
MODELS_DIR = "Experimenting_SVM_models"
os.makedirs(MODELS_DIR, exist_ok=True)
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(MODELS_DIR, "svm_rbf_tuned.joblib")
HEATMAP_PATH = os.path.join(MODELS_DIR, "hyperparam_landscape.png")
TOKENS_PATH=(r"/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens.pt")
CSV_PATH=(r"/home/ranjit/Desktop/Decision_Making_Model/Pipeline_1_Data_Acquisition/Day2_cleaned_dataset.csv")


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and labels into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def scale_features(X_train, X_test):
    """
    Fit a StandardScaler on training data and transform both train and test.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def perform_grid_search(X_train, y_train, param_grid, cv=3, n_jobs=-1):
    """
    Perform GridSearchCV with SVM (RBF kernel).
    """
    svm = SVC(kernel='rbf')
    grid = GridSearchCV(svm, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs, return_train_score=True)
    grid.fit(X_train, y_train)
    return grid


def train_final_model(grid, X_train, y_train):
    """
    Train SVM with best parameters on the full training set.
    """
    best_svm = grid.best_estimator_
    best_svm.fit(X_train, y_train)
    return best_svm


def evaluate_model(model, X_test, y_test, intent_names=None):
    """
    Print accuracy and classification report, return confusion matrix.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=intent_names))
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (raw):")
    print(cm)

    # Save evaluation report
    save_eval_report_path = os.path.join(MODELS_DIR, 'eval_report.csv')

    # Compute metrics for saving
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
    """
    Plot and show a confusion matrix with labels.
    """
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


def plot_hyperparameter_landscape(grid, param_grid):
    """
    Scatter-style plot of mean CV score for each C and gamma combination.
    """
    results = grid.cv_results_
    C_vals = results['param_C'].data if hasattr(results['param_C'], 'data') else results['param_C']
    gamma_vals = results['param_gamma'].data if hasattr(results['param_gamma'], 'data') else results['param_gamma']
    mean_scores = results['mean_test_score']

    plt.figure(dpi=300)
    sc = plt.scatter(
        x=[float(g) for g in gamma_vals],
        y=[float(c) for c in C_vals],
        c=mean_scores,
        edgecolor='k'
    )
    for i, txt in enumerate(np.round(mean_scores, 3)):
        plt.annotate(str(txt), (float(gamma_vals[i]), float(C_vals[i])), fontsize=7, ha='center')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.title('Hyperparameter Landscape (mean CV accuracy)')
    plt.colorbar(sc, label='Mean CV Accuracy')
    plt.tight_layout()
    plt.savefig(HEATMAP_PATH)
    # plt.show()  # Disabled in non-interactive environments
    plt.savefig(HEATMAP_PATH)


def main(X, y, intent_names=None):
    # 1. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # 2. Scale features
    scaler, X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved scaler to {SCALER_PATH}")

    # 3. Grid search for hyperparameters            # Best parameters: {'C': 4, 'gamma': 0.001},   
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1]
    }
    grid = perform_grid_search(X_train_scaled, y_train, param_grid)
    print(f"Best parameters: {grid.best_params_}, CV Score: {grid.best_score_:.4f}")

    # 4. Train final model
    best_model = train_final_model(grid, X_train_scaled, y_train)
    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved tuned SVM model to {MODEL_PATH}")

    # 5. Evaluate
    cm = evaluate_model(best_model, X_test_scaled, y_test, intent_names)
    plot_confusion_matrix(cm, intent_names)

    # 6. Hyperparameter landscape
    plot_hyperparameter_landscape(grid, param_grid)



# Example call (uncomment and provide X, y, and intent_names):
if __name__ == '__main__':
    # Load RoBERTa tokens
    device = torch.device("cpu")
    tokens = torch.load(TOKENS_PATH, map_location=device)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']
    # (If embeddings are not precomputed, extract them here. E.g., use extract_embeddings function.)

    # Load intent labels (pipe-delimited columns: 'utterance' | 'intent'
    df = pd.read_csv(CSV_PATH, sep='|')  # columns: ['utterance', 'intent']
    targets = df['intent'].tolist()
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder().fit(targets)
    y = label_encoder.transform(targets)
    intent_names = label_encoder.classes_


    # 4. Extract CLS embeddings
    print("Loading RobertaModel and extracting CLS embeddings...")
    roberta = RobertaModel.from_pretrained('roberta-base').to(device)
    roberta.eval()
    for param in roberta.parameters(): param.requires_grad = False
    batch_size = 32
    emb_list = []
    with torch.no_grad():
        for i in range(0, input_ids.size(0), batch_size):
            batch_ids = input_ids[i:i+batch_size].to(device)
            batch_mask = attention_mask[i:i+batch_size].to(device)
            out = roberta(input_ids=batch_ids, attention_mask=batch_mask)
            emb_list.append(out.last_hidden_state[:,0,:].cpu())
            # print("Emb_list:\t",emb_list)
    X = torch.cat(emb_list, 0).numpy()
    # print("\n\nX:\t",X)
    main(X, y, intent_names)
