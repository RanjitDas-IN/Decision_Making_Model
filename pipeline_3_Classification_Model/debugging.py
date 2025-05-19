import os
import torch
import pandas as pd
from transformers import RobertaModel, RobertaTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Paths
TOKENS_PATH = "/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens.pt"
CSV_PATH = "/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Intent_Data_Acquisition/Day2_cleaned_dataset.csv"
MODELS_DIR = "models"

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Device (CPU only)
device = torch.device("cpu")

# 1. Load tokenized inputs
print("Loading tokenized inputs...")
tokens = torch.load(TOKENS_PATH, map_location=device)
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

# 2. Load original CSV and filter rare classes
print("Loading dataset CSV and filtering rare intents...")
df = pd.read_csv(CSV_PATH, sep='|')
# Identify intent counts
token_intents = df['intent']
intent_counts = token_intents.value_counts()
# Find intents with fewer than 2 samples
rare_intents = intent_counts[intent_counts < 2]
if not rare_intents.empty:
    print(f"Dropping {rare_intents.sum()} samples from {len(rare_intents)} rare classes: {list(rare_intents.index)}")
# Create mask for valid intents
valid_mask = token_intents.isin(intent_counts[intent_counts >= 2].index)
# Filter dataframe and token tensors
df = df[valid_mask].reset_index(drop=True)
input_ids = input_ids[valid_mask.values]
attention_mask = attention_mask[valid_mask.values]
# Updated intents list
target_intents = df['intent'].tolist()

# 3. Initialize frozen RoBERTa model & tokenizer Initialize frozen RoBERTa model & tokenizer
print("Loading RobertaModel and tokenizer...")
roberta = RobertaModel.from_pretrained('roberta-base')
roberta.to(device)
roberta.eval()
for param in roberta.parameters():
    param.requires_grad = False

# (Optional) Save the frozen RoBERTa for later inference
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta.save_pretrained(os.path.join(MODELS_DIR, 'roberta_frozen'))
tokenizer.save_pretrained(os.path.join(MODELS_DIR, 'roberta_tokenizer'))

# 4. Extract CLS embeddings batch-wise
print("Extracting CLS embeddings...")
batch_size = 32
embeddings = []
with torch.no_grad():
    for i in range(0, len(input_ids), batch_size):
        batch_ids = input_ids[i : i + batch_size].to(device)
        batch_mask = attention_mask[i : i + batch_size].to(device)
        outputs = roberta(input_ids=batch_ids, attention_mask=batch_mask)
        cls_embeds = outputs.last_hidden_state[:, 0, :]
        embeddings.append(cls_embeds.cpu())

embeddings = torch.cat(embeddings, dim=0).numpy()

# 5. Encode labels
label_encoder = LabelEncoder().fit(target_intents)
y = label_encoder.transform(target_intents)

# 6. Split data (stratified 80/20)
X_train, X_test, y_train, y_test = train_test_split(
    embeddings,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7. Train Logistic Regression
print("Training Logistic Regression (solver=lbfgs, class_weight=balanced)...")
from sklearn.multiclass import OneVsRestClassifier
base_clf = LogisticRegression(
    solver='lbfgs',
    class_weight='balanced',
    max_iter=3000
)
clf = OneVsRestClassifier(base_clf)

# 8. Evaluate on test set
print("Evaluating on test set...")
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print("Confusion Matrix:")
print(cm)

# Save evaluation artifacts
report_df = pd.DataFrame(report).T
report_path = os.path.join(MODELS_DIR, 'eval_report.csv')
report_df.to_csv(report_path)
print(f"Saved classification report to {report_path}")

# 9. Save models and encoders
model_path = os.path.join(MODELS_DIR, 'logreg_model.joblib')
encoder_path = os.path.join(MODELS_DIR, 'label_encoder.joblib')
joblib.dump(clf, model_path)
joblib.dump(label_encoder, encoder_path)
print(f"Saved LogisticRegression to {model_path} and LabelEncoder to {encoder_path}")

# === Inference Snippet ===
# To load and predict a new utterance:
#
# from transformers import RobertaModel, RobertaTokenizer
# import torch, joblib
#
# # Load frozen featurizer and classifier
# roberta_inf = RobertaModel.from_pretrained('models/roberta_frozen')
# tokenizer_inf = RobertaTokenizer.from_pretrained('models/roberta_tokenizer')
# clf_inf = joblib.load('models/logreg_model.joblib')
# le_inf = joblib.load('models/label_encoder.joblib')
#
# # Prepare input
# text = "Your new user utterance here"
# inputs = tokenizer_inf([text], padding='max_length', truncation=True, max_length=128, return_tensors='pt')
#
# # Extract embedding and predict
# with torch.no_grad():
#     emb = roberta_inf(**inputs).last_hidden_state[:, 0, :].numpy()
# pred_id = clf_inf.predict(emb)[0]
# intent_label = le_inf.inverse_transform([pred_id])[0]
# print(f"Predicted intent: {intent_label}")
