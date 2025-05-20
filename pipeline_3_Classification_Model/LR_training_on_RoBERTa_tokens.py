import os
import torch
import pandas as pd
from transformers import RobertaModel, RobertaTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Paths
TOKENS_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/roberta_tokens.pt"
CSV_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/Pipeline_1_Data_Acquisition/Day2_cleaned_dataset.csv"
MODELS_DIR = "LR_models"

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Device (CPU only)
device = torch.device("cpu")

# 1. Load tokenized inputs
print("Loading tokenized inputs...")
tokens = torch.load(TOKENS_PATH, map_location=device)
input_ids = tokens['input_ids']
attention_mask = tokens['attention_mask']

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
embeddings = torch.cat(emb_list, 0).numpy()

# 5. Encode labels
df = pd.read_csv(CSV_PATH,sep='|')
targets = df['intent'].tolist()
label_encoder = LabelEncoder().fit(targets)
y = label_encoder.transform(targets)

# 6. Split data
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Hyperparameter tuning with GridSearchCV
print("Performing hyperparameter tuning...")
base = LogisticRegression(solver='lbfgs', class_weight='balanced')
ovr = OneVsRestClassifier(base)
param_grid = {'estimator__C':[0.01,0.1,1,10], 'estimator__max_iter':[1000,2000]}
grid = GridSearchCV(ovr, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}, CV score: {grid.best_score_:.4f}")
clf = grid.best_estimator_

# 8. Evaluate
print("Evaluating on test set...")
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
print(confusion_matrix(y_test, y_pred))

# Save eval report
save_eval_repor_path = os.path.join(MODELS_DIR, 'eval_report_combined.csv')

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

conf_mat = confusion_matrix(y_test, y_pred)
confusion_df = pd.DataFrame(conf_mat, 
                            index=label_encoder.classes_, 
                            columns=label_encoder.classes_)

# Write all sections to file
with open(save_eval_repor_path, 'w', encoding='utf-8') as f:
    f.write(f"Accuracy,,{accuracy:.4f}\n\n")
    f.write("Classification Report:\n")
    report_df.to_csv(f)
    f.write("\n\nConfusion Matrix:\n")
    confusion_df.to_csv(f)





# 9. Save models
joblib.dump(clf, os.path.join(MODELS_DIR,'logreg_tuned.joblib'))
joblib.dump(label_encoder, os.path.join(MODELS_DIR,'label_encoder.joblib'))
print("Saved tuned classifier and encoder.")

# Inference snippet
# from transformers import RobertaModel, RobertaTokenizer
# import torch, joblib
# roberta_inf = RobertaModel.from_pretrained('models/roberta_frozen')
# tokenizer_inf = RobertaTokenizer.from_pretrained('models/roberta_tokenizer')
# clf_inf = joblib.load('models/logreg_tuned.joblib')\# le_inf = joblib.load('models/label_encoder.joblib')
# inputs = tokenizer_inf(["example utterance"], padding='max_length', truncation=True, max_length=128, return_tensors='pt')\# emb = roberta_inf(**inputs).last_hidden_state[:,0,:].numpy()
# pred_id = clf_inf.predict(emb)[0]
# print(le_inf.inverse_transform([pred_id])[0] )
