import torch
import joblib
from transformers import RobertaTokenizer, RobertaModel
import numpy as np


MODEL_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/SVM_models/svm_rbf_tuned.joblib"
SCALER_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/SVM_models/scaler.joblib"
LABEL_ENCODER_PATH = r"/home/ranjit/Desktop/Decision_Making_Model/SVM_models/label_encoder.joblib"  

device = torch.device("cpu")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta = RobertaModel.from_pretrained('roberta-base').to(device)
roberta.eval()

def get_cls_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = roberta(**inputs)
        cls_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding

def predict_intent(user_input):
    embedding = get_cls_embedding(user_input)
    embedding_scaled = scaler.transform(embedding)
    pred = model.predict(embedding_scaled)
    intent = label_encoder.inverse_transform(pred)[0]
    return intent


n_support_vectors = model.support_vectors_.shape[0]

feature_dim = model.support_vectors_.shape[1]

dual_coef_params = model.dual_coef_.size
# print(model.dual_coef_,model.support_vectors_)
bias_params = model.intercept_.size

total_parameters = (n_support_vectors * feature_dim) + dual_coef_params + bias_params
# print(f"Support Vectors: {n_support_vectors}")
# print(f"Feature Dimension: {feature_dim}")
# print(f"Dual Coefficients: {dual_coef_params}")
# print(f"Bias Terms: {bias_params}")holi
print(f"Total Parameters: {total_parameters}")



# # --- Main loop ---
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    elif user_input == '    ' or user_input == '' or user_input==' ':
        print("BC, Enter a valid word")
    else:
        predicted_intent = predict_intent(user_input)
        print(f"Predicted Intent: {predicted_intent}")