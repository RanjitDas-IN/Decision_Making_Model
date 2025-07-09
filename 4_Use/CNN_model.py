import tensorflow as tf
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import torch

# --- Step 1: Load Trained Model ---
model = tf.keras.models.load_model("cnn_intent_classifier.h5")
print(model.summary())

# --- Step 2: Load RoBERTa (same as training) ---
device = torch.device("cpu")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta = RobertaModel.from_pretrained("roberta-base").to(device)
roberta.eval()

# --- Step 3: Define Inference Function ---
def encode_utterances(utterances):
    inputs = tokenizer(
        utterances,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = roberta(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
    return cls_embeddings.cpu().numpy()

# --- Step 4: Prepare Test Inputs ---
raw_inputs = [
    "set a reminder for tomorrow at 8",
    "play some music",
    "who is the president of India",
    "show me some cat images",
    "Show the times for Cheers for Miss Bishop at Dipson Theatres.",
    "I want to see Married to the Enemy 2 at a cinema.",
    "hey find sqrt of 64",
    "could u add seventeen and thirteen",
    "i’m curious what’s 18 less 9",
    "That essay bored me. Just a one.",
    "Give this literary saga 4 out of 6.",
    "I’m thinking this music album gets 2 points.",
    "The current short story earns a perfect 6.",
    "This new novel is a strong 5.",
    "You've got 90 seconds",
    "Agent Coulson?",
    "just wanted to say thank you very much for all of your help",
    "can u find me machine learning tutorials?",
    "how to clean bathroom vent",
    "how to use a mop",
    "how to do oil change",
    "what time is it",
    "google how to start a blog",
    "how to delete google search history",
    "google benefits of regular exercise",
    "how to set up google alerts",
    "I’d love a drawing of a peaceful beach at sunset.",
    "Can produce an image of a bustling marketplace in Morocco?",
    "Design a poster showcasing a jazz band performing live.",
    "shuffle the hits from the 80s",
    "play the experimental music",
    "play ’Besos’by Shikhar Dhawan and Jacqueline Fernandez",
    "start ’Snake’by Nora Fatehi and Jason Derulo",
    "queue ’Run It Up’by Hanumankind",
    "play ’Kanimaa’by Santhosh Narayanan and The Indin Choral Ensembale",
    "shuffle ’Tu Hain Toh Main Hoon’from Sky Force",
    "Power cycle the router and then reboot",
    "Schedule a system restart for midnight",
    "Start system updates and reboot after",
    "Can power down the device safely?",
    "Run antivirus and then restart system",
    "Shut down after backup completes",
    "Reboot my phone and clear cache",
    "Put the device into airplane mode and reboot",
]

# Convert to embeddings
X_test = encode_utterances(raw_inputs)
X_test = tf.reshape(X_test, (-1, 768, 1))

# --- Step 5: Predict Intents ---
pred_probs = model.predict(X_test)
pred_indices = np.argmax(pred_probs, axis=1)

# Optional: load your label encoder to get back the intent name
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Refit LabelEncoder on same intent set (you can also pickle+load it)
df = pd.read_csv("1_Pipeline_Data_Acquisition/Day2_cleaned_dataset.csv", sep='|')
le = LabelEncoder()
le.fit(df['intent'].values)

# Print results
for i, utter in enumerate(raw_inputs):
    predicted_intent = le.inverse_transform([pred_indices[i]])[0]
    confidence = pred_probs[i][pred_indices[i]]
    print(f"🗣️ \"{utter}\" → 🎯 {predicted_intent} ({confidence:.2f})")
