# hf_semantic_cluster_trainer.py

import os
import pandas as pd
import logging
import numpy as np
from tokenizers import Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# === CONFIGURATION ===
DATA_PATH = "/home/ranjit/Desktop/Decision_Making_Model/1_Pipeline_Intent_Data_Acquisition/Day2_cleaned_dataset.csv"
TOKENIZER_PATH = "/home/ranjit/Desktop/Decision_Making_Model/hf_tokenizer.json"
SEP = '|'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# === LOGGING SETUP ===
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# === LOAD DATA ===
logging.info("Loading dataset...")
df = pd.read_csv(DATA_PATH, sep=SEP, names=["utterance", "intent"], header=None, dtype=str)
df.dropna(subset=["utterance", "intent"], inplace=True)
df.reset_index(drop=True, inplace=True)

# === LOAD TOKENIZER ===
logging.info("Loading Hugging Face tokenizer...")
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# === TOKENIZE TEXT ===
logging.info("Tokenizing utterances...")
tokenized_texts = [" ".join(tokenizer.encode(text).tokens) for text in df["utterance"]]

# === FEATURE VECTORIZATION ===
logging.info("Vectorizing with TF-IDF...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokenized_texts)

# === LABEL ENCODING ===
logging.info("Encoding labels...")
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["intent"])

# === SPLIT DATA ===
logging.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# === TRAIN LOGISTIC REGRESSION ===
logging.info("Training Logistic Regression model...")
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# === EVALUATE ===
logging.info("Evaluating model...")
y_pred = clf.predict(X_test)

# Resolve mismatch between classes and target_names if some classes absent in y_test
y_unique = np.unique(y_test)
classes_in_test = label_encoder.classes_[y_unique]
report = classification_report(
    y_test,
    y_pred,
    labels=y_unique,
    target_names=classes_in_test
)
print(report)

logging.info("Training and evaluation complete.")
