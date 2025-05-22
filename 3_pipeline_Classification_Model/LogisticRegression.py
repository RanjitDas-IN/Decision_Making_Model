import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load cleaned dataset
csv_path = r"/home/ranjit/Desktop/Decision_Making_Model/Day2_Tokenized_dataset.csv"
df = pd.read_csv(csv_path, dtype=str)

# Drop rows with missing entries in key columns
df = df.dropna(subset=["utterance", "intent", "tokens"]).reset_index(drop=True)

# Helper to convert token list strings to whitespace-joined text
def join_tokens(token_list_str):
    try:
        tokens = eval(token_list_str)
        return " ".join(tokens)
    except Exception:
        return ""

# Prepare features and labels
X = df['tokens'].apply(join_tokens)
y = df['intent']

# Vectorize with TF-IDF including unigrams and bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_vect = vectorizer.fit_transform(X)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_vect, y, test_size=0.2, random_state=42
)

# Initialize and train Logistic Regression with balanced class weights
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optionally save the trained model and vectorizer
# joblib.dump(clf, 'intent_clf_lr_balanced.pkl')
# joblib.dump(vectorizer, 'tfidf_vectorizer_ngrams.pkl')
