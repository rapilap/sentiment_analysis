import pandas as pd
import numpy as np
import re
import nltk
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

MODEL_PATH = "model_nb.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

def train_and_save_model():
    print("Training model...")

    df = pd.read_csv('training.1600000.processed.noemoticon.csv',
                     encoding='latin-1',
                     header=None,
                     names=["target", "ids", "date", "flag", "user", "text"])
    df = df[['target', 'text']]
    df['target'] = df['target'].replace(4, 1)
    df['clean_text'] = df['text'].apply(clean_text)

    X = df['clean_text']
    y = df['target']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print("Model and vectorizer saved.")

# Load or train model
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
    train_and_save_model()

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    pred = model.predict(vector)
    return "Positive" if pred[0] == 1 else "Negative"
