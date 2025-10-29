#DECISION TREE DRAFT

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

MODEL_PATH = "spam_detector_model.joblib"

def load_data(path):
    df = pd.read_csv(path)
    df = df.fillna("")
    df["text"] = (
        df["sender"].astype(str) + " " +
        df["receiver"].astype(str) + " " +
        df["subject"].astype(str) + " " +
        df["body"].astype(str)
    )
    return df["text"], df["label"]

def train_model(data_path):
    print("Loading dataset...")
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=8000, stop_words="english")),
        ("svd", TruncatedSVD(n_components=120, random_state=42)),
        ("dt", DecisionTreeClassifier(max_depth=12,random_state=42))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("\nModel Performance:")
    print(classification_report(y_test, preds, digits=3))
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\n Model saved to: {MODEL_PATH}")

def predict_email(sender, receiver, subject, body):
    import os, joblib
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found! Train it first with train_model().")

    model = joblib.load(MODEL_PATH)
    text = f"{sender} {receiver} {subject} {body}"
    pred = model.predict([text])[0]
    prob = model.predict_proba([text])[0][1]
    return pred, prob


if __name__ == "__main__":
    print("Simple Spam Detector\n")
    choice = input("Train new model? (y/n): ").lower()
    if choice == "y":
        path = input("Enter CSV dataset path: ")
        train_model(path)
    else:
        sender = input("Sender: ")
        receiver = input("Recipient: ")
        subject = input("Subject: ")
        body = input("Body: ")
        predict_email(sender, receiver, subject, body)
