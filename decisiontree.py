#DECISION TREE DRAFT

import pandas as pd
import numpy as np

# Basic Preprocessing
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

# word frequency of spam indicators
SPAM_WORDS = ["free", "win", "click", "buy", "offer", "money", "urgent", "limited", "discount"]

def text_to_features(text):
    text = text.lower()
    return np.array([text.count(word) for word in SPAM_WORDS])

# Helper Functions for Tree Building
def gini_impurity(y):
    labels, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2)

def information_gain(y, left_idx, right_idx):
    if len(left_idx) == 0 or len(right_idx) == 0:
        return 0
    left_impurity = gini_impurity(y[left_idx])
    right_impurity = gini_impurity(y[right_idx])
    p = len(left_idx) / len(y)
    return gini_impurity(y) - (p * left_impurity + (1 - p) * right_impurity)

def best_split(X, y):
    best_gain = 0
    best_feat, best_thresh = None, None
    n_samples, n_features = X.shape

    for feat in range(n_features):
        thresholds = np.unique(X[:, feat])
        for thresh in thresholds:
            left_idx = np.where(X[:, feat] <= thresh)[0]
            right_idx = np.where(X[:, feat] > thresh)[0]
            gain = information_gain(y, left_idx, right_idx)
            if gain > best_gain:
                best_gain = gain
                best_feat, best_thresh = feat, thresh
    return best_feat, best_thresh, best_gain

# Decision Tree Node
class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# Decision Tree Classifier
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        feat, thresh, gain = best_split(X, y)
        if gain == 0 or feat is None:
            leaf_value = self._most_common_label(y)
            return DecisionNode(value=leaf_value)

        left_idx = np.where(X[:, feat] <= thresh)[0]
        right_idx = np.where(X[:, feat] > thresh)[0]

        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        return DecisionNode(feature=feat, threshold=thresh, left=left, right=right)

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict_one(self, x):
        node = self.root
        while node.value is None:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

# Main function
def main():
    print("Simple Spam Detector\n")
    choice = input("Train new model? (y/n): ").lower()

    if choice == "y":
        path = input("Enter CSV dataset path: ")
        X_text, y = load_data(path)
        X = np.array([text_to_features(t) for t in X_text])

        tree = DecisionTree(max_depth=5)
        tree.fit(X, y)

        preds = tree.predict(X)
        acc = np.mean(preds == y)
        print(f"Training complete. Accuracy: {acc:.3f}")

        # Save trained model
        import pickle
        with open("decision_tree_model.pkl", "wb") as f:
            pickle.dump(tree, f)
        print("Model saved as decision_tree_model.pkl")

    else:
        import pickle, os
        if not os.path.exists("decision_tree_model.pkl"):
            print("No trained model found. Train it first.")
            return

        with open("decision_tree_model.pkl", "rb") as f:
            tree = pickle.load(f)

        sender = input("Sender: ")
        receiver = input("Receiver: ")
        subject = input("Subject: ")
        body = input("Body: ")

        text = f"{sender} {receiver} {subject} {body}"
        x_test = text_to_features(text).reshape(1, -1)
        pred = tree.predict(x_test)[0]
        print("Prediction:", "SPAM" if pred == 1 else "HAM")

if __name__ == "__main__":
    main()
