import os
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


def load_data(benign_path, malicious_path):
    """Load benign and malicious sample data and assign labels."""
    texts = []
    labels = []

    if os.path.exists(benign_path):
        with open(benign_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(0)
    else:
        print(f"Benign sample file not found: {benign_path}")

    if os.path.exists(malicious_path):
        with open(malicious_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    labels.append(1)
    else:
        print(f"Malicious sample file not found: {malicious_path}")

    return texts, labels


def preprocess_data(texts):
    """Vectorize text data using a bag-of-words model with N-gram features."""
    vectorizer = CountVectorizer(
        ngram_range=(1, 2),
        max_features=1000,
        stop_words='english'
    )
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def train_and_evaluate(X, y, n_splits=10):
    """Perform stratified K-fold cross-validation to train and evaluate the model."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    reports = []

    X_array = X
    y_array = np.array(y)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_array), 1):
        print(f"\nTraining fold {fold}...")
        X_train, X_test = X_array[train_idx], X_array[test_idx]
        y_train, y_test = y_array[train_idx], y_array[test_idx]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        report = classification_report(y_test, y_pred, output_dict=True)
        reports.append(report)

        print(f"Fold {fold} accuracy: {acc:.4f}")

    avg_accuracy = np.mean(accuracies)
    print(f"\n=== Cross-Validation Summary ===")
    print(f"Mean accuracy: {avg_accuracy:.4f}")

    return avg_accuracy


def save_model(model, vectorizer, model_path, vectorizer_path):
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Random Forest N-gram classifier for sequence judgment")
    parser.add_argument('--benign', type=str, required=True, help="Path to benign samples file (one sequence per line)")
    parser.add_argument('--malicious', type=str, required=True, help="Path to malicious samples file (one sequence per line)")
    args = parser.parse_args()

    texts, labels = load_data(args.benign, args.malicious)
    if not texts:
        print("No data loaded, please check file paths.")
        return

    print("Preprocessing data (N-gram vectorization)...")
    X, vectorizer = preprocess_data(texts)

    train_and_evaluate(X, labels)

    print("\nTraining final model on entire dataset...")
    final_model = RandomForestClassifier(n_estimators=100, random_state=42)
    final_model.fit(X, labels)

    save_model(final_model, vectorizer, "rf_ngram_model.pkl", "ngram_vectorizer.pkl")


if __name__ == "__main__":
    main()
