"""
Phase 7 - Step 2: Train the Gesture Classifier
------------------------------------------------
Goal: Train a K-Nearest Neighbors (KNN) classifier on the collected
gesture data, evaluate its accuracy, and save the trained model to disk.

Requires: data/gesture_data.csv (created by collect_gesture_data.py)
Produces: assets/models/gesture_classifier.pkl
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

DATA_PATH = "data/gesture_data.csv"
MODEL_OUTPUT_PATH = "assets/models/gesture_classifier.pkl"


def main():
    # --- Load the collected data ---
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} total samples.")
    print("Samples per gesture:")
    print(df["label"].value_counts())
    print()

    # Separate features (the 63 landmark numbers) from labels (chord names)
    feature_columns = [col for col in df.columns if col != "label"]
    X = df[feature_columns]
    y = df["label"]

    # --- Split into training and testing sets ---
    # We hold out 20% of the data purely to evaluate accuracy afterward,
    # so we're testing on samples the model has never seen during training.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Train the KNN classifier ---
    # n_neighbors=5 means: for a new gesture, look at the 5 most similar
    # training samples and take a majority vote on the label.
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    # --- Evaluate on the held-out test set ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {accuracy * 100:.2f}%\n")
    print("Detailed report:")
    print(classification_report(y_test, y_pred))

    # --- Save the trained model to disk ---
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()