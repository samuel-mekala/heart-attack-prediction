import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    dataset_path = 'heart_new.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    print(f"Loading dataset {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Dataset shape: {df.shape}")

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    print("\n--- Training Random Forest Classifier on Clinical Heart Dataset ---")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 0] # prob of disease (class 0)
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, 1 - y_prob)
    
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"ROC AUC: {auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    filename = 'savemodel.sav'
    pickle.dump(model, open(filename, 'wb'))
    print(f"\nModel successfully saved to '{filename}'.")

if __name__ == '__main__':
    train_and_save_model()
