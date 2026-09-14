import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

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

    print("\n--- Training Random Forest Classifier ---")
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42)
    rf.fit(X_train, y_train)

    print("\n--- Training Gradient Boosting Classifier ---")
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    gb.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_pred_gb = gb.predict(X_test)
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_gb = accuracy_score(y_test, y_pred_gb)
    
    print(f"Random Forest Test Accuracy: {acc_rf*100:.2f}%")
    print(f"Gradient Boosting Test Accuracy: {acc_gb*100:.2f}%")

    model_bundle = {
        'rf': rf,
        'gb': gb,
        'features': list(X.columns)
    }

    filename = 'savemodel.sav'
    pickle.dump(model_bundle, open(filename, 'wb'))
    print(f"\nModel Bundle successfully saved to '{filename}'.")

if __name__ == '__main__':
    train_and_save_model()
