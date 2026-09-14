import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

def train_and_save_model():
    dataset_path = 'heart_attack_prediction_dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    print(f"Loading dataset {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"Raw dataset shape: {df.shape}")

    # Process Blood Pressure into Systolic and Diastolic
    if 'Blood Pressure' in df.columns:
        bp_split = df['Blood Pressure'].astype(str).str.split('/', expand=True)
        df['BP_Systolic'] = pd.to_numeric(bp_split[0], errors='coerce').fillna(120)
        df['BP_Diastolic'] = pd.to_numeric(bp_split[1], errors='coerce').fillna(80)
    elif 'BP_Systolic' not in df.columns:
        df['BP_Systolic'] = 120
        df['BP_Diastolic'] = 80

    # Categorical Mappings matching 18 form inputs
    df['Sex_Code'] = df['Sex'].map({'Male': 1, 'Female': 0, '1': 1, '0': 0}).fillna(0)
    
    alcohol_map = {'None': 0, 'No': 0, 'Light': 1, 'Moderate': 2, 'Heavy': 3, 'Yes': 1, 0: 0, 1: 1}
    df['Alcohol_Code'] = df['Alcohol Consumption'].map(alcohol_map).fillna(0)

    diet_map = {'Unhealthy': 0, 'Average': 1, 'Healthy': 2, 0: 0, 1: 1, 2: 2}
    df['Diet_Code'] = df['Diet'].map(diet_map).fillna(1)

    # 18 exact features requested from the report form
    feature_cols = [
        'Age', 'Sex_Code', 'BP_Systolic', 'BP_Diastolic', 'Cholesterol', 'Triglycerides',
        'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol_Code',
        'Medication Use', 'Diet_Code', 'Previous Heart Problems', 'Sleep Hours Per Day',
        'BMI', 'Exercise Hours Per Week'
    ]

    target_col = 'Heart Attack Risk' if 'Heart Attack Risk' in df.columns else 'Heart_Attack_Risk'

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Fill any missing values safely
    X = X.fillna(X.median())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    print("\n--- Training Random Forest Classifier on 18 Features ---")
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    filename = 'savemodel.sav'
    pickle.dump(model, open(filename, 'wb'))
    print(f"\nModel successfully saved to '{filename}'.")

if __name__ == '__main__':
    train_and_save_model()
