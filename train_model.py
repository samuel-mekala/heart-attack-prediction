import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def train_and_save_model():
    print("Loading dataset heart_new.csv...")
    df = pd.read_csv('heart_new.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    models = {
        'Random Forest Classifier': RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
        'Decision Tree Classifier': DecisionTreeClassifier(max_depth=6, criterion='entropy', random_state=42),
        'Extra Trees Classifier': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'AdaBoost Classifier': AdaBoostClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Support Vector Machine': SVC(probability=True, random_state=42)
    }

    # Try importing XGBoost safely
    try:
        from xgboost import XGBClassifier
        models['XGBoost Classifier'] = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='logloss')
    except Exception as e:
        print(f"Note: Skipping XGBoost benchmark due to environment missing libomp runtime: {e}")

    results = {}
    print("\n--- Model Evaluation ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0.0
        results[name] = (acc, auc, model)
        print(f"{name:25s} | Accuracy: {acc*100:6.2f}% | ROC-AUC: {auc:.4f}")

    # Random Forest is selected as specified in the report
    best_model_name = 'Random Forest Classifier'
    best_model = results[best_model_name][2]
    
    print(f"\nFinal Selected Model (as per report): {best_model_name}")
    y_pred_best = best_model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_best))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best))

    filename = 'savemodel.sav'
    pickle.dump(best_model, open(filename, 'wb'))
    print(f"Model successfully saved to '{filename}'.")

if __name__ == '__main__':
    train_and_save_model()
