from flask import Flask, request, jsonify, render_template, redirect, session
import joblib
import logging
import os
import json
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
app.secret_key = os.getenv(
    "SECRET_KEY",
    "temporary-dev-secret"
)
# ─── Load Dataset ─────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'heart_attack_prediction_dataset.csv')

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
print(f"Model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def dicti_vals(dicti):
    return np.array([list(dicti.values())])

def determine_lifestyle_changes(risk_prob, new_person):
    changes = []
    if new_person.get('Smoking') == 1:
        changes.append('Quit smoking')
    bmi = new_person.get('BMI', 22)
    if bmi < 18.5:
        changes.append('Gain weight to reach a healthy BMI')
    elif bmi > 25:
        changes.append('Lose weight to reach a healthy BMI')
    if new_person.get('Exercise_Hours_Per_Week', 5) < 1.25:
        changes.append('Do more exercise (aim for 150 min/week)')
    if new_person.get('Diet') in [0, 2]:
        changes.append('Eat healthier food')
    if new_person.get('Alcohol_Consumption') == 1:
        changes.append('Try reducing alcohol consumption')
    return {
        'Heart_attack_risk': round(float(risk_prob), 4),
        'Lifestyle_changes': changes
    }

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('test.html')

@app.route('/result')
def resultPage():
    # FIX: Read result from session, not global variable
    result_json = session.get('result', None)
    if result_json:
        data = json.loads(result_json)
    else:
        data = {'Heart_attack_risk': 0, 'Lifestyle_changes': []}
    return render_template('result_template.html', data=data)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        new_person = {
            'Age':                      int(request.form.get('Age')),
            'Cholesterol':            float(request.form.get('Cholesterol')),
            'BP_systolic':              int(request.form.get('BP_systolic')),
            'BP_diastolic':             int(request.form.get('BP_diastolic')),
            'Heart_Rate':               int(request.form.get('Heart_Rate')),
            'Diabetes':                 int(request.form.get('Diabetes')),
            'Family_History':           int(request.form.get('Family_History')),
            'Smoking':                  int(request.form.get('Smoking')),
            'Obesity':                  int(request.form.get('Obesity')),
            'Alcohol_Consumption':      int(request.form.get('Alcohol_Consumption')),
            'Exercise_Hours_Per_Week':  float(request.form.get('Exercise_Hours_Per_Week')),
            'Previous_Heart_Problems':  int(request.form.get('Previous_Heart_Problems')),
            'Medication_Use':           int(request.form.get('Medication_Use')),
            'BMI':                    float(request.form.get('BMI')),
            'Triglycerides':          float(request.form.get('Triglycerides')),
            'Sleep_Hours_Per_Day':      int(request.form.get('Sleep_Hours_Per_Day')),
            'Sex':                      int(request.form.get('sex')),
            'Diet':                     int(request.form.get('Diet')),
        }

        x            = dicti_vals(new_person)
        x_scaled     = scaler.transform(x)
        risk_prob    = float(model.predict_proba(x_scaled)[:, 1][0])
        result       = determine_lifestyle_changes(risk_prob, new_person)

        # FIX: Store in session instead of global variable
        session['result'] = json.dumps(result)

        print(f"Prediction: {risk_prob:.4f} | Changes: {result['Lifestyle_changes']}")
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
