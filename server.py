from flask import Flask, request, jsonify, render_template, redirect, session
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
app.secret_key = 'heart_attack_predictor_2024'

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'heart_attack_prediction_dataset.csv')

df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().replace(' ', '_') for c in df.columns]
df = df.drop('Patient_ID', axis=1)

df['BP_systolic']  = df['Blood_Pressure'].apply(lambda x: x.split("/")[0])
df['BP_diastolic'] = df['Blood_Pressure'].apply(lambda x: x.split("/")[1])

df = df[['Age', 'Sex', 'Cholesterol', 'BP_systolic', 'BP_diastolic',
         'Heart_Rate', 'Diabetes', 'Family_History', 'Smoking', 'Obesity',
         'Alcohol_Consumption', 'Exercise_Hours_Per_Week', 'Diet',
         'Previous_Heart_Problems', 'Medication_Use', 'Stress_Level',
         'Sedentary_Hours_Per_Day', 'Income', 'BMI', 'Triglycerides',
         'Physical_Activity_Days_Per_Week', 'Sleep_Hours_Per_Day',
         'Country', 'Continent', 'Hemisphere', 'Heart_Attack_Risk']]

df['BP_systolic']  = pd.to_numeric(df['BP_systolic'])
df['BP_diastolic'] = pd.to_numeric(df['BP_diastolic'])

df2 = df[['Age', 'Sex', 'Cholesterol', 'BP_systolic', 'BP_diastolic',
          'Heart_Rate', 'Diabetes', 'Family_History', 'Smoking', 'Obesity',
          'Alcohol_Consumption', 'Exercise_Hours_Per_Week', 'Diet',
          'Previous_Heart_Problems', 'Medication_Use', 'Stress_Level',
          'Sedentary_Hours_Per_Day', 'Income', 'BMI', 'Triglycerides',
          'Physical_Activity_Days_Per_Week', 'Sleep_Hours_Per_Day',
          'Heart_Attack_Risk']].copy()

df3 = df2[['Sex', 'Diet']].copy()
le  = LabelEncoder()
label_encoder = {}
for column in df3.columns:
    label_encoder[column] = le
    df3[column] = label_encoder[column].fit_transform(df2[column])

df2 = df2.drop(['Sex', 'Diet', 'Income'], axis=1)
df2 = pd.concat([df2, df3], axis=1)

X = df2[['Age', 'Cholesterol', 'BP_systolic', 'BP_diastolic', 'Heart_Rate',
         'Diabetes', 'Family_History', 'Smoking', 'Obesity',
         'Alcohol_Consumption', 'Exercise_Hours_Per_Week',
         'Previous_Heart_Problems', 'Medication_Use',
         'BMI', 'Triglycerides', 'Sleep_Hours_Per_Day', 'Sex', 'Diet']]
y = df2[['Heart_Attack_Risk']]

smote = SMOTE(random_state=50)
X_res, y_res = smote.fit_resample(X, y)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, random_state=42)

y_train = np.ravel(y_train)
y_test  = np.ravel(y_test)

model = RandomForestClassifier(random_state=17)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
print(f"Model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

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

@app.route('/')
def index():
    return render_template('test.html')

@app.route('/result')
def resultPage():
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
            'Exercise_Hours_Per_Week':  int(request.form.get('Exercise_Hours_Per_Week')),
            'Previous_Heart_Problems':  int(request.form.get('Previous_Heart_Problems')),
            'Medication_Use':           int(request.form.get('Medication_Use')),
            'BMI':                    float(request.form.get('BMI')),
            'Triglycerides':          float(request.form.get('Triglycerides')),
            'Sleep_Hours_Per_Day':      int(request.form.get('Sleep_Hours_Per_Day')),
            'Sex':                      int(request.form.get('sex')),
            'Diet':                     int(request.form.get('Diet')),
        }
        x         = dicti_vals(new_person)
        x_scaled  = scaler.transform(x)
        risk_prob = float(model.predict_proba(x_scaled)[:, 1][0])
        result    = determine_lifestyle_changes(risk_prob, new_person)
        session['result'] = json.dumps(result)
        print(f"Prediction: {risk_prob:.4f}")
        return jsonify(result)
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
