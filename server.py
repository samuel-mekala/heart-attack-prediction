from flask import Flask, request, jsonify, render_template, redirect
import os
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ─── Load Dataset ─────────────────────────────────────────────────────────────
# Place heart_attack_prediction_dataset.csv in the same folder as this file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'heart_attack_prediction_dataset.csv')

df = pd.read_csv(DATA_PATH)

df = df.drop('Patient ID', axis=1)

# Split Blood Pressure into systolic / diastolic
df['BP_systolic']  = df['Blood Pressure'].apply(lambda x: x.split("/")[0])
df['BP_diastolic'] = df['Blood Pressure'].apply(lambda x: x.split("/")[1])

df = df[['Age', 'Sex', 'Cholesterol',
         'BP_systolic', 'BP_diastolic',
         'Heart Rate', 'Diabetes',
         'Family History', 'Smoking', 'Obesity', 'Alcohol Consumption',
         'Exercise Hours Per Week', 'Diet', 'Previous Heart Problems',
         'Medication Use', 'Stress Level', 'Sedentary Hours Per Day', 'Income',
         'BMI', 'Triglycerides', 'Physical Activity Days Per Week',
         'Sleep Hours Per Day', 'Country', 'Continent', 'Hemisphere',
         'Heart Attack Risk']]

df['BP_systolic']  = pd.to_numeric(df['BP_systolic'])
df['BP_diastolic'] = pd.to_numeric(df['BP_diastolic'])

df2 = df[['Age', 'Sex', 'Cholesterol', 'BP_systolic', 'BP_diastolic',
          'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
          'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
          'Previous Heart Problems', 'Medication Use', 'Stress Level',
          'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
          'Physical Activity Days Per Week', 'Sleep Hours Per Day',
          'Heart Attack Risk']].copy()

# Encode categorical columns
df3 = df2[['Sex', 'Diet']].copy()
le = LabelEncoder()
label_encoder = {}
for column in df3.columns:
    label_encoder[column] = le
    df3[column] = label_encoder[column].fit_transform(df2[column])

df2 = df2.drop(['Sex', 'Diet', 'Income'], axis=1)
df2 = pd.concat([df2, df3], axis=1)

# ─── Features Used for Training ───────────────────────────────────────────────
FEATURES = ['Age', 'Cholesterol', 'BP_systolic', 'BP_diastolic', 'Heart Rate',
            'Diabetes', 'Family History', 'Smoking', 'Obesity',
            'Alcohol Consumption', 'Exercise Hours Per Week',
            'Previous Heart Problems', 'Medication Use',
            'BMI', 'Triglycerides', 'Sleep Hours Per Day',
            'Sex', 'Diet']

X = df2[FEATURES]
y = df2[['Heart Attack Risk']]

smote = SMOTE(random_state=50)
X_resample, y_resample = smote.fit_resample(X, y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_resample)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resample, test_size=0.2, random_state=42
)

y_train = np.ravel(y_train)
y_test  = np.ravel(y_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(f"Model trained. Test Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def dicti_vals(dicti):
    x = list(dicti.values())
    return np.array([x])


def determine_lifestyle_changes(predict_prob, new_person):
    lifestyle_changes = []

    if predict_prob > 0:
        if new_person.get('Smoking') == 1:
            lifestyle_changes.append('Quit smoking')
        bmi = new_person.get('BMI', 22)
        if bmi < 18.5:
            lifestyle_changes.append('Gain weight to reach a healthy BMI')
        elif bmi > 25:
            lifestyle_changes.append('Lose weight to reach a healthy BMI')
        if new_person.get('Exercise Hours Per Week', 5) < 1.25:
            lifestyle_changes.append('Do more exercise (aim for 150 min/week)')
        if new_person.get('Diet') in [0, 2]:   # Average or Unhealthy
            lifestyle_changes.append('Eat healthier food')
        if new_person.get('Alcohol Consumption') == 1:
            lifestyle_changes.append('Try reducing alcohol consumption')
        if new_person.get('Sleep Hours Per Day', 7) < 6:
            lifestyle_changes.append('Get 7–9 hours of sleep per night')
        if new_person.get('Stress Level', 5) > 7:
            lifestyle_changes.append('Practice stress-reduction techniques')

    return {
        'Heart_attack_risk': round(float(predict_prob), 4),
        'Lifestyle_changes': lifestyle_changes
    }

# ─── Global result store (session-level) ──────────────────────────────────────
data = {'Heart_attack_risk': 0, 'Lifestyle_changes': []}

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('test.html')


@app.route('/result', methods=['GET', 'POST'])
def resultPage():
    return render_template('result_template.html', data=data)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read form values — names match the HTML form exactly
        new_person = {
            'Age':                        int(request.form.get('Age', 0)),
            'Sex':                        int(request.form.get('sex', 1)),
            'BP_systolic':                int(request.form.get('BP_systolic', 120)),
            'BP_diastolic':               int(request.form.get('BP_diastolic', 80)),
            'Cholesterol':              float(request.form.get('Cholesterol', 200)),
            'Triglycerides':            float(request.form.get('Triglycerides', 150)),
            'Heart Rate':                 int(request.form.get('Heart_Rate', 72)),
            'Diabetes':                   int(request.form.get('Diabetes', 0)),
            'Family History':             int(request.form.get('Family_History', 0)),
            'Smoking':                    int(request.form.get('Smoking', 0)),
            'Obesity':                    int(request.form.get('Obesity', 0)),
            'Alcohol Consumption':        int(request.form.get('Alcohol_Consumption', 0)),
            'Exercise Hours Per Week':    int(request.form.get('Exercise_Hours_Per_Week', 3)),
            'Diet':                       int(request.form.get('Diet', 1)),
            'Previous Heart Problems':    int(request.form.get('Previous_Heart_Problems', 0)),
            'Medication Use':             int(request.form.get('Medication_Use', 0)),
            'BMI':                      float(request.form.get('BMI', 22)),
            'Sleep Hours Per Day':        int(request.form.get('Sleep_Hours_Per_Day', 7)),
        }

        x = dicti_vals(new_person)
        x_scaled = scaler.transform(x)

        predict_prob = model.predict_proba(x_scaled)[:, 1][0]
        result = determine_lifestyle_changes(predict_prob, new_person)

        global data
        data = result
        print(result)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
