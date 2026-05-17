# Updated server.py (Fully Fixed Version)

```python
from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'fallback_secret')

# =========================
# LOAD DATA
# =========================

file_path = 'heart_attack_prediction_dataset.csv'
df = pd.read_csv(file_path)

# =========================
# CLEAN DATA
# =========================

# Split blood pressure

df[['BP_systolic', 'BP_diastolic']] = (
    df['Blood Pressure']
    .str.split('/', expand=True)
    .astype(int)
)

# Drop unnecessary columns

df.drop(columns=[
    'Patient ID',
    'Blood Pressure',
    'Country',
    'Continent',
    'Hemisphere'
], inplace=True)

# =========================
# ENCODE CATEGORICAL DATA
# =========================

categorical_cols = [
    'Sex',
    'Diet',
    'Diabetes',
    'Family History',
    'Smoking',
    'Obesity',
    'Alcohol Consumption',
    'Previous Heart Problems',
    'Medication Use'
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# =========================
# FEATURE ORDER (VERY IMPORTANT)
# =========================

FEATURE_COLUMNS = [
    'Age',
    'Sex',
    'Cholesterol',
    'Heart Rate',
    'Diabetes',
    'Family History',
    'Smoking',
    'Obesity',
    'Alcohol Consumption',
    'Exercise Hours Per Week',
    'Diet',
    'Previous Heart Problems',
    'Medication Use',
    'Stress Level',
    'BMI',
    'Triglycerides',
    'Physical Activity Days Per Week',
    'Sleep Hours Per Day',
    'BP_systolic',
    'BP_diastolic'
]

X = df[FEATURE_COLUMNS]
y = df['Heart Attack Risk']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# APPLY SMOTE ONLY ON TRAIN
# =========================

smote = SMOTE(random_state=50)
X_train, y_train = smote.fit_resample(X_train, y_train)

# =========================
# MODEL
# =========================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=17,
    n_jobs=-1
)

model.fit(X_train, y_train)

# =========================
# ACCURACY
# =========================

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print(f'Accuracy: {accuracy * 100:.2f}%')

# =========================
# RECOMMENDATION FUNCTION
# =========================

def determine_lifestyle_changes(data, risk_prob):
    changes = []

    if data['Smoking'] == 1:
        changes.append('Quit smoking')

    if data['Alcohol Consumption'] == 1:
        changes.append('Reduce alcohol consumption')

    if data['Exercise Hours Per Week'] < 3:
        changes.append('Increase exercise hours')

    if data['BMI'] > 25:
        changes.append('Maintain healthy body weight')

    if data['Sleep Hours Per Day'] < 7:
        changes.append('Improve sleep schedule')

    if data['Stress Level'] > 6:
        changes.append('Reduce stress levels')

    if risk_prob < 0.35:
        risk_level = 'Low Risk'
    elif risk_prob < 0.65:
        risk_level = 'Moderate Risk'
    else:
        risk_level = 'High Risk'

    return {
        'Heart_attack_risk': round(float(risk_prob * 100), 2),
        'Risk_Level': risk_level,
        'Lifestyle_changes': changes
    }

# =========================
# ROUTES
# =========================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        # =========================
        # VALIDATION
        # =========================

        required_fields = [
            'Age',
            'Sex',
            'Cholesterol',
            'Heart_Rate'
        ]

        for field in required_fields:
            if not request.form.get(field):
                return jsonify({'error': f'{field} is required'}), 400

        # =========================
        # INPUT DATA
        # =========================

        new_person = {
            'Age': int(request.form.get('Age')),
            'Sex': label_encoders['Sex'].transform([
                request.form.get('Sex')
            ])[0],
            'Cholesterol': int(request.form.get('Cholesterol')),
            'Heart Rate': int(request.form.get('Heart_Rate')),
            'Diabetes': label_encoders['Diabetes'].transform([
                request.form.get('Diabetes')
            ])[0],
            'Family History': label_encoders['Family History'].transform([
                request.form.get('Family_History')
            ])[0],
            'Smoking': label_encoders['Smoking'].transform([
                request.form.get('Smoking')
            ])[0],
            'Obesity': label_encoders['Obesity'].transform([
                request.form.get('Obesity')
            ])[0],
            'Alcohol Consumption': label_encoders['Alcohol Consumption'].transform([
                request.form.get('Alcohol_Consumption')
            ])[0],
            'Exercise Hours Per Week': float(request.form.get('Exercise_Hours_Per_Week')),
            'Diet': label_encoders['Diet'].transform([
                request.form.get('Diet')
            ])[0],
            'Previous Heart Problems': label_encoders['Previous Heart Problems'].transform([
                request.form.get('Previous_Heart_Problems')
            ])[0],
            'Medication Use': label_encoders['Medication Use'].transform([
                request.form.get('Medication_Use')
            ])[0],
            'Stress Level': int(request.form.get('Stress_Level')),
            'BMI': float(request.form.get('BMI')),
            'Triglycerides': int(request.form.get('Triglycerides')),
            'Physical Activity Days Per Week': int(request.form.get('Physical_Activity_Days_Per_Week')),
            'Sleep Hours Per Day': float(request.form.get('Sleep_Hours_Per_Day')),
            'BP_systolic': int(request.form.get('BP_systolic')),
            'BP_diastolic': int(request.form.get('BP_diastolic'))
        }

        # =========================
        # FIXED FEATURE ORDER
        # =========================

        x = np.array([[new_person[col] for col in FEATURE_COLUMNS]])

        print('FEATURE ORDER:', FEATURE_COLUMNS)
        print('INPUT VALUES:', x)

        # =========================
        # PREDICTION
        # =========================

        risk_prob = float(model.predict_proba(x)[:, 1][0])

        result = determine_lifestyle_changes(new_person, risk_prob)

        session['result'] = result

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/result')
def result():

    result_data = session.get('result', None)

    if not result_data:
        return render_template('result.html', error='No prediction found')

    return render_template(
        'result.html',
        risk=result_data['Heart_attack_risk'],
        risk_level=result_data['Risk_Level'],
        lifestyle_changes=result_data['Lifestyle_changes'],
        accuracy=round(accuracy * 100, 2)
    )

# =========================
# MAIN
# =========================

if __name__ == '__main__':
    app.run(debug=True)
```

# IMPORTANT NOTES

You MUST also ensure your HTML form field names match exactly:

Examples:

```html
name="Heart_Rate"
name="Family_History"
name="Alcohol_Consumption"
```

because Flask reads using:

```python
request.form.get('Heart_Rate')
```

If names mismatch even slightly:

* wrong values
* empty values
* incorrect predictions

can happen.
