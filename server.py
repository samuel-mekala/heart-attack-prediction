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
# FIX 1: Removed hardcoded Windows path D:\\IntrainTech\\...
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'heart_attack_prediction_dataset.csv')

df = pd.read_csv(DATA_PATH)

# FIX 2: CSV has spaces in column names e.g. "Blood Pressure", "Heart Rate"
# Rename all columns to use underscores for consistency
df.columns = [c.strip().replace(' ', '_') for c in df.columns]

df = df.drop('Patient_ID', axis=1)

# Split Blood_Pressure "120/80" → two numeric columns
df['BP_systolic']  = df['Blood_Pressure'].apply(lambda x: x.split("/")[0])
df['BP_diastolic'] = df['Blood_Pressure'].apply(lambda x: x.split("/")[1])

df = df[['Age', 'Sex', 'Cholesterol',
         'BP_systolic', 'BP_diastolic',
         'Heart_Rate', 'Diabetes',
         'Family_History', 'Smoking', 'Obesity', 'Alcohol_Consumption',
         'Exercise_Hours_Per_Week', 'Diet', 'Previous_Heart_Problems',
         'Medication_Use', 'Stress_Level', 'Sedentary_Hours_Per_Day', 'Income',
         'BMI', 'Triglycerides', 'Physical_Activity_Days_Per_Week',
         'Sleep_Hours_Per_Day', 'Country', 'Continent', 'Hemisphere',
         'Heart_Attack_Risk']]

df['BP_systolic']  = pd.to_numeric(df['BP_systolic'])
df['BP_diastolic'] = pd.to_numeric(df['BP_diastolic'])

df2 = df[['Age', 'Sex', 'Cholesterol', 'BP_systolic', 'BP_diastolic',
          'Heart_Rate', 'Diabetes', 'Family_History', 'Smoking', 'Obesity',
          'Alcohol_Consumption', 'Exercise_Hours_Per_Week', 'Diet',
          'Previous_Heart_Problems', 'Medication_Use', 'Stress_Level',
          'Sedentary_Hours_Per_Day', 'Income', 'BMI', 'Triglycerides',
          'Physical_Activity_Days_Per_Week', 'Sleep_Hours_Per_Day',
          'Heart_Attack_Risk']].copy()

# FIX 3: Removed second hardcoded path df4 = pd.read_csv('D:\\IntrainTech\\...')
# Just use df2 directly for label encoding
df3 = df2[['Sex', 'Diet']].copy()
le  = LabelEncoder()
label_encoder = {}
for column in df3.columns:
    label_encoder[column] = le
    df3[column] = label_encoder[column].fit_transform(df2[column])

df2 = df2.drop(['Sex', 'Diet', 'Income'], axis=1)
df2 = pd.concat([df2, df3], axis=1)

# ─── Features ─────────────────────────────────────────────────────────────────
X = df2[['Age', 'Cholesterol', 'BP_systolic', 'BP_diastolic', 'Heart_Rate',
         'Diabetes', 'Family_History', 'Smoking', 'Obesity',
         'Alcohol_Consumption', 'Exercise_Hours_Per_Week',
         'Previous_Heart_Problems', 'Medication_Use',
         'BMI', 'Triglycerides', 'Sleep_Hours_Per_Day',
         'Sex', 'Diet']]
y = df2[['Heart_Attack_Risk']]

smote            = SMOTE(random_state=50)
X_resample, y_resample = smote.fit_resample(X, y)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_resample)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_resample, test_size=0.2, random_state=42)

y_train = np.ravel(y_train)
y_test  = np.ravel(y_test)

model = RandomForestClassifier(random_state=17)
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
print(f"Model trained. Accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.2f}%")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def dicti_vals(dicti):
    x = list(dicti.values())
    return np.array([x])

def determine_lifestyle_changes(predict_type, new_person):
    lifestyle_changes = []
    if predict_type > 0:
        if new_person.get('Smoking') == 1:
            lifestyle_changes.append('Quit smoking')
        bmi = new_person.get('BMI', 22)
        if bmi < 18.5:
            lifestyle_changes.append('Gain weight to reach a healthy BMI')
        elif bmi > 25:
            lifestyle_changes.append('Lose weight to reach a healthy BMI')
        if new_person.get('Exercise_Hours_Per_Week', 5) < 1.25:
            lifestyle_changes.append('Do more exercise (aim for 150 min/week)')
        if new_person.get('Diet') in [0, 2]:
            lifestyle_changes.append('Eat healthier food')
        if new_person.get('Alcohol_Consumption') == 1:
            lifestyle_changes.append('Try reducing alcohol consumption')

        return {
            'Heart_attack_risk': round(float(predict_type), 4),
            'Lifestyle_changes': lifestyle_changes
        }

    return {
        'Heart_attack_risk': round(float(predict_type), 4),
        'Lifestyle_changes': lifestyle_changes
    }

# ─── Global result store ──────────────────────────────────────────────────────
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
    # FIX 4: Form field names now match what test.html sends
    age                   = int(request.form.get('Age'))
    sex                   = int(request.form.get('sex'))
    BP_systolic           = int(request.form.get('BP_systolic'))
    BP_diastolic          = int(request.form.get('BP_diastolic'))
    Cholesterol           = float(request.form.get('Cholesterol'))
    Triglycerides         = float(request.form.get('Triglycerides'))
    Heart_Rate            = int(request.form.get('Heart_Rate'))
    Diabetes              = int(request.form.get('Diabetes'))
    Family_History        = int(request.form.get('Family_History'))
    Smoking               = int(request.form.get('Smoking'))
    Obesity               = int(request.form.get('Obesity'))
    Alcohol_Consumption   = int(request.form.get('Alcohol_Consumption'))
    Medication_Use        = int(request.form.get('Medication_Use'))
    Diet                  = int(request.form.get('Diet'))
    Sleep_Hours_Per_Day   = int(request.form.get('Sleep_Hours_Per_Day'))
    Previous_Heart_Problems = int(request.form.get('Previous_Heart_Problems'))
    BMI                   = float(request.form.get('BMI'))
    Exercise_Hours_Per_Week = int(request.form.get('Exercise_Hours_Per_Week'))

    new_person = {
        'Age':                    age,
        'Cholesterol':            Cholesterol,
        'BP_systolic':            BP_systolic,
        'BP_diastolic':           BP_diastolic,
        'Heart_Rate':             Heart_Rate,
        'Diabetes':               Diabetes,
        'Family_History':         Family_History,
        'Smoking':                Smoking,
        'Obesity':                Obesity,
        'Alcohol_Consumption':    Alcohol_Consumption,
        'Exercise_Hours_Per_Week': Exercise_Hours_Per_Week,
        'Previous_Heart_Problems': Previous_Heart_Problems,
        'Medication_Use':         Medication_Use,
        'BMI':                    BMI,
        'Triglycerides':          Triglycerides,
        'Sleep_Hours_Per_Day':    Sleep_Hours_Per_Day,
        'Sex':                    sex,
        'Diet':                   Diet,
    }

    x            = dicti_vals(new_person)
    x_scaled     = scaler.transform(x)
    predict_type = model.predict_proba(x_scaled)[:, 1][0]
    result       = determine_lifestyle_changes(predict_type, new_person)

    global data
    data = result
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
