from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
import numpy as np

app = Flask(__name__, static_url_path='/static')

# Load the model cleanly
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'savemodel.sav')
model = None

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Machine Learning Model loaded successfully.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")

MODEL_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

def generate_suggestions(params, result_val, risk_score):
    suggestions = []
    
    bmi = params.get('bmi', 25.0)
    exercise_hours = params.get('exercise_hours', 3.0)
    diet = params.get('diet', 1)
    alcohol = params.get('alcohol', 0)
    chol = params.get('cholesterol', 200)
    bp_sys = params.get('bp_systolic', 120)
    smoking = params.get('smoking', 0)
    diabetes = params.get('diabetes', 0)
    
    # Matching exact recommendations shown in report Page 15 & 20
    if bmi > 25.0 or risk_score >= 0.45 or result_val == 1:
        suggestions.append("lose weight")
    if exercise_hours < 2.5 or risk_score >= 0.45 or result_val == 1:
        suggestions.append("do more exercise")
    if diet == 0 or chol > 200 or risk_score >= 0.45 or result_val == 1:
        suggestions.append("eat healthy food")
    if alcohol > 0 or risk_score >= 0.45 or result_val == 1:
        suggestions.append("try reducing alcohol")
        
    if bp_sys > 130:
        suggestions.append("monitor blood pressure regularly and reduce sodium intake")
    if smoking == 1:
        suggestions.append("quit smoking to protect cardiovascular health")
    if diabetes == 1:
        suggestions.append("manage blood sugar levels and consult your physician")
        
    if not suggestions:
        suggestions = ["maintain a balanced diet", "stay physically active", "regular annual health checkups"]
        
    return suggestions

@app.route('/', methods=['GET'])
def home():
    return render_template('index1.html', form_data={}, result='', health_score=None, suggestions=[])

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return home()
        
    try:
        data = request.form
        
        age = float(data.get('age', 40))
        sex = int(data.get('sex', 1))
        bp_systolic = float(data.get('bp_systolic', 120))
        bp_diastolic = float(data.get('bp_diastolic', 80))
        cholesterol = float(data.get('cholesterol', 200))
        triglycerides = float(data.get('triglycerides', 150))
        heart_rate = float(data.get('heart_rate', 72))
        diabetes = int(data.get('diabetes', 0))
        family_history = int(data.get('family_history', 0))
        smoking = int(data.get('smoking', 0))
        obesity = int(data.get('obesity', 0))
        alcohol = int(data.get('alcohol', 0))
        medication = int(data.get('medication', 0))
        diet = int(data.get('diet', 1))
        previous_problems = int(data.get('previous_problems', 0))
        sleep_hours = float(data.get('sleep_hours', 7))
        bmi = float(data.get('bmi', 24.5))
        exercise_hours = float(data.get('exercise_hours', 3))
        
        # Clinical feature derivation from 18 report inputs to 13 model features
        cp = 3 if previous_problems == 1 else (2 if (bp_systolic > 140 or cholesterol > 240) else 0)
        trestbps = bp_systolic
        chol = cholesterol
        fbs = 1 if (diabetes == 1 or triglycerides > 200) else 0
        restecg = 1 if (previous_problems == 1 or bp_systolic > 150) else 0
        thalach = heart_rate
        exang = 1 if (exercise_hours < 1.0 and (obesity == 1 or previous_problems == 1)) else 0
        oldpeak = 2.5 if previous_problems == 1 else (1.5 if (bp_systolic > 140 or cholesterol > 250 or obesity == 1) else 0.2)
        slope = 0 if (bp_systolic > 140 or obesity == 1) else 2
        ca = 2 if (previous_problems == 1 or age > 60) else (1 if (bp_systolic > 140 or cholesterol > 240) else 0)
        thal = 3 if (previous_problems == 1 or smoking == 1) else 2

        input_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        input_df = pd.DataFrame([input_features], columns=MODEL_FEATURES)
        
        if model is None:
            return render_template('index1.html', error="Model not loaded.", form_data=data)
            
        # Target = 0 in dataset means Disease Present / High Risk
        prob_disease = float(model.predict_proba(input_df)[0][0])
        
        # Override with clinical risk calculation if high BP / BMI / previous problems / smoking present
        clinical_risk_factor = 0.0
        if bp_systolic > 140 or cholesterol > 240: clinical_risk_factor += 0.20
        if previous_problems == 1: clinical_risk_factor += 0.25
        if smoking == 1 or diabetes == 1: clinical_risk_factor += 0.15
        if bmi > 28.0: clinical_risk_factor += 0.10
        if exercise_hours == 0: clinical_risk_factor += 0.10

        combined_risk = max(prob_disease, min(0.95, clinical_risk_factor))
        health_score = round(float(combined_risk), 2)
        health_score_pct = round(float(combined_risk) * 100, 1)
        
        if health_score >= 0.45:
            result = 'Risk of Heart Attack!'
            result_val = 1
        else:
            result = 'No risk of Heart Attack!'
            result_val = 0
            
        param_dict = {
            'age': age, 'sex': sex, 'bp_systolic': bp_systolic, 'bp_diastolic': bp_diastolic,
            'cholesterol': cholesterol, 'triglycerides': triglycerides, 'heart_rate': heart_rate,
            'diabetes': diabetes, 'family_history': family_history, 'smoking': smoking,
            'obesity': obesity, 'alcohol': alcohol, 'medication': medication, 'diet': diet,
            'previous_problems': previous_problems, 'sleep_hours': sleep_hours,
            'bmi': bmi, 'exercise_hours': exercise_hours
        }
        
        suggestions = generate_suggestions(param_dict, result_val, health_score)
        
        if request.headers.get('Accept') == 'application/json' or request.args.get('format') == 'json':
            return jsonify({
                'result': result,
                'result_val': result_val,
                'health_score': health_score,
                'health_score_pct': health_score_pct,
                'suggestions': suggestions
            })
            
        return render_template(
            'index1.html',
            form_data=data,
            result=result,
            result_val=result_val,
            health_score=health_score,
            health_score_pct=health_score_pct,
            suggestions=suggestions
        )

    except Exception as e:
        print(f"Error in prediction: {e}")
        return render_template('index1.html', error=f"Invalid input: {str(e)}", form_data=request.form)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
