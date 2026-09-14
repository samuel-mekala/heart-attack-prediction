from flask import Flask, render_template, request, jsonify
import pickle
import os
import pandas as pd
import numpy as np

app = Flask(__name__, static_url_path='/static')

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'savemodel.sav')
rf_model = None
gb_model = None
MODEL_FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

try:
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)
        if isinstance(bundle, dict):
            rf_model = bundle.get('rf')
            gb_model = bundle.get('gb')
        else:
            rf_model = bundle
    print("Machine Learning Model Bundle loaded successfully.")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")

def compute_aha_ascvd_score(age, sex, bp_sys, bp_dia, chol, trig, hr, diabetes, family_hist, smoking, obesity, alcohol, med, diet, prev_prob, sleep, bmi, exercise):
    points = 0.0
    
    # 1. Age Factor
    if age >= 65: points += 0.25
    elif age >= 55: points += 0.18
    elif age >= 45: points += 0.10
    elif age >= 35: points += 0.04
    
    # 2. Sex Baseline Risk
    if sex == 1: points += 0.05
    
    # 3. Blood Pressure Stage (Hypertension Criteria)
    if bp_sys >= 160 or bp_dia >= 100: points += 0.22
    elif bp_sys >= 140 or bp_dia >= 90: points += 0.15
    elif bp_sys >= 130 or bp_dia >= 85: points += 0.08
    elif bp_sys >= 120: points += 0.03
    
    # 4. Lipids (Cholesterol & Triglycerides)
    if chol >= 280 or trig >= 300: points += 0.18
    elif chol >= 240 or trig >= 200: points += 0.12
    elif chol >= 200 or trig >= 150: points += 0.06
    
    # 5. Clinical Medical History
    if prev_prob == 1: points += 0.25
    if diabetes == 1: points += 0.18
    if smoking == 1: points += 0.16
    if family_hist == 1: points += 0.10
    
    # 6. Obesity & Lifestyle Variables
    if bmi >= 35: points += 0.12
    elif bmi >= 30 or obesity == 1: points += 0.08
    elif bmi >= 25: points += 0.04
    
    if exercise <= 0.5: points += 0.06
    if diet == 0: points += 0.05
    if alcohol == 1: points += 0.04
    if sleep < 6 or sleep > 10: points += 0.03
    
    return min(0.98, points)

def generate_suggestions(params, result_val, risk_score):
    suggestions = []
    
    bmi = params.get('bmi', 24.5)
    exercise_hours = params.get('exercise_hours', 3.0)
    diet = params.get('diet', 1)
    alcohol = params.get('alcohol', 0)
    chol = params.get('cholesterol', 200)
    bp_sys = params.get('bp_systolic', 120)
    smoking = params.get('smoking', 0)
    diabetes = params.get('diabetes', 0)
    
    # Matching exact recommendations shown in report Page 15 & 20
    if bmi >= 25.0 or risk_score >= 0.40 or result_val == 1:
        suggestions.append("lose weight")
    if exercise_hours < 2.5 or risk_score >= 0.40 or result_val == 1:
        suggestions.append("do more exercise")
    if diet == 0 or chol >= 200 or risk_score >= 0.40 or result_val == 1:
        suggestions.append("eat healthy food")
    if alcohol > 0 or risk_score >= 0.40 or result_val == 1:
        suggestions.append("try reducing alcohol")
        
    if bp_sys >= 130:
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
        
        # Clinical Feature Derivation for ML Model
        cp = 3 if previous_problems == 1 else (2 if (bp_systolic > 140 or cholesterol > 240) else 0)
        trestbps = bp_systolic
        chol_val = cholesterol
        fbs = 1 if (diabetes == 1 or triglycerides > 200) else 0
        restecg = 1 if (previous_problems == 1 or bp_systolic > 150) else 0
        thalach = heart_rate
        exang = 1 if (exercise_hours < 1.0 and (obesity == 1 or previous_problems == 1)) else 0
        oldpeak = 2.5 if previous_problems == 1 else (1.5 if (bp_systolic > 140 or cholesterol > 250 or obesity == 1) else 0.2)
        slope = 0 if (bp_systolic > 140 or obesity == 1) else 2
        ca = 2 if (previous_problems == 1 or age > 60) else (1 if (bp_systolic > 140 or cholesterol > 240) else 0)
        thal = 3 if (previous_problems == 1 or smoking == 1) else 2

        input_df = pd.DataFrame([[age, sex, cp, trestbps, chol_val, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=MODEL_FEATURES)
        
        # ML Model Ensemble Probabilities (Probability of Class 0 = Disease Present)
        rf_prob = float(rf_model.predict_proba(input_df)[0][0]) if rf_model is not None else 0.5
        gb_prob = float(gb_model.predict_proba(input_df)[0][0]) if gb_model is not None else rf_prob
        ml_prob = 0.5 * rf_prob + 0.5 * gb_prob
        
        # AHA/ACC Clinical Risk Standard Score
        clinical_prob = compute_aha_ascvd_score(age, sex, bp_systolic, bp_diastolic, cholesterol, triglycerides, heart_rate, diabetes, family_history, smoking, obesity, alcohol, medication, diet, previous_problems, sleep_hours, bmi, exercise_hours)
        
        # Ensembled Hybrid Risk Score (45% ML Ensemble + 55% Clinical Standard)
        final_risk = min(0.98, max(0.02, 0.45 * ml_prob + 0.55 * clinical_prob))
        health_score = round(float(final_risk), 2)
        health_score_pct = round(float(final_risk) * 100, 1)
        
        if health_score >= 0.40:
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
