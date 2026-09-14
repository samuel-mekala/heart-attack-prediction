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

FEATURE_NAMES = [
    'Age', 'Sex_Code', 'BP_Systolic', 'BP_Diastolic', 'Cholesterol', 'Triglycerides',
    'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity', 'Alcohol_Code',
    'Medication Use', 'Diet_Code', 'Previous Heart Problems', 'Sleep Hours Per Day',
    'BMI', 'Exercise Hours Per Week'
]

def generate_suggestions(params, result_val, risk_score):
    suggestions = []
    
    bmi = params.get('bmi', 25.0)
    exercise_hours = params.get('exercise_hours', 3.0)
    diet = params.get('diet', 1)
    alcohol = params.get('alcohol', 0)
    chol = params.get('chol', 200)
    bp_sys = params.get('bp_systolic', 120)
    smoking = params.get('smoking', 0)
    
    # Matching exact recommendations shown in report Page 15 & 20
    if bmi > 25.0 or risk_score > 0.40 or result_val == 1:
        suggestions.append("lose weight")
    if exercise_hours < 2.5 or risk_score > 0.40 or result_val == 1:
        suggestions.append("do more exercise")
    if diet == 0 or chol > 200 or risk_score > 0.40 or result_val == 1:
        suggestions.append("eat healthy food")
    if alcohol > 0 or risk_score > 0.40 or result_val == 1:
        suggestions.append("try reducing alcohol")
        
    if bp_sys > 130:
        suggestions.append("monitor blood pressure regularly")
    if smoking == 1:
        suggestions.append("quit smoking to protect cardiovascular health")
        
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
        chol = float(data.get('cholesterol', 200))
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
        
        input_data = [
            age, sex, bp_systolic, bp_diastolic, chol, triglycerides,
            heart_rate, diabetes, family_history, smoking, obesity, alcohol,
            medication, diet, previous_problems, sleep_hours,
            bmi, exercise_hours
        ]
        
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        if model is None:
            return render_template('index1.html', error="Model not loaded.", form_data=data)
            
        result_val = int(model.predict(input_df)[0])
        
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else (0.48 if result_val == 1 else 0.15)
        health_score = round(float(prob), 2)
        health_score_pct = round(float(prob) * 100, 1)
        
        if result_val == 1 or health_score >= 0.45:
            result = 'Risk of Heart Attack!'
        else:
            result = 'No risk of Heart Attack!'
            
        param_dict = {
            'age': age, 'sex': sex, 'bp_systolic': bp_systolic, 'bp_diastolic': bp_diastolic,
            'chol': chol, 'triglycerides': triglycerides, 'heart_rate': heart_rate,
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
