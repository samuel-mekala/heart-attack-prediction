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

FEATURE_NAMES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

def generate_suggestions(params, result_val, risk_score):
    suggestions = []
    
    chol = params.get('chol', 200)
    trestbps = params.get('trestbps', 120)
    exang = params.get('exang', 0)
    oldpeak = params.get('oldpeak', 0.0)
    thalach = params.get('thalach', 150)
    fbs = params.get('fbs', 0)
    
    if risk_score > 0.40 or result_val == 1:
        suggestions.append("lose weight")
        suggestions.append("do more exercise")
        suggestions.append("eat healthy food")
        suggestions.append("try reducing alcohol")
    
    if chol > 200:
        if "eat healthy food" not in suggestions:
            suggestions.append("eat healthy food (low saturated fat & cholesterol)")
    if trestbps > 130:
        suggestions.append("monitor blood pressure regularly and reduce sodium intake")
    if fbs == 1:
        suggestions.append("manage blood sugar levels and consult a dietitian")
    if exang == 1 or thalach < 120:
        if "do more exercise" not in suggestions:
            suggestions.append("do regular moderate cardiovascular exercise")
    if oldpeak > 1.5:
        suggestions.append("schedule a detailed cardiac checkup with your physician")
        
    if not suggestions:
        suggestions = ["maintain a balanced diet", "stay physically active", "regular annual health checkups"]
        
    return suggestions

@app.route('/', methods=['GET'])
def home():
    # Blank form_data on fresh load
    return render_template('index1.html', form_data={}, result='', health_score=None, suggestions=[])

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'GET':
        return home()
        
    try:
        data = request.form
        
        age = int(data.get('age', 42))
        sex = int(data.get('sex', 0))
        cp = int(data.get('cp', 0))
        trestbps = int(data.get('trestbps', 120))
        chol = int(data.get('chol', 200))
        fbs = int(data.get('fbs', 0))
        restecg = int(data.get('restecg', 0))
        thalach = int(data.get('thalach', 150))
        exang = int(data.get('exang', 0))
        oldpeak = float(data.get('oldpeak', 0.0))
        slope = int(data.get('slope', 1))
        ca = int(data.get('ca', 0))
        thal = int(data.get('thal', 2))
        
        input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        
        if model is None:
            return render_template('index1.html', error="Model not loaded.", form_data=data)
            
        result_val = int(model.predict(input_df)[0])
        
        prob = model.predict_proba(input_df)[0][1] if hasattr(model, 'predict_proba') else (0.85 if result_val == 1 else 0.15)
        health_score = round(float(prob), 2)
        health_score_pct = round(float(prob) * 100, 1)
        
        if result_val == 1:
            result = 'Risk of Heart Attack!'
        else:
            result = 'No risk of Heart Attack!'
            
        param_dict = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
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
    try:
        app.run(host='0.0.0.0', port=port, debug=False)
    except OSError:
        app.run(host='0.0.0.0', port=5002, debug=False)
