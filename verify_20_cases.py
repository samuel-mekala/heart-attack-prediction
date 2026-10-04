import pickle
import os
import pandas as pd
import numpy as np
from deploy import compute_aha_ascvd_score, generate_suggestions, MODEL_FEATURES, rf_model, gb_model

test_cases = [
    # Category 1: Optimal / Low Risk
    {
        "id": 1, "category": "Optimal / Low Risk", "profile": "19yo Active Female",
        "params": {"age": 19, "sex": 0, "bp_systolic": 110, "bp_diastolic": 70, "cholesterol": 165, "triglycerides": 110, "heart_rate": 65, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 8, "bmi": 20.0, "exercise_hours": 5.0}
    },
    {
        "id": 2, "category": "Optimal / Low Risk", "profile": "24yo Healthy Male Athlete",
        "params": {"age": 24, "sex": 1, "bp_systolic": 115, "bp_diastolic": 75, "cholesterol": 175, "triglycerides": 120, "heart_rate": 58, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 8, "bmi": 22.5, "exercise_hours": 6.0}
    },
    {
        "id": 3, "category": "Optimal / Low Risk", "profile": "42yo Fit Non-Smoker Female",
        "params": {"age": 42, "sex": 0, "bp_systolic": 118, "bp_diastolic": 78, "cholesterol": 180, "triglycerides": 130, "heart_rate": 68, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.5, "bmi": 21.0, "exercise_hours": 4.0}
    },
    {
        "id": 4, "category": "Optimal / Low Risk", "profile": "52yo Active Male",
        "params": {"age": 52, "sex": 1, "bp_systolic": 120, "bp_diastolic": 80, "cholesterol": 190, "triglycerides": 140, "heart_rate": 70, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.0, "bmi": 23.5, "exercise_hours": 3.5}
    },

    # Category 2: Borderline / Moderate Risk
    {
        "id": 5, "category": "Borderline / Moderate", "profile": "45yo Male with Stage-1 HTN",
        "params": {"age": 45, "sex": 1, "bp_systolic": 138, "bp_diastolic": 88, "cholesterol": 215, "triglycerides": 175, "heart_rate": 76, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 6.5, "bmi": 26.5, "exercise_hours": 2.0}
    },
    {
        "id": 6, "category": "Borderline / Moderate", "profile": "50yo Female Borderline Lipids & Low Sleep",
        "params": {"age": 50, "sex": 0, "bp_systolic": 128, "bp_diastolic": 82, "cholesterol": 235, "triglycerides": 190, "heart_rate": 74, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 0, "previous_problems": 0, "sleep_hours": 5.0, "bmi": 25.0, "exercise_hours": 2.0}
    },
    {
        "id": 7, "category": "Borderline / Moderate", "profile": "38yo Sedentary Overweight Male",
        "params": {"age": 38, "sex": 1, "bp_systolic": 132, "bp_diastolic": 85, "cholesterol": 220, "triglycerides": 180, "heart_rate": 80, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 1, "alcohol": 1, "medication": 0, "diet": 0, "previous_problems": 0, "sleep_hours": 6.0, "bmi": 28.5, "exercise_hours": 0.5}
    },
    {
        "id": 8, "category": "Borderline / Moderate", "profile": "58yo Male with Family History",
        "params": {"age": 58, "sex": 1, "bp_systolic": 134, "bp_diastolic": 84, "cholesterol": 210, "triglycerides": 160, "heart_rate": 72, "diabetes": 0, "family_history": 1, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.0, "bmi": 24.5, "exercise_hours": 2.5}
    },

    # Category 3: High Risk / Severe
    {
        "id": 9, "category": "High Risk / Severe", "profile": "65yo Diabetic Smoker w/ HTN",
        "params": {"age": 65, "sex": 1, "bp_systolic": 165, "bp_diastolic": 100, "cholesterol": 270, "triglycerides": 250, "heart_rate": 88, "diabetes": 1, "family_history": 1, "smoking": 1, "obesity": 1, "alcohol": 0, "medication": 1, "diet": 0, "previous_problems": 0, "sleep_hours": 5.5, "bmi": 32.0, "exercise_hours": 0.5}
    },
    {
        "id": 10, "category": "High Risk / Severe", "profile": "72yo Prior Heart Attack Patient",
        "params": {"age": 72, "sex": 1, "bp_systolic": 155, "bp_diastolic": 95, "cholesterol": 290, "triglycerides": 320, "heart_rate": 82, "diabetes": 0, "family_history": 1, "smoking": 0, "obesity": 1, "alcohol": 0, "medication": 1, "diet": 0, "previous_problems": 1, "sleep_hours": 6.0, "bmi": 31.0, "exercise_hours": 1.0}
    },
    {
        "id": 11, "category": "High Risk / Severe", "profile": "62yo Severe Multi-Risk Factor",
        "params": {"age": 62, "sex": 1, "bp_systolic": 175, "bp_diastolic": 105, "cholesterol": 310, "triglycerides": 350, "heart_rate": 92, "diabetes": 1, "family_history": 1, "smoking": 1, "obesity": 1, "alcohol": 1, "medication": 1, "diet": 0, "previous_problems": 1, "sleep_hours": 5.0, "bmi": 36.0, "exercise_hours": 0.0}
    },
    {
        "id": 12, "category": "High Risk / Severe", "profile": "78yo Elderly Patient w/ CAD History",
        "params": {"age": 78, "sex": 0, "bp_systolic": 160, "bp_diastolic": 90, "cholesterol": 260, "triglycerides": 230, "heart_rate": 85, "diabetes": 1, "family_history": 0, "smoking": 0, "obesity": 1, "alcohol": 0, "medication": 1, "diet": 0, "previous_problems": 1, "sleep_hours": 6.5, "bmi": 30.5, "exercise_hours": 1.0}
    },

    # Category 4: Conflicting Risk Factors
    {
        "id": 13, "category": "Conflicting Risk Factors", "profile": "26yo Active Smoker w/ Normal Vitals",
        "params": {"age": 26, "sex": 1, "bp_systolic": 118, "bp_diastolic": 75, "cholesterol": 170, "triglycerides": 125, "heart_rate": 64, "diabetes": 0, "family_history": 0, "smoking": 1, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.5, "bmi": 22.0, "exercise_hours": 5.0}
    },
    {
        "id": 14, "category": "Conflicting Risk Factors", "profile": "35yo Obese Non-Smoker w/ Normal BP",
        "params": {"age": 35, "sex": 0, "bp_systolic": 120, "bp_diastolic": 80, "cholesterol": 190, "triglycerides": 150, "heart_rate": 72, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 1, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.0, "bmi": 34.0, "exercise_hours": 2.0}
    },
    {
        "id": 15, "category": "Conflicting Risk Factors", "profile": "70yo Fit Senior w/ High Cholesterol",
        "params": {"age": 70, "sex": 0, "bp_systolic": 122, "bp_diastolic": 78, "cholesterol": 285, "triglycerides": 160, "heart_rate": 66, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 8.0, "bmi": 22.5, "exercise_hours": 4.5}
    },
    {
        "id": 16, "category": "Conflicting Risk Factors", "profile": "29yo Diabetic w/ Active Lifestyle",
        "params": {"age": 29, "sex": 1, "bp_systolic": 115, "bp_diastolic": 75, "cholesterol": 175, "triglycerides": 140, "heart_rate": 68, "diabetes": 1, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.5, "bmi": 21.5, "exercise_hours": 4.0}
    },

    # Category 5: Extreme Edge & Boundary Cases
    {
        "id": 17, "category": "Extreme Edge / Boundary", "profile": "Extreme High Vitals Upper Boundary",
        "params": {"age": 85, "sex": 1, "bp_systolic": 230, "bp_diastolic": 140, "cholesterol": 550, "triglycerides": 600, "heart_rate": 115, "diabetes": 1, "family_history": 1, "smoking": 1, "obesity": 1, "alcohol": 1, "medication": 1, "diet": 0, "previous_problems": 1, "sleep_hours": 3.0, "bmi": 45.0, "exercise_hours": 0.0}
    },
    {
        "id": 18, "category": "Extreme Edge / Boundary", "profile": "Young Minimum Age Boundary (18yo)",
        "params": {"age": 18, "sex": 0, "bp_systolic": 95, "bp_diastolic": 60, "cholesterol": 130, "triglycerides": 80, "heart_rate": 60, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 9.0, "bmi": 18.5, "exercise_hours": 6.0}
    },
    {
        "id": 19, "category": "Extreme Edge / Boundary", "profile": "48yo Severe Sleep Deprivation (3h)",
        "params": {"age": 48, "sex": 1, "bp_systolic": 145, "bp_diastolic": 92, "cholesterol": 245, "triglycerides": 210, "heart_rate": 84, "diabetes": 0, "family_history": 1, "smoking": 0, "obesity": 0, "alcohol": 1, "medication": 0, "diet": 0, "previous_problems": 0, "sleep_hours": 3.0, "bmi": 27.0, "exercise_hours": 0.0}
    },
    {
        "id": 20, "category": "Extreme Edge / Boundary", "profile": "90yo Senior w/ Controlled Vitals",
        "params": {"age": 90, "sex": 0, "bp_systolic": 125, "bp_diastolic": 80, "cholesterol": 195, "triglycerides": 135, "heart_rate": 68, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 8.0, "bmi": 23.0, "exercise_hours": 3.0}
    }
]

results = []

for case in test_cases:
    p = case["params"]
    age = p["age"]
    sex = p["sex"]
    bp_sys = p["bp_systolic"]
    bp_dia = p["bp_diastolic"]
    chol = p["cholesterol"]
    trig = p["triglycerides"]
    hr = p["heart_rate"]
    diabetes = p["diabetes"]
    family_hist = p["family_history"]
    smoking = p["smoking"]
    obesity = p["obesity"]
    alcohol = p["alcohol"]
    med = p["medication"]
    diet = p["diet"]
    prev_prob = p["previous_problems"]
    sleep = p["sleep_hours"]
    bmi = p["bmi"]
    exercise = p["exercise_hours"]

    # Derived ML inputs
    cp = 3 if prev_prob == 1 else (2 if (bp_sys > 140 or chol > 240) else 0)
    trestbps = bp_sys
    chol_val = chol
    fbs = 1 if (diabetes == 1 or trig > 200) else 0
    restecg = 1 if (prev_prob == 1 or bp_sys > 150) else 0
    thalach = hr
    exang = 1 if (exercise < 1.0 and (obesity == 1 or prev_prob == 1)) else 0
    oldpeak = 2.5 if prev_prob == 1 else (1.5 if (bp_sys > 140 or chol > 250 or obesity == 1) else 0.2)
    slope = 0 if (bp_sys > 140 or obesity == 1) else 2
    ca = 2 if (prev_prob == 1 or age > 60) else (1 if (bp_sys > 140 or chol > 240) else 0)
    thal = 3 if (prev_prob == 1 or smoking == 1) else 2

    input_df = pd.DataFrame([[age, sex, cp, trestbps, chol_val, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=MODEL_FEATURES)

    rf_prob = float(rf_model.predict_proba(input_df)[0][0]) if rf_model is not None else 0.5
    gb_prob = float(gb_model.predict_proba(input_df)[0][0]) if gb_model is not None else rf_prob
    ml_prob = 0.5 * rf_prob + 0.5 * gb_prob

    clinical_prob = compute_aha_ascvd_score(age, sex, bp_sys, bp_dia, chol, trig, hr, diabetes, family_hist, smoking, obesity, alcohol, med, diet, prev_prob, sleep, bmi, exercise)

    final_risk = min(0.98, max(0.02, 0.45 * ml_prob + 0.55 * clinical_prob))
    health_score = round(float(final_risk), 2)
    health_score_pct = round(float(final_risk) * 100, 1)

    result_text = "Risk of Heart Attack!" if health_score >= 0.40 else "No risk of Heart Attack!"
    result_val = 1 if health_score >= 0.40 else 0

    suggestions = generate_suggestions(p, result_val, health_score)

    results.append({
        "id": case["id"],
        "category": case["category"],
        "profile": case["profile"],
        "ml_prob": round(ml_prob, 3),
        "clinical_prob": round(clinical_prob, 3),
        "risk_score": health_score,
        "risk_pct": health_score_pct,
        "result": result_text,
        "suggestions": suggestions
    })

print("=== 20 TEST CASES EVALUATION COMPLETED ===")
for r in results:
    print(f"Case {r['id']:2d} | [{r['category']}] {r['profile']:<38} | ML Prob: {r['ml_prob']:.2f} | ASCVD: {r['clinical_prob']:.2f} | Final Risk: {r['risk_score']:.2f} ({r['risk_pct']}%) | Result: {r['result']}")
    print(f"        Suggestions: {', '.join(r['suggestions'])}\n")
