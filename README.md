# 🫀 Heart Attack Risk Prediction

🌐 **Live Demo:** https://heart-attack-prediction-ilk1.onrender.com

> **ML Internship Project** · IntrainTech, Bangalore · Aug–Nov 2023
> **Role:** Machine Learning Engineer Intern

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=flat-square&logo=powerbi)](Dashboard.pbix)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?style=flat-square&logo=github-actions)](/.github/workflows/ci.yml)

---

## 📌 What This Project Does

End-to-end heart attack risk prediction system — from raw clinical data to a **live Flask web application** and **Power BI dashboard**. Patients fill a form, the model predicts risk probability, and the app returns personalised lifestyle change recommendations.

---

## 🏗️ System Architecture

```
Patient fills web form (test.html)
            │
            ▼ POST /predict
┌──────────────────────────────┐
│         server.py            │
│                              │
│  1. Parse form inputs        │
│  2. Scale with StandardScaler│
│  3. model.predict_proba()    │
│  4. determine_lifestyle_     │
│     changes(prob, inputs)    │
│  5. Return JSON response     │
└──────────────┬───────────────┘
               │
               ▼
    result_template.html
    • Risk probability score
    • High / Low risk label
    • Personalised recommendations
      (smoking, BMI, exercise,
       diet, sleep, stress)
```

---

## 📊 Model Benchmark (10-Fold Cross-Validation)

| Model | Accuracy |
|---|---|
| **Random Forest** ✅ | **69.17%** |
| Light Gradient Boost | ~67% |
| SVM | ~65% |
| XGBoost | ~64% |
| KNN | ~63% |
| Logistic Regression | ~62% |
| Decision Tree | ~58% |
| Naive Bayes | ~57% |

**Random Forest selected** — best cross-validated accuracy across 10 folds. Evaluated using Accuracy, F1-Score, ROC-AUC, Precision, and Recall.

---

## 🔑 Key Engineering Decisions

**Why SMOTE before training?**
Heart attack risk classes are imbalanced. SMOTE generates synthetic minority samples preserving feature distributions, preventing the model from always predicting the majority class.

**Why StandardScaler?**
Features like Cholesterol (100–300), BMI (15–45), and Heart Rate (60–100) have very different ranges. Scaling ensures no single feature dominates distance-based calculations.

**Why lifestyle recommendations?**
A risk score alone isn't actionable. The recommendations engine maps specific input values (Smoking=1, BMI>25, Exercise<1.25h/week) to concrete changes — making the app clinically useful.

**Why split Blood Pressure?**
The raw dataset stores BP as "120/80" string. Splitting into systolic and diastolic gives the model two meaningful numeric features instead of one useless string.

---

## 🗂️ Dataset

**Heart Attack Risk Prediction Dataset** — 8,763 patient records, 25 features:

| Category | Features |
|---|---|
| Demographics | Age, Sex, Country, Continent, Hemisphere |
| Vitals | BP (Systolic/Diastolic split), Heart Rate, Cholesterol, BMI, Triglycerides |
| Lifestyle | Smoking, Alcohol, Exercise Hrs/Week, Diet, Sedentary Hrs, Stress Level, Sleep Hrs |
| Medical History | Diabetes, Family History, Previous Heart Problems, Obesity, Medication Use |
| Target | Heart Attack Risk (0 = Low, 1 = High) |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat-square&logo=powerbi&logoColor=black)

---

## 📁 Project Structure

```
heart-attack-prediction/
│
├── server.py
├── train_model.py
├── model.pkl
├── scaler.pkl
├── requirements.txt
├── render.yaml
├── .env.example
├── README.md
│
├── templates/
│   ├── test.html
│   └── result_template.html
│
└── .github/
    └── workflows/
        └── ci.yml
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/samuel-mekala/heart-attack-prediction.git
cd heart-attack-prediction

# Install dependencies
pip install -r requirements.txt

# Run the Flask app
# (model trains automatically on startup — ~10-15 seconds)
python server.py

# Open browser → http://localhost:5000

# To explore EDA and all 8 models:
jupyter notebook 1.ipynb
```

---

## 🔮 Future Work

- [ ] Deploy on Render for public access
- [ ] Add SHAP explainability — show which features drive each prediction
- [ ] REST API endpoint for hospital management system integration
- [ ] Patient history tracking with database backend
- [ ] Retrain pipeline with new patient data

---

*IntrainTech Internship · Bangalore · Aug–Nov 2023*
