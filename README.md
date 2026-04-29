# 🫀 Heart Attack Prediction

> **ML Internship Project** · IntrainTech, Bangalore · Aug–Nov 2023  
> **Role:** Machine Learning Engineer Intern  
> **My Role:** Data preprocessing · Model implementation · Flask deployment

---

## 📌 Overview

Heart Attack is one of the leading causes of death worldwide. This project builds a **machine learning-powered web application** that predicts the likelihood of a patient having heart Attack based on their clinical profile — and suggests personalized lifestyle changes.

We benchmarked **8 ML algorithms** on a comprehensive healthcare dataset and deployed the best model as a **live Flask web app** with real-time predictions.

---

## 🏆 Model Comparison Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| **Random Forest** | **67%** | **0.673** |
| Extra Trees | 65% | 0.652 |
| SVM | 65% | 0.652 |
| XGBoost | 63% | 0.633 |
| Gradient Boosting | 64% | 0.635 |
| AdaBoost | 63% | 0.628 |
| KNN | 61% | 0.608 |
| Decision Tree | 58% | 0.581 |

> **Random Forest Classifier selected as the final model** — best overall accuracy and ROC-AUC.

---

## 📊 Dataset

**76 clinical attributes** (subset of 26 used). Key features include:

| Category | Features |
|---|---|
| **Demographics** | Age, Sex, Country, Continent |
| **Vitals** | Blood Pressure, Heart Rate, Cholesterol, BMI, Triglycerides |
| **Lifestyle** | Smoking, Alcohol, Exercise Hours/Week, Diet, Sedentary Hours, Stress Level |
| **Medical History** | Diabetes, Family History, Previous Heart Problems, Medication Use |
| **Target** | Heart Attack Risk (1 = Yes, 0 = No) |

---

## ⚙️ Methodology

### Preprocessing Pipeline
```
Raw Data
    → Handle Missing Values
    → Feature Engineering
    → Label Encoding (categorical → numeric)
    → SMOTE (handle class imbalance)
    → Train/Test Split (80/20)
    → Standardization (StandardScaler)
```

### Models Benchmarked
1. Decision Tree Classifier
2. Support Vector Machines (SVM)
3. Random Forest Classifier ✅ (Best)
4. Gradient Boosting Classifier
5. AdaBoost Classifier
6. Extra Trees Classifier
7. K-Nearest Neighbors (KNN)
8. XGBoost Classifier

### Evaluation Metrics
- Accuracy · Precision · Recall · F1-Score
- ROC-AUC Score
- Confusion Matrix · Correlation Heatmap

---

## 🖥️ Flask Web Application

The model is deployed as an interactive web app:
- **Input:** Patient fills a clinical form (age, BP, cholesterol, lifestyle, etc.)
- **Output:** Heart attack risk prediction (High / Low)
- **Bonus:** Personalized lifestyle change recommendations based on risk factors

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-3c7ebf?style=flat-square)
![Power BI](https://img.shields.io/badge/Power%20BI-F2C811?style=flat-square&logo=powerbi&logoColor=black)

---

## 📁 Project Structure

```
heart-Attack-prediction/
├── data/
│   └── heart_Attack_dataset.csv
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── models/
│   └── random_forest_model.pkl
├── app/
│   ├── app.py              # Flask application
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── static/
│       └── style.css
├── preprocessing.py
├── train.py
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/samuel-mekala/heart-Attack-prediction.git
cd heart-Attack-prediction

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Run the Flask app
python app/app.py

# Open browser → http://localhost:5000
```

---

## 📈 Key Visualizations

- **Correlation Heatmap** — Identifies most predictive clinical features
- **Distribution Plots** — Class balance analysis before and after SMOTE
- **Heart Attack Risk by Country** — Geographic distribution of risk
- **ROC Curves** — Comparative model performance

---

## 🔮 Future Work

- [ ] Improve accuracy with ensemble stacking
- [ ] Deploy on cloud (AWS / Heroku)
- [ ] Build mobile app version
- [ ] Add SHAP explainability for each prediction

---

*IntrainTech Internship Project · Bangalore · Aug–Nov 2023*
