# Heart Attack Prediction & Health Advisor

A machine learning web application built with Python, Flask, and Scikit-Learn to predict heart attack risk based on clinical attributes and provide personalized lifestyle recommendations.

---

## 📌 Project Overview

Heart disease is one of the leading global causes of mortality. Early prediction and lifestyle modification can significantly mitigate cardiac risks. This application takes a patient's medical attributes (e.g. blood pressure, cholesterol, max heart rate, chest pain type, resting ECG, ST depression) and uses a trained **Random Forest Classifier** to compute:
1. **Risk Status**: `Risk of Heart Attack!` or `No risk of Heart Attack!`
2. **Health Score**: A normalized risk score (0.0 to 1.0) indicating cardiac risk probability.
3. **Personalized Lifestyle Suggestions**: Actionable clinical and lifestyle recommendations (e.g. weight management, cardiovascular exercise, diet modification, blood pressure & sugar monitoring) as specified in the project report.

---

## 📊 Dataset & Model Architecture

- **Dataset**: `heart_new.csv` (5,125 patient records with 13 medical attributes).
- **Features Used**:
  - `age`: Age (years)
  - `sex`: Gender (1 = Male, 0 = Female)
  - `cp`: Chest Pain Type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal, 3: Asymptomatic)
  - `trestbps`: Resting Blood Pressure (mm Hg)
  - `chol`: Serum Cholesterol (mg/dl)
  - `fbs`: Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)
  - `restecg`: Resting ECG (0: Normal, 1: ST-T Abnormality, 2: LV Hypertrophy)
  - `thalach`: Maximum Heart Rate Achieved
  - `exang`: Exercise Induced Angina (1 = Yes, 0 = No)
  - `oldpeak`: ST Depression Induced by Exercise Relative to Rest
  - `slope`: Slope of Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)
  - `ca`: Number of Major Vessels (0-4) Colored by Fluoroscopy
  - `thal`: Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect)
- **Primary Model**: **Random Forest Classifier** (Selected as top-performing model in report comparative benchmarks).

---

## 🚀 Quickstart Guide (Local Execution)

### 1. Environment Setup
```bash
# Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train / Retrain ML Model
```bash
python train_model.py
```
*This evaluates model performance across algorithms and saves the optimal model to `savemodel.sav`.*

### 3. Run the Web Application
```bash
python deploy.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000/`**

### 4. Run Automated Tests
```bash
python test_app.py
```

---

## 🌐 Free Cloud Deployment Instructions

### Option 1: Deploying to Render (Recommended)
1. Push your repository to **GitHub**.
2. Log in to [Render.com](https://render.com/) and click **New + -> Web Service**.
3. Connect your GitHub repository (`samuel-mekala/heart-attack-prediction`).
4. Select environment: **Python 3**.
5. Set the build & start commands:
   - **Build Command**: `pip install -r requirements.txt && python train_model.py`
   - **Start Command**: `gunicorn wsgi:app`
6. Click **Create Web Service**. Render will automatically build and deploy your app.

### Option 2: Deploying to Railway
1. Sign up on [Railway.app](https://railway.app/).
2. Click **New Project -> Deploy from GitHub repo**.
3. Select this repository. Railway automatically detects `Procfile` (`web: gunicorn wsgi:app`) and deploys the service.

### Option 3: Deploying to Hugging Face Spaces (Docker / Streamlit / Gradio / Flask)
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) and select **Docker** SDK (or **Blank**).
2. Push this codebase. The `Procfile` / `deploy.py` will launch your Flask app directly on Hugging Face Spaces.

---

## 🛠️ Project Structure

```
Heart_Attack_Prediction/
├── deploy.py               # Main Flask web server & advice engine
├── wsgi.py                 # WSGI production entry point
├── train_model.py          # ML training & benchmark script
├── test_app.py             # Automated unit testing suite
├── savemodel.sav           # Serialized Random Forest model artifact
├── heart_new.csv           # Clinical dataset (5,125 rows)
├── templates/
│   └── index1.html         # Responsive web UI & results dashboard
├── requirements.txt        # Python dependency declarations
├── Procfile                # Gunicorn process config for cloud deployment
├── render.yaml             # Render deployment configuration
└── README.md               # Project documentation
```
