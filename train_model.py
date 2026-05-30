import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv("heart_attack_prediction_dataset.csv")

df.columns = [c.strip().replace(" ", "_") for c in df.columns]

df = df.drop("Patient_ID", axis=1)

df["BP_systolic"] = df["Blood_Pressure"].apply(
    lambda x: int(x.split("/")[0])
)

df["BP_diastolic"] = df["Blood_Pressure"].apply(
    lambda x: int(x.split("/")[1])
)

le_sex = LabelEncoder()
le_diet = LabelEncoder()

df["Sex"] = le_sex.fit_transform(df["Sex"])
df["Diet"] = le_diet.fit_transform(df["Diet"])

features = [
    "Age",
    "Cholesterol",
    "BP_systolic",
    "BP_diastolic",
    "Heart_Rate",
    "Diabetes",
    "Family_History",
    "Smoking",
    "Obesity",
    "Alcohol_Consumption",
    "Exercise_Hours_Per_Week",
    "Previous_Heart_Problems",
    "Medication_Use",
    "BMI",
    "Triglycerides",
    "Sleep_Hours_Per_Day",
    "Sex",
    "Diet"
]

X = df[features]
y = df["Heart_Attack_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Saved model.pkl")
print("Saved scaler.pkl")
