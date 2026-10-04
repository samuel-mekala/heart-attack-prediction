import unittest
import json
from deploy import app

class Test20CasesFlask(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def run_case(self, case_id, category, profile, params):
        response = self.app.post('/predict?format=json', data=params)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        return data

test_cases = [
    # Category 1: Optimal / Low Risk
    (1, "Optimal / Low Risk", "19yo Active Female", {"age": 19, "sex": 0, "bp_systolic": 110, "bp_diastolic": 70, "cholesterol": 165, "triglycerides": 110, "heart_rate": 65, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 8, "bmi": 20.0, "exercise_hours": 5.0}),
    (2, "Optimal / Low Risk", "24yo Healthy Male Athlete", {"age": 24, "sex": 1, "bp_systolic": 115, "bp_diastolic": 75, "cholesterol": 175, "triglycerides": 120, "heart_rate": 58, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 8, "bmi": 22.5, "exercise_hours": 6.0}),
    (3, "Optimal / Low Risk", "42yo Fit Non-Smoker Female", {"age": 42, "sex": 0, "bp_systolic": 118, "bp_diastolic": 78, "cholesterol": 180, "triglycerides": 130, "heart_rate": 68, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.5, "bmi": 21.0, "exercise_hours": 4.0}),
    (4, "Optimal / Low Risk", "52yo Active Male", {"age": 52, "sex": 1, "bp_systolic": 120, "bp_diastolic": 80, "cholesterol": 190, "triglycerides": 140, "heart_rate": 70, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.0, "bmi": 23.5, "exercise_hours": 3.5}),

    # Category 2: Borderline / Moderate Risk
    (5, "Borderline / Moderate", "45yo Male with Stage-1 HTN", {"age": 45, "sex": 1, "bp_systolic": 138, "bp_diastolic": 88, "cholesterol": 215, "triglycerides": 175, "heart_rate": 76, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 6.5, "bmi": 26.5, "exercise_hours": 2.0}),
    (6, "Borderline / Moderate", "50yo Female Borderline Lipids & Low Sleep", {"age": 50, "sex": 0, "bp_systolic": 128, "bp_diastolic": 82, "cholesterol": 235, "triglycerides": 190, "heart_rate": 74, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 0, "previous_problems": 0, "sleep_hours": 5.0, "bmi": 25.0, "exercise_hours": 2.0}),
    (7, "Borderline / Moderate", "38yo Sedentary Overweight Male", {"age": 38, "sex": 1, "bp_systolic": 132, "bp_diastolic": 85, "cholesterol": 220, "triglycerides": 180, "heart_rate": 80, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 1, "alcohol": 1, "medication": 0, "diet": 0, "previous_problems": 0, "sleep_hours": 6.0, "bmi": 28.5, "exercise_hours": 0.5}),
    (8, "Borderline / Moderate", "58yo Male with Family History", {"age": 58, "sex": 1, "bp_systolic": 134, "bp_diastolic": 84, "cholesterol": 210, "triglycerides": 160, "heart_rate": 72, "diabetes": 0, "family_history": 1, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.0, "bmi": 24.5, "exercise_hours": 2.5}),

    # Category 3: High Risk / Severe
    (9, "High Risk / Severe", "65yo Diabetic Smoker w/ HTN", {"age": 65, "sex": 1, "bp_systolic": 165, "bp_diastolic": 100, "cholesterol": 270, "triglycerides": 250, "heart_rate": 88, "diabetes": 1, "family_history": 1, "smoking": 1, "obesity": 1, "alcohol": 0, "medication": 1, "diet": 0, "previous_problems": 0, "sleep_hours": 5.5, "bmi": 32.0, "exercise_hours": 0.5}),
    (10, "High Risk / Severe", "72yo Prior Heart Attack Patient", {"age": 72, "sex": 1, "bp_systolic": 155, "bp_diastolic": 95, "cholesterol": 290, "triglycerides": 320, "heart_rate": 82, "diabetes": 0, "family_history": 1, "smoking": 0, "obesity": 1, "alcohol": 0, "medication": 1, "diet": 0, "previous_problems": 1, "sleep_hours": 6.0, "bmi": 31.0, "exercise_hours": 1.0}),
    (11, "High Risk / Severe", "62yo Severe Multi-Risk Factor", {"age": 62, "sex": 1, "bp_systolic": 175, "bp_diastolic": 105, "cholesterol": 310, "triglycerides": 350, "heart_rate": 92, "diabetes": 1, "family_history": 1, "smoking": 1, "obesity": 1, "alcohol": 1, "medication": 1, "diet": 0, "previous_problems": 1, "sleep_hours": 5.0, "bmi": 36.0, "exercise_hours": 0.0}),
    (12, "High Risk / Severe", "78yo Elderly Patient w/ CAD History", {"age": 78, "sex": 0, "bp_systolic": 160, "bp_diastolic": 90, "cholesterol": 260, "triglycerides": 230, "heart_rate": 85, "diabetes": 1, "family_history": 0, "smoking": 0, "obesity": 1, "alcohol": 0, "medication": 1, "diet": 0, "previous_problems": 1, "sleep_hours": 6.5, "bmi": 30.5, "exercise_hours": 1.0}),

    # Category 4: Conflicting Risk Factors
    (13, "Conflicting Risk Factors", "26yo Active Smoker w/ Normal Vitals", {"age": 26, "sex": 1, "bp_systolic": 118, "bp_diastolic": 75, "cholesterol": 170, "triglycerides": 125, "heart_rate": 64, "diabetes": 0, "family_history": 0, "smoking": 1, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.5, "bmi": 22.0, "exercise_hours": 5.0}),
    (14, "Conflicting Risk Factors", "35yo Obese Non-Smoker w/ Normal BP", {"age": 35, "sex": 0, "bp_systolic": 120, "bp_diastolic": 80, "cholesterol": 190, "triglycerides": 150, "heart_rate": 72, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 1, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.0, "bmi": 34.0, "exercise_hours": 2.0}),
    (15, "Conflicting Risk Factors", "70yo Fit Senior w/ High Cholesterol", {"age": 70, "sex": 0, "bp_systolic": 122, "bp_diastolic": 78, "cholesterol": 285, "triglycerides": 160, "heart_rate": 66, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 8.0, "bmi": 22.5, "exercise_hours": 4.5}),
    (16, "Conflicting Risk Factors", "29yo Diabetic w/ Active Lifestyle", {"age": 29, "sex": 1, "bp_systolic": 115, "bp_diastolic": 75, "cholesterol": 175, "triglycerides": 140, "heart_rate": 68, "diabetes": 1, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 7.5, "bmi": 21.5, "exercise_hours": 4.0}),

    # Category 5: Extreme Edge & Boundary Cases
    (17, "Extreme Edge / Boundary", "Extreme High Vitals Upper Boundary", {"age": 85, "sex": 1, "bp_systolic": 230, "bp_diastolic": 140, "cholesterol": 550, "triglycerides": 600, "heart_rate": 115, "diabetes": 1, "family_history": 1, "smoking": 1, "obesity": 1, "alcohol": 1, "medication": 1, "diet": 0, "previous_problems": 1, "sleep_hours": 3.0, "bmi": 45.0, "exercise_hours": 0.0}),
    (18, "Extreme Edge / Boundary", "Young Minimum Age Boundary (18yo)", {"age": 18, "sex": 0, "bp_systolic": 95, "bp_diastolic": 60, "cholesterol": 130, "triglycerides": 80, "heart_rate": 60, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 9.0, "bmi": 18.5, "exercise_hours": 6.0}),
    (19, "Extreme Edge / Boundary", "48yo Severe Sleep Deprivation (3h)", {"age": 48, "sex": 1, "bp_systolic": 145, "bp_diastolic": 92, "cholesterol": 245, "triglycerides": 210, "heart_rate": 84, "diabetes": 0, "family_history": 1, "smoking": 0, "obesity": 1, "alcohol": 1, "medication": 0, "diet": 0, "previous_problems": 0, "sleep_hours": 3.0, "bmi": 27.0, "exercise_hours": 0.0}),
    (20, "Extreme Edge / Boundary", "90yo Senior w/ Controlled Vitals", {"age": 90, "sex": 0, "bp_systolic": 125, "bp_diastolic": 80, "cholesterol": 195, "triglycerides": 135, "heart_rate": 68, "diabetes": 0, "family_history": 0, "smoking": 0, "obesity": 0, "alcohol": 0, "medication": 0, "diet": 1, "previous_problems": 0, "sleep_hours": 8.0, "bmi": 23.0, "exercise_hours": 3.0})
]

if __name__ == '__main__':
    tester = Test20CasesFlask()
    tester.setUp()
    print("=" * 110)
    print(f"{'CASE':<5} | {'CATEGORY':<25} | {'PATIENT PROFILE':<38} | {'RISK SCORE':<12} | {'PREDICTION':<22}")
    print("=" * 110)
    for cid, cat, prof, params in test_cases:
        res = tester.run_case(cid, cat, prof, params)
        print(f"#{cid:<4} | {cat:<25} | {prof:<38} | {res['health_score_pct']}% ({res['health_score']}) | {res['result']}")
        print(f"       Suggestions: {', '.join(res['suggestions'])}\n")
    print("=" * 110)
    print("ALL 20 TEST CASES EXECUTED SUCCESSFULLY VIA FLASK TEST CLIENT!")
