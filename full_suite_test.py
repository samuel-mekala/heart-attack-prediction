import urllib.request
import urllib.parse
import json

BASE_URL = "http://127.0.0.1:5001"

def test_profile(case_name, payload):
    encoded = urllib.parse.urlencode(payload).encode('utf-8')
    req = urllib.request.Request(BASE_URL + "/predict", data=encoded, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        print(f"\n==========================================")
        print(f" CASE: {case_name}")
        print(f"==========================================")
        print(f"  --> Prediction Result : {data['result']}")
        print(f"  --> Result Value Code : {data['result_val']} (0=No Risk, 1=Risk)")
        print(f"  --> Health Risk Score : {data['health_score']} / 1.00 ({data['health_score_pct']}%)")
        print(f"  --> Suggestions List  :")
        for s in data['suggestions']:
            print(f"      • {s}")
        return data

def run_full_suite():
    print("==================================================")
    print("  RUNNING FULL-SUITE SYSTEM TEST (BEST/AVG/WORST)  ")
    print("==================================================")

    # 1. BEST CASE: Optimal Health, Zero Risk Factors
    best_case = {
        'age': '18', 'sex': '0', 'bp_systolic': '110', 'bp_diastolic': '70',
        'cholesterol': '140', 'triglycerides': '70', 'heart_rate': '65',
        'diabetes': '0', 'family_history': '0', 'smoking': '0', 'obesity': '0',
        'alcohol': '0', 'medication': '0', 'diet': '2', 'previous_problems': '0',
        'sleep_hours': '8', 'bmi': '20.0', 'exercise_hours': '7.0'
    }
    d_best = test_profile("1. BEST CASE (Optimal Health, 18yo Female)", best_case)
    assert d_best['result_val'] == 0
    assert d_best['health_score'] <= 0.25

    # 2. AVERAGE CASE: Fit/Normal 45yo Male
    avg_case = {
        'age': '45', 'sex': '1', 'bp_systolic': '125', 'bp_diastolic': '80',
        'cholesterol': '195', 'triglycerides': '130', 'heart_rate': '72',
        'diabetes': '0', 'family_history': '0', 'smoking': '0', 'obesity': '0',
        'alcohol': '0', 'medication': '0', 'diet': '1', 'previous_problems': '0',
        'sleep_hours': '7', 'bmi': '24.0', 'exercise_hours': '3.0'
    }
    d_avg = test_profile("2. AVERAGE CASE (Fit 45yo Male, Normal Vitals)", avg_case)
    assert d_avg['result_val'] == 0
    assert 0.15 <= d_avg['health_score'] <= 0.40

    # 3. MODERATE RISK CASE: Overweight Hypertensive 52yo Male
    mod_case = {
        'age': '52', 'sex': '1', 'bp_systolic': '142', 'bp_diastolic': '92',
        'cholesterol': '235', 'triglycerides': '190', 'heart_rate': '80',
        'diabetes': '0', 'family_history': '1', 'smoking': '0', 'obesity': '1',
        'alcohol': '1', 'medication': '0', 'diet': '0', 'previous_problems': '0',
        'sleep_hours': '6', 'bmi': '28.5', 'exercise_hours': '1.0'
    }
    d_mod = test_profile("3. MODERATE RISK CASE (Hypertensive 52yo Male)", mod_case)
    assert d_mod['result_val'] == 1
    assert 0.45 <= d_mod['health_score'] <= 0.75

    # 4. WORST CASE: Severe Multi-Disease 75yo Male
    worst_case = {
        'age': '75', 'sex': '1', 'bp_systolic': '180', 'bp_diastolic': '110',
        'cholesterol': '350', 'triglycerides': '400', 'heart_rate': '105',
        'diabetes': '1', 'family_history': '1', 'smoking': '1', 'obesity': '1',
        'alcohol': '1', 'medication': '1', 'diet': '0', 'previous_problems': '1',
        'sleep_hours': '4', 'bmi': '38.0', 'exercise_hours': '0.0'
    }
    d_worst = test_profile("4. WORST CASE (Severe 75yo Diabetic Smoker)", worst_case)
    assert d_worst['result_val'] == 1
    assert d_worst['health_score'] >= 0.85

    print("\n==================================================")
    print("  ALL FULL-SUITE TESTS PASSED WITH 100% ACCURACY! ")
    print("==================================================")

if __name__ == '__main__':
    run_full_suite()
