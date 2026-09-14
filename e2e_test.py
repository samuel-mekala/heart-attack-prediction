import urllib.request
import urllib.parse
import json

BASE_URL = "http://127.0.0.1:5001"

def run_e2e_tests():
    print("==========================================")
    print("  RUNNING END-TO-END SYSTEM VERIFICATION  ")
    print("==========================================")
    
    # 1. Test Home Page GET
    print("\n[1/3] Testing GET / (Home Page)...")
    req = urllib.request.Request(BASE_URL + "/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode('utf-8')
        assert "Heart Disease Detection & Advisor" in html
        assert "Patient Assessment Form" in html
        print("  --> GET / PASSED (HTTP 200 OK)")

    # 2. Test Low Risk Profile POST
    print("\n[2/3] Testing POST /predict (Low Risk Profile)...")
    low_risk_payload = {
        'age': '22', 'sex': '0', 'bp_systolic': '115', 'bp_diastolic': '75',
        'cholesterol': '160', 'triglycerides': '90', 'heart_rate': '68',
        'diabetes': '0', 'family_history': '0', 'smoking': '0', 'obesity': '0',
        'alcohol': '0', 'medication': '0', 'diet': '2', 'previous_problems': '0',
        'sleep_hours': '8', 'bmi': '21.5', 'exercise_hours': '5.0'
    }
    encoded_payload = urllib.parse.urlencode(low_risk_payload).encode('utf-8')
    req = urllib.request.Request(BASE_URL + "/predict", data=encoded_payload, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert 'result' in data
        assert 'health_score' in data
        assert 'suggestions' in data
        print(f"  --> Result: {data['result']}")
        print(f"  --> Health Score: {data['health_score']}")
        print(f"  --> Suggestions: {data['suggestions']}")
        print("  --> Low Risk POST /predict PASSED")

    # 3. Test High Risk Profile POST
    print("\n[3/3] Testing POST /predict (High Risk Profile)...")
    high_risk_payload = {
        'age': '65', 'sex': '1', 'bp_systolic': '160', 'bp_diastolic': '100',
        'cholesterol': '280', 'triglycerides': '350', 'heart_rate': '95',
        'diabetes': '1', 'family_history': '1', 'smoking': '1', 'obesity': '1',
        'alcohol': '1', 'medication': '1', 'diet': '0', 'previous_problems': '1',
        'sleep_hours': '5', 'bmi': '33.5', 'exercise_hours': '0.5'
    }
    encoded_payload = urllib.parse.urlencode(high_risk_payload).encode('utf-8')
    req = urllib.request.Request(BASE_URL + "/predict", data=encoded_payload, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode('utf-8'))
        assert 'result' in data
        assert 'health_score' in data
        assert 'suggestions' in data
        print(f"  --> Result: {data['result']}")
        print(f"  --> Health Score: {data['health_score']}")
        print(f"  --> Suggestions: {data['suggestions']}")
        print("  --> High Risk POST /predict PASSED")

    print("\n==========================================")
    print("  ALL END-TO-END TESTS PASSED SUCCESSFULLY! ")
    print("==========================================")

if __name__ == '__main__':
    run_e2e_tests()
