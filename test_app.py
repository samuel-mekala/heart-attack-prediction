import unittest
from deploy import app, generate_suggestions

class HeartPredictionTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Heart Disease Detection', response.data)

    def test_prediction_post(self):
        # 18-feature payload from report mockup
        payload = {
            'age': '20',
            'sex': '1',
            'bp_systolic': '120',
            'bp_diastolic': '90',
            'cholesterol': '80',
            'triglycerides': '100',
            'heart_rate': '72',
            'diabetes': '0',
            'family_history': '0',
            'smoking': '0',
            'obesity': '1',
            'alcohol': '1',
            'medication': '0',
            'diet': '0',
            'previous_problems': '1',
            'sleep_hours': '10',
            'bmi': '29.5',
            'exercise_hours': '0'
        }
        response = self.app.post('/predict', data=payload, headers={'Accept': 'application/json'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('result', data)
        self.assertIn('health_score', data)
        self.assertIn('suggestions', data)
        print("\nTest 18-Feature Prediction Response:")
        print(data)

    def test_suggestions_engine(self):
        params = {'bmi': 29.5, 'exercise_hours': 0, 'diet': 0, 'alcohol': 1}
        suggestions = generate_suggestions(params, result_val=1, risk_score=0.48)
        self.assertTrue(len(suggestions) > 0)
        self.assertIn("lose weight", suggestions)
        self.assertIn("do more exercise", suggestions)
        self.assertIn("eat healthy food", suggestions)
        self.assertIn("try reducing alcohol", suggestions)

if __name__ == '__main__':
    unittest.main()
