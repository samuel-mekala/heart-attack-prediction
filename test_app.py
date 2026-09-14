import unittest
from deploy import app, generate_suggestions

class HeartPredictionTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Heart Attack Prediction', response.data)
        self.assertIn(b'Patient Attributes Form', response.data)

    def test_prediction_post(self):
        # Sample high-risk payload
        payload = {
            'age': '60',
            'sex': '1',
            'cp': '0',
            'trestbps': '150',
            'chol': '280',
            'fbs': '1',
            'restecg': '1',
            'thalach': '110',
            'exang': '1',
            'oldpeak': '2.5',
            'slope': '1',
            'ca': '2',
            'thal': '3'
        }
        response = self.app.post('/predict', data=payload, headers={'Accept': 'application/json'})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('result', data)
        self.assertIn('health_score', data)
        self.assertIn('suggestions', data)
        print("\nTest Prediction Payload Response:")
        print(data)

    def test_suggestions_engine(self):
        params = {'chol': 250, 'trestbps': 140, 'exang': 1, 'oldpeak': 2.0, 'thalach': 100, 'fbs': 1}
        suggestions = generate_suggestions(params, result_val=1, risk_score=0.85)
        self.assertTrue(len(suggestions) > 0)
        self.assertIn("lose weight", suggestions)
        self.assertIn("do more exercise", suggestions)

if __name__ == '__main__':
    unittest.main()
