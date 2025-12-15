
"""
Unit tests for Task 5 - Model Training
"""

import unittest
import pandas as pd
import numpy as np
import joblib
import os

class TestModelTraining(unittest.TestCase):

    def setUp(self):
        """Load test data and model"""
        self.model_path = 'models/task5/best_model.pkl'
        self.data_path = 'data/processed/task5/X_test.csv'
        self.target_path = 'data/processed/task5/y_test.csv'

    def test_model_exists(self):
        """Test that the best model file exists"""
        self.assertTrue(os.path.exists(self.model_path), 
                       f"Model file not found at {self.model_path}")

    def test_data_exists(self):
        """Test that test data exists"""
        self.assertTrue(os.path.exists(self.data_path), 
                       f"Test data not found at {self.data_path}")
        self.assertTrue(os.path.exists(self.target_path), 
                       f"Test target not found at {self.target_path}")

    def test_model_loading(self):
        """Test that the model can be loaded"""
        if os.path.exists(self.model_path):
            model = joblib.load(self.model_path)
            self.assertIsNotNone(model, "Failed to load model")

    def test_prediction_shape(self):
        """Test that predictions have correct shape"""
        if os.path.exists(self.model_path) and os.path.exists(self.data_path):
            model = joblib.load(self.model_path)
            X_test = pd.read_csv(self.data_path)
            predictions = model.predict(X_test)
            self.assertEqual(len(predictions), len(X_test), 
                           "Predictions shape doesn't match test data")

    def test_feature_count(self):
        """Test that feature count is consistent"""
        if os.path.exists(self.data_path):
            X_test = pd.read_csv(self.data_path)
            # Assuming model expects 10 features based on Task 4
            self.assertEqual(X_test.shape[1], 10, 
                           f"Expected 10 features, got {X_test.shape[1]}")

if __name__ == '__main__':
    unittest.main()
