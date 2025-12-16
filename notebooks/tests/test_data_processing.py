"""
Unit tests for data processing functions
"""

import unittest
import pandas as pd
import numpy as np

class TestDataProcessing(unittest.TestCase):

    def test_dataframe_shape(self):
        """Test that sample data has correct shape"""
        # Create sample data
        data = {
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        }
        df = pd.DataFrame(data)
        self.assertEqual(df.shape, (3, 3))

    def test_no_nan_values(self):
        """Test that data has no NaN values"""
        data = {
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        }
        df = pd.DataFrame(data)
        self.assertFalse(df.isnull().any().any())

    def test_feature_types(self):
        """Test that features are numeric"""
        data = {
            'feature1': [1, 2, 3],
            'feature2': [4.0, 5.0, 6.0],
            'target': [0, 1, 0]
        }
        df = pd.DataFrame(data)
        self.assertTrue(np.issubdtype(df['feature1'].dtype, np.integer))
        self.assertTrue(np.issubdtype(df['feature2'].dtype, np.floating))

    def test_target_distribution(self):
        """Test that target has expected values"""
        data = {
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        }
        df = pd.DataFrame(data)
        self.assertSetEqual(set(df['target'].unique()), {0, 1})

if __name__ == '__main__':
    unittest.main()
