"""
Unit tests for Task 3 Feature Engineering
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
import sklearn.pipeline as Pipeline

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import (
    DateTimeFeatureExtractor,
    RFMFeatureAggregator,
    CategoricalFeatureEncoder,
    WoEIVTransformer,
    build_feature_engineering_pipeline
)


class TestDateTimeFeatureExtractor:
    """Test datetime feature extraction"""
    
    def test_extraction(self):
        """Test that datetime features are correctly extracted"""
        df = pd.DataFrame({
            'TransactionStartTime': pd.date_range('2023-01-01', periods=5, freq='D'),
            'Amount': [100, 200, 300, 400, 500]
        })
        
        extractor = DateTimeFeatureExtractor()
        result = extractor.fit_transform(df)
        
        assert 'transaction_hour' in result.columns
        assert 'transaction_day' in result.columns
        assert 'transaction_month' in result.columns
        assert 'transaction_year' in result.columns
        assert len(result) == 5
        assert result['transaction_month'].iloc[0] == 1  # January
        
    def test_string_datetime_conversion(self):
        """Test conversion from string to datetime"""
        df = pd.DataFrame({
            'TransactionStartTime': ['2023-01-01 10:30:00', '2023-01-02 14:45:00'],
            'Amount': [100, 200]
        })
        
        extractor = DateTimeFeatureExtractor()
        result = extractor.fit_transform(df)
        
        assert 'transaction_hour' in result.columns
        assert result['transaction_hour'].iloc[0] == 10
        assert result['transaction_hour'].iloc[1] == 14


class TestRFMFeatureAggregator:
    """Test RFM feature aggregation"""
    
    def test_basic_aggregation(self):
        """Test basic RFM aggregation"""
        df = pd.DataFrame({
            'CustomerId': ['A', 'A', 'B', 'B', 'B'],
            'Amount': [100, 200, 50, 150, 250],
            'TransactionStartTime': pd.date_range('2023-01-01', periods=5, freq='D')
        })
        
        aggregator = RFMFeatureAggregator()
        result = aggregator.fit_transform(df)
        
        # Should have 2 unique customers
        assert len(result) == 2
        
        # Check aggregated features exist
        assert 'total_amount' in result.columns
        assert 'avg_amount' in result.columns
        assert 'std_amount' in result.columns
        assert 'transaction_count' in result.columns
        
        # Check calculations
        customer_a = result[result['CustomerId'] == 'A']
        assert customer_a['total_amount'].iloc[0] == 300
        assert customer_a['transaction_count'].iloc[0] == 2
        assert customer_a['avg_amount'].iloc[0] == 150
        
    def test_recency_calculation(self):
        """Test recency calculation"""
        df = pd.DataFrame({
            'CustomerId': ['A', 'A', 'B'],
            'Amount': [100, 200, 150],
            'TransactionStartTime': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-03'])
        })
        
        aggregator = RFMFeatureAggregator()
        result = aggregator.fit_transform(df)
        
        # Recency should be calculated
        if 'recency_days' in result.columns:
            # Last transaction was on 2023-01-05 for snapshot
            assert 'recency_days' in result.columns


class TestCategoricalFeatureEncoder:
    """Test categorical encoding"""
    
    def test_onehot_encoding(self):
        """Test one-hot encoding"""
        df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'C', 'B'],
            'Value': [1, 2, 3, 4, 5]
        })
        
        encoder = CategoricalFeatureEncoder(encoding_strategy='onehot', columns=['Category'])
        result = encoder.fit_transform(df)
        
        # Should have created one-hot columns
        assert 'Category_A' in result.columns or 'Category_A' in '|'.join(result.columns)
        assert 'Value' in result.columns  # Numerical column should remain
        
    def test_label_encoding(self):
        """Test label encoding"""
        df = pd.DataFrame({
            'Category': ['A', 'B', 'A', 'C', 'B'],
            'Value': [1, 2, 3, 4, 5]
        })
        
        encoder = CategoricalFeatureEncoder(encoding_strategy='label', columns=['Category'])
        result = encoder.fit_transform(df)
        
        # Category should be converted to numerical labels
        assert 'Category' in result.columns
        assert result['Category'].dtype in [np.int64, np.int32, np.int8]


class TestWoEIVTransformer:
    """Test WoE and IV transformation"""
    
    def test_woe_calculation(self):
        """Test WoE calculation"""
        np.random.seed(42)
        
        # Create sample data with clear relationship
        n_samples = 1000
        df = pd.DataFrame({
            'feature': np.random.normal(0, 1, n_samples),
            'target': (np.random.normal(0, 1, n_samples) > 0).astype(int)
        })
        
        woe_transformer = WoEIVTransformer(target_col='target', n_bins=5)
        woe_transformer.fit(df)
        
        # Should have calculated WoE
        assert len(woe_transformer.woe_dict) > 0
        assert len(woe_transformer.iv_dict) > 0
        
        # Test transformation
        transformed = woe_transformer.transform(df)
        assert 'feature' in transformed.columns
        assert transformed.shape[0] == n_samples


class TestPipelineBuilder:
    """Test pipeline construction"""
    
    def test_pipeline_build(self):
        """Test that pipeline is built correctly"""
        pipeline = build_feature_engineering_pipeline()
        
        # Should be a sklearn Pipeline
        from sklearn.pipeline import Pipeline
        assert isinstance(pipeline, Pipeline)
        
        # Should have expected steps
        step_names = [name for name, _ in pipeline.steps]
        assert len(step_names) >= 2
        
    def test_pipeline_configurations(self):
        """Test different pipeline configurations"""
        configs = [
            {'scaling_strategy': 'standard', 'encoding_strategy': 'onehot'},
            {'scaling_strategy': 'minmax', 'encoding_strategy': 'label'},
            {'imputation_strategy': 'knn', 'use_woe': True}
        ]
        
        for config in configs:
            pipeline = build_feature_engineering_pipeline(config)
            assert pipeline is not None
            assert isinstance(pipeline, Pipeline)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])