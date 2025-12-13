import pytest
import pandas as pd
import numpy as np
from src.data_processing import DatetimeFeatures, CustomerAggregation

def test_datetime_features():
    """Test that DatetimeFeatures extracts correct features"""
    # Create test data
    df = pd.DataFrame({
        'TransactionStartTime': pd.date_range('2023-01-01', periods=5, freq='D'),
        'Amount': [100, 200, 300, 400, 500]
    })
    
    transformer = DatetimeFeatures()
    result = transformer.fit_transform(df)
    
    assert 'transaction_hour' in result.columns
    assert 'transaction_day' in result.columns
    assert 'transaction_month' in result.columns
    assert 'transaction_year' in result.columns
    assert len(result) == 5

def test_customer_aggregation():
    """Test that CustomerAggregation creates RFM features"""
    df = pd.DataFrame({
        'CustomerId': ['A', 'A', 'B', 'B', 'B'],
        'Amount': [100, 200, 50, 150, 250],
        'TransactionStartTime': pd.date_range('2023-01-01', periods=5, freq='D')
    })
    
    transformer = CustomerAggregation()
    result = transformer.fit_transform(df)
    
    assert 'total_transaction_amount' in result.columns
    assert 'avg_transaction_amount' in result.columns
    assert 'transaction_count' in result.columns
    assert len(result) == 2  # Two unique customers