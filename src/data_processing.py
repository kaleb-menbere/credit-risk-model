"""
Data Processing Pipeline for Credit Risk Model
Author: Bati Bank Analytics Engineering Team
Date: December 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ============================================================================
# 1. Datetime Feature Extraction
# ============================================================================

class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """
    Extract temporal features from transaction timestamp.
    """
    
    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(X[self.datetime_col]):
            X[self.datetime_col] = pd.to_datetime(X[self.datetime_col])
        
        # Extract temporal features
        X["transaction_hour"] = X[self.datetime_col].dt.hour
        X["transaction_day"] = X[self.datetime_col].dt.day
        X["transaction_month"] = X[self.datetime_col].dt.month
        X["transaction_year"] = X[self.datetime_col].dt.year
        X["transaction_dayofweek"] = X[self.datetime_col].dt.dayofweek
        X["transaction_weekend"] = X[self.datetime_col].dt.dayofweek.isin([5, 6]).astype(int)
        
        return X

# ============================================================================
# 2. Customer Aggregation
# ============================================================================

class CustomerAggregation(BaseEstimator, TransformerMixin):
    """
    Aggregate transaction-level data to customer-level RFMS features.
    """
    
    def __init__(self, customer_id_col="CustomerId", amount_col="Amount"):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        
        # Ensure numeric amount
        X[self.amount_col] = pd.to_numeric(X[self.amount_col], errors='coerce')
        
        # Define snapshot date for recency calculation
        snapshot_date = X["TransactionStartTime"].max()
        
        # Group by customer
        agg_dict = {
            self.amount_col: [
                ("total_transaction_amount", "sum"),
                ("avg_transaction_amount", "mean"),
                ("std_transaction_amount", "std"),
                ("min_transaction_amount", "min"),
                ("max_transaction_amount", "max")
            ],
            "TransactionStartTime": [
                ("transaction_count", "count"),
                ("first_transaction_date", "min"),
                ("last_transaction_date", "max")
            ]
        }
        
        # Create multi-level aggregation
        customer_agg = X.groupby(self.customer_id_col).agg(agg_dict)
        
        # Flatten column names
        customer_agg.columns = ['_'.join(col).strip() for col in customer_agg.columns.values]
        
        # Calculate recency (days since last transaction)
        customer_agg["recency_days"] = (snapshot_date - customer_agg["TransactionStartTime_last_transaction_date"]).dt.days
        
        # Calculate customer tenure (days since first transaction)
        customer_agg["customer_tenure_days"] = (
            customer_agg["TransactionStartTime_last_transaction_date"] - 
            customer_agg["TransactionStartTime_first_transaction_date"]
        ).dt.days
        
        # Calculate frequency (transactions per day of tenure)
        customer_agg["frequency_per_day"] = (
            customer_agg["transaction_count"] / 
            np.maximum(customer_agg["customer_tenure_days"], 1)
        )
        
        # Drop original datetime columns
        customer_agg = customer_agg.drop([
            "TransactionStartTime_first_transaction_date",
            "TransactionStartTime_last_transaction_date"
        ], axis=1)
        
        # Reset index
        customer_agg = customer_agg.reset_index()
        
        return customer_agg

# ============================================================================
# 3. Feature Engineering Pipeline Builder
# ============================================================================

def build_feature_pipeline(numerical_features, categorical_features):
    """
    Build a scikit-learn pipeline for feature processing.
    
    Parameters:
    -----------
    numerical_features : list
        List of numerical feature names
    categorical_features : list
        List of categorical feature names
        
    Returns:
    --------
    pipeline : ColumnTransformer
        Complete feature processing pipeline
    """
    
    # Numerical feature pipeline
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical feature pipeline - FIXED VERSION
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # CORRECTED: No sparse parameter in newer sklearn versions
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )
    
    return preprocessor

# ============================================================================
# 4. Alternative simplified pipeline builder (if above still fails)
# ============================================================================

def build_simple_pipeline(numerical_features):
    """
    Simplified pipeline for only numerical features.
    Use this if the main pipeline fails.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    return pipeline

# ============================================================================
# 5. WoE and IV Transformation (Optional for Logistic Regression)
# ============================================================================

class WoETransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence transformation for categorical/binned features.
    Requires the xverse package: pip install xverse
    """
    
    def __init__(self, target_col="is_high_risk"):
        self.target_col = target_col
        self.woe_dict = {}
        
    def fit(self, X, y=None):
        # This requires the xverse package
        try:
            from xverse.transformer import WOE
            self.woe_transformer = WOE()
            
            # If y is provided separately
            if y is not None:
                self.woe_transformer.fit(X, y)
            # If target is in X
            elif self.target_col in X.columns:
                self.woe_transformer.fit(X.drop(columns=[self.target_col]), X[self.target_col])
                
        except ImportError:
            print("Warning: xverse package not installed. Install with: pip install xverse")
            self.woe_transformer = None
            
        return self
    
    def transform(self, X, y=None):
        if self.woe_transformer is None:
            return X
            
        if self.target_col in X.columns:
            X_transformed = self.woe_transformer.transform(X.drop(columns=[self.target_col]))
            X_transformed[self.target_col] = X[self.target_col]
            return X_transformed
        else:
            return self.woe_transformer.transform(X)