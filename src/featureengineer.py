# src/featureengineer.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for creating new features before preprocessing."""
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Drop irrelevant columns
        drop_cols = ['Unnamed: 0', 'Customer Name', 'Customer_Address']
        X = X.drop(columns=[col for col in drop_cols if col in X.columns], errors="ignore")

        # Custom feature
        if 'Category Premium' in X.columns and 'BMI' in X.columns:
            X['Premium_to_BMI'] = X['Category Premium'] / (X['BMI'].replace(0, np.nan))
            # Add tiny noise to avoid perfect memorization
            X['Premium_to_BMI'] += np.random.normal(0, 0.01, size=len(X))
            X['Premium_to_BMI'] = X['Premium_to_BMI'].fillna(0)

        # High-cardinality encoding for Company Name
        if 'Company Name' in X.columns:
            freq = X['Company Name'].value_counts() / len(X)
            X['CompanyName_FE'] = X['Company Name'].map(freq)
            X = X.drop(columns=['Company Name'])

        return X
