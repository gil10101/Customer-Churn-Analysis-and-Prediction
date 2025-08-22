"""
Unit tests for Feature Engineering module.

This module tests all components of the FeatureEngineer class
to ensure correct implementation of feature engineering requirements.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import warnings

# Suppress warnings for cleaner test output
warnings.filterwarnings("ignore")

# Import the module under test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.feature_engineering import FeatureEngineer
from utils.config import FeatureEngineeringConfig


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample customer data for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'customerID': [f'CUST_{i:04d}' for i in range(n_samples)],
            'tenure': np.random.randint(1, 72, n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(100, 8000, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n_samples),
            'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def feature_engineer(self):
        """Create FeatureEngineer instance for testing."""
        config = FeatureEngineeringConfig()
        return FeatureEngineer(config)
    
    def test_initialization(self, feature_engineer):
        """Test proper initialization of FeatureEngineer."""
        assert feature_engineer.config is not None
        assert feature_engineer.scaler is not None
        assert feature_engineer.label_encoders == {}
        assert feature_engineer.target_encoders == {}
        assert feature_engineer.feature_names == []
        assert feature_engineer.skewed_features == []
    
    def test_create_usage_ratio_features(self, feature_engineer, sample_data):
        """Test creation of usage ratio features (Requirement 1.1)."""
        result = feature_engineer.create_usage_ratio_features(sample_data)
        
        # Check that new features are created
        expected_features = [
            'usage_efficiency_ratio', 'value_per_service_ratio', 
            'cost_per_gb_ratio', 'ActiveServicesCount'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Feature {feature} not created"
        
        # Check that ratios are calculated correctly
        assert result['usage_efficiency_ratio'].notna().all()
        assert result['value_per_service_ratio'].notna().all()
        assert result['cost_per_gb_ratio'].notna().all()
        assert result['ActiveServicesCount'].min() >= 1  # Should be at least 1
        
        # Check for no infinite values
        assert np.isfinite(result['usage_efficiency_ratio']).all()
        assert np.isfinite(result['value_per_service_ratio']).all()
        assert np.isfinite(result['cost_per_gb_ratio']).all()
    
    def test_create_engagement_trend_features(self, feature_engineer, sample_data):
        """Test creation of engagement trend features (Requirement 1.2)."""
        result = feature_engineer.create_engagement_trend_features(sample_data)
        
        expected_features = [
            'engagement_score', 'engagement_decline_flag', 'usage_trend_slope'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Feature {feature} not created"
        
        # Check data types and ranges
        assert result['engagement_decline_flag'].dtype == int
        assert result['engagement_decline_flag'].isin([0, 1]).all()
        assert result['engagement_score'].notna().all()
        assert result['usage_trend_slope'].notna().all()
    
    def test_create_tenure_band_features(self, feature_engineer, sample_data):
        """Test creation of tenure band features (Requirement 1.3)."""
        result = feature_engineer.create_tenure_band_features(sample_data)
        
        expected_features = ['tenure_band', 'tenure_risk_score']
        
        for feature in expected_features:
            assert feature in result.columns, f"Feature {feature} not created"
        
        # Check that tenure bands are created correctly
        assert result['tenure_band'].notna().all()
        assert result['tenure_risk_score'].between(0, 1).all()
        
        # Check that tenure bands correspond to configured bins
        unique_bands = result['tenure_band'].unique()
        assert len(unique_bands) <= len(feature_engineer.config.tenure_bins)
    
    def test_create_interaction_features(self, feature_engineer, sample_data):
        """Test creation of interaction and support features (Requirement 1.4)."""
        result = feature_engineer.create_interaction_features(sample_data)
        
        expected_features = [
            'support_tickets_per_month', 'avg_resolution_time',
            'interaction_intensity', 'support_efficiency_score'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Feature {feature} not created"
        
        # Check reasonable ranges
        assert result['support_tickets_per_month'].min() >= 0
        assert result['avg_resolution_time'].between(0.5, 24).all()
        assert result['interaction_intensity'].min() >= 0
        assert result['support_efficiency_score'].min() > 0
    
    def test_create_billing_behavior_features(self, feature_engineer, sample_data):
        """Test creation of billing behavior features (Requirement 1.5)."""
        result = feature_engineer.create_billing_behavior_features(sample_data)
        
        expected_features = [
            'payment_method_risk_score', 'late_payment_frequency',
            'payment_reliability_score', 'auto_pay_enabled', 'billing_complexity_score'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Feature {feature} not created"
        
        # Check ranges and data types
        assert result['payment_method_risk_score'].between(0, 1).all()
        assert result['late_payment_frequency'].between(0, 1).all()
        assert result['payment_reliability_score'].between(0, 1).all()
        assert result['auto_pay_enabled'].isin([0, 1]).all()
        assert result['billing_complexity_score'].min() >= 0
    
    def test_apply_target_encoding(self, feature_engineer, sample_data):
        """Test target encoding with cross-validation (Requirement 1.6)."""
        result = feature_engineer.apply_target_encoding(sample_data, target_col='Churn')
        
        # Check that target encoded features are created
        categorical_cols = ['Contract', 'PaymentMethod', 'InternetService']
        
        for col in categorical_cols:
            encoded_col = f"{col}_target_encoded"
            if encoded_col in result.columns:
                assert result[encoded_col].notna().all()
                assert result[encoded_col].dtype == float
        
        # Check that encoders are stored
        assert len(feature_engineer.target_encoders) > 0
    
    def test_create_polynomial_features(self, feature_engineer, sample_data):
        """Test creation of polynomial and interaction features (Requirement 1.7)."""
        result = feature_engineer.create_polynomial_features(sample_data, degree=2, interaction_only=True)
        
        # Check that polynomial features are created
        poly_features = [col for col in result.columns if col.startswith('poly_')]
        assert len(poly_features) > 0
        
        # Check that polynomial transformer is stored
        assert feature_engineer.polynomial_features is not None
        
        # Check for no infinite or NaN values in polynomial features
        for feature in poly_features:
            assert np.isfinite(result[feature]).all()
    
    def test_apply_log_transformation(self, feature_engineer, sample_data):
        """Test log transformation of skewed features (Requirement 1.8)."""
        # Create some skewed data
        sample_data['skewed_feature'] = np.random.exponential(2, len(sample_data))
        
        result = feature_engineer.apply_log_transformation(sample_data, skewness_threshold=1.0)
        
        # Check that log-transformed features are created for skewed data
        log_features = [col for col in result.columns if col.endswith('_log')]
        
        if len(log_features) > 0:
            for feature in log_features:
                assert result[feature].notna().all()
                assert np.isfinite(result[feature]).all()
        
        # Check that skewed features list is populated
        assert isinstance(feature_engineer.skewed_features, list)
    
    def test_transform_pipeline(self, feature_engineer, sample_data):
        """Test complete feature engineering pipeline."""
        result = feature_engineer.transform_pipeline(sample_data, target_col='Churn')
        
        # Check that result has more columns than input
        assert len(result.columns) > len(sample_data.columns)
        
        # Check that feature names are stored
        assert len(feature_engineer.feature_names) > 0
        
        # Check that no NaN values are introduced in key features
        key_features = ['usage_efficiency_ratio', 'tenure_risk_score', 'payment_method_risk_score']
        for feature in key_features:
            if feature in result.columns:
                assert result[feature].notna().all()
    
    def test_get_feature_names(self, feature_engineer, sample_data):
        """Test getting feature names after transformation."""
        feature_engineer.transform_pipeline(sample_data)
        feature_names = feature_engineer.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
    
    def test_get_feature_importance_mapping(self, feature_engineer):
        """Test getting feature importance mapping."""
        mapping = feature_engineer.get_feature_importance_mapping()
        
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        
        # Check that key features are in the mapping
        key_features = ['usage_efficiency_ratio', 'tenure_risk_score', 'payment_method_risk_score']
        for feature in key_features:
            assert feature in mapping
    
    def test_get_skewed_features(self, feature_engineer, sample_data):
        """Test getting list of skewed features."""
        # Add skewed data
        sample_data['very_skewed'] = np.random.exponential(5, len(sample_data))
        
        feature_engineer.apply_log_transformation(sample_data, skewness_threshold=0.5)
        skewed_features = feature_engineer.get_skewed_features()
        
        assert isinstance(skewed_features, list)
    
    def test_get_target_encoders(self, feature_engineer, sample_data):
        """Test getting target encoders."""
        feature_engineer.apply_target_encoding(sample_data, target_col='Churn')
        encoders = feature_engineer.get_target_encoders()
        
        assert isinstance(encoders, dict)
    
    def test_edge_cases(self, feature_engineer):
        """Test edge cases and error handling."""
        # Test with empty dataframe
        empty_df = pd.DataFrame()
        result = feature_engineer.create_usage_ratio_features(empty_df)
        assert len(result) == 0
        
        # Test with missing columns
        minimal_df = pd.DataFrame({
            'MonthlyCharges': [50.0, 75.0],
            'tenure': [12, 24]
        })
        result = feature_engineer.create_usage_ratio_features(minimal_df)
        assert len(result) == 2
        assert 'usage_efficiency_ratio' in result.columns
    
    def test_configuration_usage(self):
        """Test that configuration is properly used."""
        custom_config = FeatureEngineeringConfig(
            tenure_bins=[0, 12, 24, 48],
            default_usage_minutes=100.0
        )
        
        feature_engineer = FeatureEngineer(custom_config)
        assert feature_engineer.config.tenure_bins == [0, 12, 24, 48]
        assert feature_engineer.config.default_usage_minutes == 100.0


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])

