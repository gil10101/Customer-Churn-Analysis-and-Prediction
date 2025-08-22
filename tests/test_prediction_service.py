"""
Unit tests for the Prediction Service module.

Tests cover data models, model registry, prediction service functionality,
and error handling scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
import json
import pickle
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from utils.prediction_service import (
    CustomerInput, PredictionResult, ModelRegistry, PredictionService,
    create_sample_customer, quick_prediction
)


class TestCustomerInput:
    """Test cases for CustomerInput data model."""
    
    def test_valid_customer_input_creation(self):
        """Test creating a valid CustomerInput instance."""
        customer = CustomerInput(
            customer_id="TEST_001",
            gender="Female",
            senior_citizen=False,
            partner=True,
            dependents=False,
            tenure=24,
            contract="One year",
            paperless_billing=True,
            payment_method="Credit card (automatic)",
            phone_service=True,
            multiple_lines="No",
            internet_service="Fiber optic",
            online_security="Yes",
            online_backup="No",
            device_protection="Yes",
            tech_support="No",
            streaming_tv="Yes",
            streaming_movies="Yes",
            monthly_charges=75.50,
            total_charges=1815.00
        )
        
        assert customer.customer_id == "TEST_001"
        assert customer.gender == "Female"
        assert customer.tenure == 24
        assert customer.monthly_charges == 75.50
        assert customer.prediction_timestamp is not None
    
    def test_customer_input_with_optional_fields(self):
        """Test CustomerInput with optional enhanced features."""
        customer = CustomerInput(
            customer_id="TEST_002",
            gender="Male",
            senior_citizen=True,
            partner=False,
            dependents=True,
            tenure=12,
            contract="Month-to-month",
            paperless_billing=False,
            payment_method="Electronic check",
            phone_service=True,
            multiple_lines="Yes",
            internet_service="DSL",
            online_security="No",
            online_backup="Yes",
            device_protection="No",
            tech_support="Yes",
            streaming_tv="No",
            streaming_movies="No",
            monthly_charges=45.20,
            total_charges=542.40,
            usage_minutes_monthly=320.5,
            data_usage_gb_monthly=8.2,
            support_interactions_count=3,
            complaint_count=1,
            satisfaction_score=6.5,
            payment_delay_frequency=0.1,
            service_change_frequency=0.2
        )
        
        assert customer.usage_minutes_monthly == 320.5
        assert customer.support_interactions_count == 3
        assert customer.satisfaction_score == 6.5
    
    def test_customer_input_validation_errors(self):
        """Test validation errors for invalid input."""
        # Test empty customer_id
        with pytest.raises(ValueError, match="customer_id is required"):
            CustomerInput(
                customer_id="",
                gender="Female",
                senior_citizen=False,
                partner=True,
                dependents=False,
                tenure=24,
                contract="One year",
                paperless_billing=True,
                payment_method="Credit card (automatic)",
                phone_service=True,
                multiple_lines="No",
                internet_service="Fiber optic",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="Yes",
                streaming_movies="Yes",
                monthly_charges=75.50,
                total_charges=1815.00
            )
        
        # Test invalid contract
        with pytest.raises(ValueError, match="contract must be one of"):
            CustomerInput(
                customer_id="TEST_001",
                gender="Female",
                senior_citizen=False,
                partner=True,
                dependents=False,
                tenure=24,
                contract="Invalid Contract",
                paperless_billing=True,
                payment_method="Credit card (automatic)",
                phone_service=True,
                multiple_lines="No",
                internet_service="Fiber optic",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="Yes",
                streaming_movies="Yes",
                monthly_charges=75.50,
                total_charges=1815.00
            )
        
        # Test negative tenure
        with pytest.raises(ValueError, match="tenure must be non-negative"):
            CustomerInput(
                customer_id="TEST_001",
                gender="Female",
                senior_citizen=False,
                partner=True,
                dependents=False,
                tenure=-5,
                contract="One year",
                paperless_billing=True,
                payment_method="Credit card (automatic)",
                phone_service=True,
                multiple_lines="No",
                internet_service="Fiber optic",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="Yes",
                streaming_movies="Yes",
                monthly_charges=75.50,
                total_charges=1815.00
            )
        
        # Test invalid satisfaction score
        with pytest.raises(ValueError, match="satisfaction_score must be between 0 and 10"):
            CustomerInput(
                customer_id="TEST_001",
                gender="Female",
                senior_citizen=False,
                partner=True,
                dependents=False,
                tenure=24,
                contract="One year",
                paperless_billing=True,
                payment_method="Credit card (automatic)",
                phone_service=True,
                multiple_lines="No",
                internet_service="Fiber optic",
                online_security="Yes",
                online_backup="No",
                device_protection="Yes",
                tech_support="No",
                streaming_tv="Yes",
                streaming_movies="Yes",
                monthly_charges=75.50,
                total_charges=1815.00,
                satisfaction_score=15.0
            )
    
    def test_customer_input_serialization(self):
        """Test CustomerInput to_dict and from_dict methods."""
        customer = create_sample_customer()
        customer_dict = customer.to_dict()
        
        # Check that all fields are present
        assert "customer_id" in customer_dict
        assert "gender" in customer_dict
        assert "monthly_charges" in customer_dict
        assert "prediction_timestamp" in customer_dict
        
        # Test round-trip conversion
        customer_restored = CustomerInput.from_dict(customer_dict)
        assert customer_restored.customer_id == customer.customer_id
        assert customer_restored.monthly_charges == customer.monthly_charges
        assert customer_restored.usage_minutes_monthly == customer.usage_minutes_monthly


class TestPredictionResult:
    """Test cases for PredictionResult data model."""
    
    def test_valid_prediction_result_creation(self):
        """Test creating a valid PredictionResult instance."""
        result = PredictionResult(
            customer_id="TEST_001",
            churn_probability=0.75,
            risk_level="high",
            confidence_score=0.85,
            key_risk_factors=["Month-to-month contract", "High monthly charges"],
            recommendations=["Offer discount", "Upgrade to annual contract"],
            model_version="v1.0",
            prediction_timestamp=datetime.now().isoformat()
        )
        
        assert result.customer_id == "TEST_001"
        assert result.churn_probability == 0.75
        assert result.risk_level == "high"
        assert result.confidence_score == 0.85
        assert len(result.key_risk_factors) == 2
        assert len(result.recommendations) == 2
    
    def test_prediction_result_with_optional_fields(self):
        """Test PredictionResult with optional fields."""
        result = PredictionResult(
            customer_id="TEST_002",
            churn_probability=0.45,
            risk_level="medium",
            confidence_score=0.70,
            key_risk_factors=["New customer"],
            recommendations=["Monitor engagement"],
            model_version="v1.1",
            prediction_timestamp=datetime.now().isoformat(),
            risk_score=45.0,
            risk_percentile=65.0,
            feature_contributions={"tenure": -0.2, "monthly_charges": 0.3},
            estimated_clv=1200.0,
            retention_cost_benefit=150.0
        )
        
        assert result.risk_score == 45.0
        assert result.risk_percentile == 65.0
        assert result.estimated_clv == 1200.0
        assert "tenure" in result.feature_contributions
    
    def test_prediction_result_validation_errors(self):
        """Test validation errors for invalid prediction results."""
        # Test invalid churn probability
        with pytest.raises(ValueError, match="churn_probability must be between 0 and 1"):
            PredictionResult(
                customer_id="TEST_001",
                churn_probability=1.5,
                risk_level="high",
                confidence_score=0.85,
                key_risk_factors=[],
                recommendations=[],
                model_version="v1.0",
                prediction_timestamp=datetime.now().isoformat()
            )
        
        # Test invalid risk level
        with pytest.raises(ValueError, match="risk_level must be one of"):
            PredictionResult(
                customer_id="TEST_001",
                churn_probability=0.75,
                risk_level="extreme",
                confidence_score=0.85,
                key_risk_factors=[],
                recommendations=[],
                model_version="v1.0",
                prediction_timestamp=datetime.now().isoformat()
            )
        
        # Test invalid confidence score
        with pytest.raises(ValueError, match="confidence_score must be between 0 and 1"):
            PredictionResult(
                customer_id="TEST_001",
                churn_probability=0.75,
                risk_level="high",
                confidence_score=1.2,
                key_risk_factors=[],
                recommendations=[],
                model_version="v1.0",
                prediction_timestamp=datetime.now().isoformat()
            )
    
    def test_prediction_result_serialization(self):
        """Test PredictionResult to_dict and from_dict methods."""
        result = PredictionResult(
            customer_id="TEST_001",
            churn_probability=0.75,
            risk_level="high",
            confidence_score=0.85,
            key_risk_factors=["Factor 1", "Factor 2"],
            recommendations=["Rec 1", "Rec 2"],
            model_version="v1.0",
            prediction_timestamp=datetime.now().isoformat(),
            estimated_clv=1500.0
        )
        
        result_dict = result.to_dict()
        
        # Check that all fields are present
        assert "customer_id" in result_dict
        assert "churn_probability" in result_dict
        assert "estimated_clv" in result_dict
        
        # Test round-trip conversion
        result_restored = PredictionResult.from_dict(result_dict)
        assert result_restored.customer_id == result.customer_id
        assert result_restored.churn_probability == result.churn_probability
        assert result_restored.estimated_clv == result.estimated_clv


class TestModelRegistry:
    """Test cases for ModelRegistry."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_path = Path(self.temp_dir)
        self.registry = ModelRegistry(self.models_path)
        
        # Create a simple test model
        self.test_model = LogisticRegression(random_state=42)
        self.test_model.fit([[1, 2], [3, 4], [5, 6]], [0, 1, 0])
        
        self.test_metadata = {
            "version": "v1.0",
            "created_date": "2024-01-01",
            "feature_names": ["feature1", "feature2"],
            "performance_metrics": {"accuracy": 0.85, "auc": 0.90}
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_save_and_load_model(self):
        """Test saving and loading a model."""
        # Save model
        self.registry.save_model(
            self.test_model,
            "test_model",
            "v1.0",
            self.test_metadata
        )
        
        # Check files were created
        model_file = self.models_path / "test_model_v1.0.pkl"
        metadata_file = self.models_path / "test_model_v1.0_metadata.json"
        
        assert model_file.exists()
        assert metadata_file.exists()
        
        # Load model
        loaded_model, loaded_metadata = self.registry.load_model("test_model", "v1.0")
        
        # Verify model functionality
        test_prediction = loaded_model.predict([[2, 3]])
        original_prediction = self.test_model.predict([[2, 3]])
        assert test_prediction[0] == original_prediction[0]
        
        # Verify metadata
        assert loaded_metadata["version"] == "v1.0"
        assert loaded_metadata["feature_names"] == ["feature1", "feature2"]
    
    def test_load_latest_model(self):
        """Test loading the latest version of a model."""
        # Save multiple versions
        for i, version in enumerate(["v1.0", "v1.1", "v2.0"]):
            model = LogisticRegression(random_state=42 + i)
            model.fit([[1, 2], [3, 4]], [0, 1])
            
            self.registry.save_model(model, "test_model", version)
        
        # Load latest (should be v2.0 based on filename)
        loaded_model, loaded_metadata = self.registry.load_model("test_model", "latest")
        
        # The latest should be the most recently created file
        assert loaded_model is not None
    
    def test_load_nonexistent_model(self):
        """Test loading a model that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            self.registry.load_model("nonexistent_model", "v1.0")
    
    def test_list_models(self):
        """Test listing available models."""
        # Save some models
        models_to_save = [
            ("model_a", "v1.0"),
            ("model_a", "v1.1"),
            ("model_b", "v1.0")
        ]
        
        for model_name, version in models_to_save:
            self.registry.save_model(self.test_model, model_name, version)
        
        # List models
        models_list = self.registry.list_models()
        
        assert len(models_list) == 3
        
        # Check that all saved models are in the list
        model_identifiers = [(m["name"], m["version"]) for m in models_list]
        for model_name, version in models_to_save:
            assert (model_name, version) in model_identifiers
    
    def test_model_caching(self):
        """Test that models are cached after loading."""
        # Save model
        self.registry.save_model(self.test_model, "cached_model", "v1.0")
        
        # Load model twice
        model1, metadata1 = self.registry.load_model("cached_model", "v1.0")
        model2, metadata2 = self.registry.load_model("cached_model", "v1.0")
        
        # Should be the same object from cache
        assert model1 is model2
        assert metadata1 is metadata2


class TestPredictionService:
    """Test cases for PredictionService."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.models_path = Path(self.temp_dir)
        
        # Create mock model registry
        self.mock_registry = Mock(spec=ModelRegistry)
        
        # Create a test model with predict_proba method
        self.test_model = Mock()
        self.test_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        self.test_model.predict.return_value = np.array([1])
        
        self.test_metadata = {
            "version": "v1.0",
            "feature_names": [
                "gender", "senior_citizen", "partner", "dependents", "tenure",
                "contract", "paperless_billing", "payment_method", "phone_service",
                "multiple_lines", "internet_service", "online_security",
                "online_backup", "device_protection", "tech_support",
                "streaming_tv", "streaming_movies", "monthly_charges", "total_charges"
            ]
        }
        
        # Configure mock registry
        self.mock_registry.load_model.return_value = (self.test_model, self.test_metadata)
        
        # Create prediction service
        self.service = PredictionService(
            model_registry=self.mock_registry,
            default_model_name="test_model",
            default_model_version="v1.0"
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_service_initialization(self):
        """Test PredictionService initialization."""
        assert self.service.current_model is not None
        assert self.service.default_model_name == "test_model"
        assert self.service.default_model_version == "v1.0"
        assert len(self.service.feature_names) > 0
    
    def test_preprocess_input(self):
        """Test input preprocessing."""
        customer = create_sample_customer()
        features = self.service.preprocess_input(customer)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 1  # Single customer
        assert features.shape[1] == len(self.service.feature_names)
    
    def test_predict_churn_probability(self):
        """Test churn probability prediction."""
        # Create dummy features
        features = np.array([[1, 0, 1, 0, 24, 1, 1, 2, 1, 0, 2, 1, 0, 1, 0, 1, 1, 75.5, 1815.0]])
        
        churn_prob, confidence = self.service.predict_churn_probability(features)
        
        assert 0 <= churn_prob <= 1
        assert 0 <= confidence <= 1
        assert churn_prob == 0.7  # Based on mock return value
    
    def test_identify_key_risk_factors(self):
        """Test risk factor identification."""
        customer = create_sample_customer()
        customer.contract = "Month-to-month"
        customer.tenure = 3
        customer.monthly_charges = 85.0
        
        features = np.array([[1, 0, 1, 0, 3, 0, 1, 2, 1, 0, 2, 1, 0, 1, 0, 1, 1, 85.0, 255.0]])
        
        risk_factors = self.service.identify_key_risk_factors(features, customer)
        
        assert isinstance(risk_factors, list)
        assert len(risk_factors) > 0
        assert any("Month-to-month" in factor for factor in risk_factors)
        assert any("new customer" in factor.lower() for factor in risk_factors)
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        customer = create_sample_customer()
        customer.contract = "Month-to-month"
        customer.payment_method = "Electronic check"
        
        prediction_result = PredictionResult(
            customer_id=customer.customer_id,
            churn_probability=0.8,
            risk_level="high",
            confidence_score=0.9,
            key_risk_factors=["Month-to-month contract"],
            recommendations=[],
            model_version="v1.0",
            prediction_timestamp=datetime.now().isoformat()
        )
        
        recommendations = self.service.generate_recommendations(prediction_result, customer)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("retention intervention" in rec.lower() for rec in recommendations)
        assert any("contract" in rec.lower() for rec in recommendations)
    
    def test_full_prediction_workflow(self):
        """Test complete prediction workflow."""
        customer = create_sample_customer()
        
        result = self.service.predict(customer)
        
        assert isinstance(result, PredictionResult)
        assert result.customer_id == customer.customer_id
        assert 0 <= result.churn_probability <= 1
        assert result.risk_level in ["low", "medium", "high"]
        assert 0 <= result.confidence_score <= 1
        assert len(result.key_risk_factors) > 0
        assert len(result.recommendations) > 0
        assert result.model_version == "v1.0"
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        customers = [create_sample_customer() for _ in range(3)]
        
        # Give each customer a unique ID
        for i, customer in enumerate(customers):
            customer.customer_id = f"BATCH_TEST_{i:03d}"
        
        results = self.service.batch_predict(customers)
        
        assert len(results) == 3
        assert all(isinstance(result, PredictionResult) for result in results)
        assert all(result.customer_id.startswith("BATCH_TEST_") for result in results)
    
    def test_model_info(self):
        """Test getting model information."""
        info = self.service.get_model_info()
        
        assert "model_name" in info
        assert "model_version" in info
        assert "feature_count" in info
        assert "feature_names" in info
        assert info["model_name"] == "test_model"
        assert info["model_version"] == "v1.0"
    
    def test_health_check(self):
        """Test service health check."""
        health = self.service.health_check()
        
        assert "status" in health
        assert "timestamp" in health
        assert "model_loaded" in health
        assert health["status"] == "healthy"
        assert health["model_loaded"] is True
    
    def test_service_without_model(self):
        """Test service behavior when no model is loaded."""
        # Create service with mock registry that raises FileNotFoundError
        mock_registry = Mock(spec=ModelRegistry)
        mock_registry.load_model.side_effect = FileNotFoundError("Model not found")
        
        service = PredictionService(
            model_registry=mock_registry,
            default_model_name="nonexistent_model"
        )
        
        # Health check should show unhealthy status
        health = service.health_check()
        assert health["status"] == "unhealthy"
        assert "No model loaded" in health["issues"]
        
        # Prediction should raise error
        customer = create_sample_customer()
        with pytest.raises(RuntimeError, match="No model loaded"):
            service.predict(customer)
    
    def test_model_without_predict_proba(self):
        """Test handling of models without predict_proba method."""
        # Create model with only decision_function
        mock_model = Mock()
        mock_model.decision_function.return_value = np.array([1.5])
        del mock_model.predict_proba  # Remove predict_proba method
        
        self.mock_registry.load_model.return_value = (mock_model, self.test_metadata)
        
        service = PredictionService(
            model_registry=self.mock_registry,
            default_model_name="test_model"
        )
        
        features = np.array([[1, 0, 1, 0, 24, 1, 1, 2, 1, 0, 2, 1, 0, 1, 0, 1, 1, 75.5, 1815.0]])
        churn_prob, confidence = service.predict_churn_probability(features)
        
        assert 0 <= churn_prob <= 1
        assert 0 <= confidence <= 1
    
    def test_risk_level_determination(self):
        """Test risk level determination based on probability."""
        # Test low risk
        assert self.service._determine_risk_level(0.2) == "low"
        
        # Test medium risk
        assert self.service._determine_risk_level(0.5) == "medium"
        
        # Test high risk
        assert self.service._determine_risk_level(0.8) == "high"
    
    def test_customer_lifetime_value_estimation(self):
        """Test CLV estimation."""
        customer = create_sample_customer()
        customer.monthly_charges = 100.0
        customer.contract = "Two year"
        customer.tenure = 48
        
        clv = self.service._estimate_customer_lifetime_value(customer)
        
        assert clv > 0
        assert clv > customer.monthly_charges * 24  # Should be higher than base calculation
    
    def test_retention_cost_benefit_calculation(self):
        """Test retention cost-benefit calculation."""
        customer = create_sample_customer()
        customer.monthly_charges = 100.0
        
        benefit = self.service._calculate_retention_cost_benefit(0.8, customer)
        
        # Should be positive for high churn probability and high value customer
        assert isinstance(benefit, float)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_create_sample_customer(self):
        """Test sample customer creation."""
        customer = create_sample_customer()
        
        assert isinstance(customer, CustomerInput)
        assert customer.customer_id == "SAMPLE_001"
        assert customer.monthly_charges > 0
        assert customer.total_charges > 0
    
    @patch('utils.prediction_service.PredictionService')
    def test_quick_prediction(self, mock_service_class):
        """Test quick prediction function."""
        # Mock the service and its predict method
        mock_service = Mock()
        mock_result = PredictionResult(
            customer_id="TEST_001",
            churn_probability=0.6,
            risk_level="medium",
            confidence_score=0.8,
            key_risk_factors=["Test factor"],
            recommendations=["Test recommendation"],
            model_version="v1.0",
            prediction_timestamp=datetime.now().isoformat()
        )
        mock_service.predict.return_value = mock_result
        mock_service_class.return_value = mock_service
        
        customer = create_sample_customer()
        result = quick_prediction(customer, "test_model")
        
        assert isinstance(result, PredictionResult)
        assert result.customer_id == "TEST_001"
        mock_service_class.assert_called_once_with(default_model_name="test_model")
        mock_service.predict.assert_called_once_with(customer)


class TestErrorHandling:
    """Test cases for error handling scenarios."""
    
    def test_invalid_feature_array_shape(self):
        """Test handling of invalid feature array shapes."""
        mock_registry = Mock(spec=ModelRegistry)
        mock_model = Mock()
        mock_model.predict_proba.side_effect = ValueError("Invalid shape")
        
        mock_registry.load_model.return_value = (mock_model, {"version": "v1.0", "feature_names": ["f1", "f2"]})
        
        service = PredictionService(model_registry=mock_registry)
        
        # This should raise an error during prediction
        features = np.array([[1, 2, 3]])  # Wrong number of features
        
        with pytest.raises(ValueError):
            service.predict_churn_probability(features)
    
    def test_model_loading_failure(self):
        """Test handling of model loading failures."""
        mock_registry = Mock(spec=ModelRegistry)
        mock_registry.load_model.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            PredictionService(model_registry=mock_registry)
    
    def test_batch_prediction_with_errors(self):
        """Test batch prediction with some failing predictions."""
        mock_registry = Mock(spec=ModelRegistry)
        mock_model = Mock()
        
        # Make predict_proba fail for certain inputs
        def side_effect_predict_proba(features):
            if features[0, 0] == 999:  # Special value to trigger error
                raise ValueError("Prediction failed")
            return np.array([[0.3, 0.7]])
        
        mock_model.predict_proba.side_effect = side_effect_predict_proba
        mock_model.predict.return_value = np.array([1])
        
        mock_registry.load_model.return_value = (mock_model, {
            "version": "v1.0",
            "feature_names": ["feature1", "feature2"]
        })
        
        service = PredictionService(model_registry=mock_registry)
        
        # Create customers, one that will cause an error
        customers = [create_sample_customer() for _ in range(2)]
        customers[0].customer_id = "GOOD_001"
        customers[1].customer_id = "BAD_002"
        
        # Mock preprocessing to return error-triggering features for second customer
        original_preprocess = service.preprocess_input
        def mock_preprocess(customer_data):
            if customer_data.customer_id == "BAD_002":
                return np.array([[999, 0]])  # This will trigger the error
            return original_preprocess(customer_data)
        
        service.preprocess_input = mock_preprocess
        
        results = service.batch_predict(customers)
        
        assert len(results) == 2
        assert results[0].customer_id == "GOOD_001"
        assert results[1].customer_id == "BAD_002"
        assert results[1].model_version == "error"
        assert "Prediction failed" in results[1].key_risk_factors


if __name__ == "__main__":
    pytest.main([__file__])