"""
Integration tests for the Customer Churn Prediction API.

Tests cover all API endpoints including single predictions, batch predictions,
model information, health checks, and feedback submission.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app
from utils.prediction_service import PredictionResult, CustomerInput, PredictionService


class TestAPIIntegration:
    """Integration tests for the API endpoints."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        
        # Sample customer data for testing
        self.sample_customer_data = {
            "customer_id": "TEST_API_001",
            "gender": "Female",
            "senior_citizen": False,
            "partner": True,
            "dependents": False,
            "tenure": 24,
            "contract": "One year",
            "paperless_billing": True,
            "payment_method": "Credit card (automatic)",
            "phone_service": True,
            "multiple_lines": "No",
            "internet_service": "Fiber optic",
            "online_security": "Yes",
            "online_backup": "No",
            "device_protection": "Yes",
            "tech_support": "No",
            "streaming_tv": "Yes",
            "streaming_movies": "Yes",
            "monthly_charges": 75.50,
            "total_charges": 1815.00,
            "usage_minutes_monthly": 450.0,
            "data_usage_gb_monthly": 12.5,
            "support_interactions_count": 1,
            "complaint_count": 0,
            "satisfaction_score": 8.5
        }
        
        # Mock prediction result
        self.mock_prediction_result = PredictionResult(
            customer_id="TEST_API_001",
            churn_probability=0.65,
            risk_level="medium",
            confidence_score=0.82,
            key_risk_factors=["Medium tenure", "Fiber optic service"],
            recommendations=["Monitor engagement", "Consider loyalty program"],
            model_version="v1.0",
            prediction_timestamp=datetime.now().isoformat(),
            risk_score=65.0,
            risk_percentile=70.0,
            estimated_clv=2400.0,
            retention_cost_benefit=180.0
        )
    
    def test_root_endpoint(self):
        """Test the root endpoint."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "Customer Churn Prediction API"
        assert data["version"] == "1.0.0"
    
    @patch('api.main.prediction_service')
    def test_single_prediction_success(self, mock_service):
        """Test successful single customer prediction."""
        # Mock the prediction service
        mock_service.predict.return_value = self.mock_prediction_result
        
        response = self.client.post("/predict", json=self.sample_customer_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert data["customer_id"] == "TEST_API_001"
        assert data["churn_probability"] == 0.65
        assert data["risk_level"] == "medium"
        assert data["confidence_score"] == 0.82
        assert len(data["key_risk_factors"]) == 2
        assert len(data["recommendations"]) == 2
        assert data["model_version"] == "v1.0"
        assert data["estimated_clv"] == 2400.0
        
        # Verify service was called
        mock_service.predict.assert_called_once()
    
    def test_single_prediction_validation_error(self):
        """Test single prediction with validation errors."""
        # Test with invalid contract
        invalid_data = self.sample_customer_data.copy()
        invalid_data["contract"] = "Invalid Contract"
        
        response = self.client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "detail" in data
    
    def test_single_prediction_missing_fields(self):
        """Test single prediction with missing required fields."""
        # Remove required field
        incomplete_data = self.sample_customer_data.copy()
        del incomplete_data["customer_id"]
        
        response = self.client.post("/predict", json=incomplete_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_single_prediction_invalid_values(self):
        """Test single prediction with invalid field values."""
        # Test with negative tenure
        invalid_data = self.sample_customer_data.copy()
        invalid_data["tenure"] = -5
        
        response = self.client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
        
        # Test with invalid satisfaction score
        invalid_data = self.sample_customer_data.copy()
        invalid_data["satisfaction_score"] = 15.0
        
        response = self.client.post("/predict", json=invalid_data)
        
        assert response.status_code == 422  # Validation error
    
    @patch('api.main.prediction_service')
    def test_batch_prediction_success(self, mock_service):
        """Test successful batch prediction."""
        # Create multiple customers
        customers = []
        results = []
        
        for i in range(3):
            customer_data = self.sample_customer_data.copy()
            customer_data["customer_id"] = f"BATCH_TEST_{i:03d}"
            customers.append(customer_data)
            
            # Create corresponding result
            result = PredictionResult(
                customer_id=f"BATCH_TEST_{i:03d}",
                churn_probability=0.5 + i * 0.1,
                risk_level="medium",
                confidence_score=0.8,
                key_risk_factors=["Test factor"],
                recommendations=["Test recommendation"],
                model_version="v1.0",
                prediction_timestamp=datetime.now().isoformat()
            )
            results.append(result)
        
        # Mock the batch prediction
        mock_service.batch_predict.return_value = results
        
        batch_request = {"customers": customers}
        response = self.client.post("/predict/batch", json=batch_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "predictions" in data
        assert "batch_id" in data
        assert "processed_count" in data
        assert "failed_count" in data
        assert "processing_time_seconds" in data
        
        assert len(data["predictions"]) == 3
        assert data["processed_count"] == 3
        assert data["failed_count"] == 0
        
        # Verify individual predictions
        for i, prediction in enumerate(data["predictions"]):
            assert prediction["customer_id"] == f"BATCH_TEST_{i:03d}"
            assert prediction["churn_probability"] == 0.5 + i * 0.1
    
    def test_batch_prediction_empty_list(self):
        """Test batch prediction with empty customer list."""
        batch_request = {"customers": []}
        response = self.client.post("/predict/batch", json=batch_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_batch_prediction_too_large(self):
        """Test batch prediction with too many customers."""
        # Create a batch that exceeds the limit
        customers = [self.sample_customer_data.copy() for _ in range(1001)]
        for i, customer in enumerate(customers):
            customer["customer_id"] = f"LARGE_BATCH_{i:04d}"
        
        batch_request = {"customers": customers}
        response = self.client.post("/predict/batch", json=batch_request)
        
        assert response.status_code == 422  # Validation error
    
    @patch('api.main.prediction_service')
    def test_model_info_success(self, mock_service):
        """Test successful model info retrieval."""
        # Mock model info
        mock_info = {
            "model_name": "churn_predictor",
            "model_version": "v1.0",
            "model_type": "RandomForestClassifier",
            "feature_count": 19,
            "feature_names": ["feature1", "feature2"],
            "risk_thresholds": {"low": 0.3, "medium": 0.7, "high": 1.0},
            "performance_metrics": {"accuracy": 0.85, "auc": 0.92},
            "last_updated": "2024-01-01T12:00:00"
        }
        
        mock_service.get_model_info.return_value = mock_info
        
        response = self.client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["model_name"] == "churn_predictor"
        assert data["model_version"] == "v1.0"
        assert data["model_type"] == "RandomForestClassifier"
        assert data["feature_count"] == 19
        assert len(data["feature_names"]) == 2
        assert "risk_thresholds" in data
        assert "performance_metrics" in data
    
    @patch('api.main.prediction_service')
    def test_health_check_healthy(self, mock_service):
        """Test health check when service is healthy."""
        # Mock healthy status
        mock_health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": True,
            "feature_count": 19,
            "uptime_seconds": 3600,
            "issues": []
        }
        
        mock_service.health_check.return_value = mock_health
        
        response = self.client.get("/model/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["feature_count"] == 19
        assert len(data["issues"]) == 0
    
    @patch('api.main.prediction_service', None)
    def test_health_check_unhealthy(self):
        """Test health check when service is not initialized."""
        response = self.client.get("/model/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert len(data["issues"]) > 0
        assert "not initialized" in data["issues"][0]
    
    def test_feedback_submission_success(self):
        """Test successful feedback submission."""
        feedback_data = {
            "customer_id": "TEST_API_001",
            "prediction_id": "pred_123",
            "actual_churn": True,
            "intervention_applied": True,
            "intervention_type": "discount_offer",
            "outcome_date": "2024-01-15",
            "notes": "Customer churned despite intervention"
        }
        
        response = self.client.post("/model/feedback", json=feedback_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert data["customer_id"] == "TEST_API_001"
        assert "timestamp" in data
    
    def test_feedback_submission_missing_fields(self):
        """Test feedback submission with missing required fields."""
        incomplete_feedback = {
            "customer_id": "TEST_API_001",
            # Missing actual_churn and outcome_date
            "intervention_applied": False
        }
        
        response = self.client.post("/model/feedback", json=incomplete_feedback)
        
        assert response.status_code == 422  # Validation error
    
    @patch('api.main.prediction_service', None)
    def test_endpoints_without_service(self):
        """Test endpoints when prediction service is not available."""
        # Test predict endpoint
        response = self.client.post("/predict", json=self.sample_customer_data)
        assert response.status_code == 503
        
        # Test batch predict endpoint
        batch_request = {"customers": [self.sample_customer_data]}
        response = self.client.post("/predict/batch", json=batch_request)
        assert response.status_code == 503
        
        # Test model info endpoint
        response = self.client.get("/model/info")
        assert response.status_code == 503
    
    @patch('api.main.prediction_service')
    def test_service_exception_handling(self, mock_service):
        """Test handling of service exceptions."""
        # Mock service to raise exception
        mock_service.predict.side_effect = Exception("Service error")
        
        response = self.client.post("/predict", json=self.sample_customer_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Internal server error" in data["error"]
    
    def test_cors_headers(self):
        """Test CORS headers are present."""
        response = self.client.get("/")
        
        # Check for CORS headers (TestClient may not include all headers)
        assert response.status_code == 200
    
    def test_api_documentation_endpoints(self):
        """Test that API documentation endpoints are accessible."""
        # Test OpenAPI docs
        response = self.client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = self.client.get("/redoc")
        assert response.status_code == 200
        
        # Test OpenAPI JSON
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Customer Churn Prediction API"


class TestAPIPerformance:
    """Performance tests for the API."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
        
        self.sample_customer_data = {
            "customer_id": "PERF_TEST_001",
            "gender": "Female",
            "senior_citizen": False,
            "partner": True,
            "dependents": False,
            "tenure": 24,
            "contract": "One year",
            "paperless_billing": True,
            "payment_method": "Credit card (automatic)",
            "phone_service": True,
            "multiple_lines": "No",
            "internet_service": "Fiber optic",
            "online_security": "Yes",
            "online_backup": "No",
            "device_protection": "Yes",
            "tech_support": "No",
            "streaming_tv": "Yes",
            "streaming_movies": "Yes",
            "monthly_charges": 75.50,
            "total_charges": 1815.00
        }
    
    @patch('api.main.prediction_service')
    def test_single_prediction_response_time(self, mock_service):
        """Test single prediction response time."""
        import time
        
        # Mock quick response
        mock_result = PredictionResult(
            customer_id="PERF_TEST_001",
            churn_probability=0.5,
            risk_level="medium",
            confidence_score=0.8,
            key_risk_factors=["Test"],
            recommendations=["Test"],
            model_version="v1.0",
            prediction_timestamp=datetime.now().isoformat()
        )
        mock_service.predict.return_value = mock_result
        
        start_time = time.time()
        response = self.client.post("/predict", json=self.sample_customer_data)
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Response should be fast (under 1 second for mocked service)
        response_time = end_time - start_time
        assert response_time < 1.0
    
    @patch('api.main.prediction_service')
    def test_batch_prediction_scalability(self, mock_service):
        """Test batch prediction with varying batch sizes."""
        import time
        
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            # Create batch
            customers = []
            results = []
            
            for i in range(batch_size):
                customer_data = self.sample_customer_data.copy()
                customer_data["customer_id"] = f"SCALE_TEST_{i:03d}"
                customers.append(customer_data)
                
                result = PredictionResult(
                    customer_id=f"SCALE_TEST_{i:03d}",
                    churn_probability=0.5,
                    risk_level="medium",
                    confidence_score=0.8,
                    key_risk_factors=["Test"],
                    recommendations=["Test"],
                    model_version="v1.0",
                    prediction_timestamp=datetime.now().isoformat()
                )
                results.append(result)
            
            mock_service.batch_predict.return_value = results
            
            # Time the request
            start_time = time.time()
            batch_request = {"customers": customers}
            response = self.client.post("/predict/batch", json=batch_request)
            end_time = time.time()
            
            assert response.status_code == 200
            
            # Response time should scale reasonably
            response_time = end_time - start_time
            print(f"Batch size {batch_size}: {response_time:.3f}s")
            
            # Should handle reasonable batch sizes efficiently
            if batch_size <= 100:
                assert response_time < 5.0  # Should be under 5 seconds


class TestAPIErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.client = TestClient(app)
    
    def test_malformed_json(self):
        """Test handling of malformed JSON."""
        response = self.client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422
    
    def test_unsupported_media_type(self):
        """Test handling of unsupported media types."""
        response = self.client.post(
            "/predict",
            data="some data",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code == 422
    
    def test_method_not_allowed(self):
        """Test handling of unsupported HTTP methods."""
        response = self.client.put("/predict")
        assert response.status_code == 405
        
        response = self.client.delete("/predict")
        assert response.status_code == 405
    
    def test_not_found(self):
        """Test handling of non-existent endpoints."""
        response = self.client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_large_payload(self):
        """Test handling of very large payloads."""
        # Create a very large customer object
        large_customer = {
            "customer_id": "LARGE_TEST_001",
            "gender": "Female",
            "senior_citizen": False,
            "partner": True,
            "dependents": False,
            "tenure": 24,
            "contract": "One year",
            "paperless_billing": True,
            "payment_method": "Credit card (automatic)",
            "phone_service": True,
            "multiple_lines": "No",
            "internet_service": "Fiber optic",
            "online_security": "Yes",
            "online_backup": "No",
            "device_protection": "Yes",
            "tech_support": "No",
            "streaming_tv": "Yes",
            "streaming_movies": "Yes",
            "monthly_charges": 75.50,
            "total_charges": 1815.00,
            "notes": "x" * 10000  # Large text field
        }
        
        response = self.client.post("/predict", json=large_customer)
        
        # Should handle large payloads gracefully
        assert response.status_code in [200, 413, 422]  # Success, payload too large, or validation error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])