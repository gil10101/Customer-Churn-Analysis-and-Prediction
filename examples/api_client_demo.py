"""
API Client Demo for Customer Churn Prediction API.

This script demonstrates how to interact with the FastAPI endpoints
for churn prediction, including single predictions, batch predictions,
and model information retrieval.
"""

import requests
import json
import time
from typing import List, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.logging_setup import get_notebook_logger

logger = get_notebook_logger(__name__)


class ChurnPredictionAPIClient:
    """Client for interacting with the Customer Churn Prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set default headers
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health status of the API."""
        try:
            response = self.session.get(f"{self.base_url}/model/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model info: {e}")
            raise
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single customer.
        
        Args:
            customer_data: Customer data dictionary
            
        Returns:
            Prediction result dictionary
        """
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=customer_data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Single prediction failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
    
    def predict_batch(self, customers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make predictions for multiple customers.
        
        Args:
            customers: List of customer data dictionaries
            
        Returns:
            Batch prediction result dictionary
        """
        try:
            batch_request = {"customers": customers}
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=batch_request
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch prediction failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
    
    def submit_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit feedback on a prediction.
        
        Args:
            feedback_data: Feedback data dictionary
            
        Returns:
            Feedback submission result
        """
        try:
            response = self.session.post(
                f"{self.base_url}/model/feedback",
                json=feedback_data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Feedback submission failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise


def create_sample_customer_data(customer_id: str) -> Dict[str, Any]:
    """Create sample customer data for testing."""
    return {
        "customer_id": customer_id,
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


def demo_health_check(client: ChurnPredictionAPIClient):
    """Demonstrate health check functionality."""
    logger.info("\n" + "="*60)
    logger.info("HEALTH CHECK DEMO")
    logger.info("="*60)
    
    try:
        health = client.health_check()
        
        logger.info(f"API Health Status: {health['status'].upper()}")
        logger.info(f"Model Loaded: {health['model_loaded']}")
        logger.info(f"Feature Count: {health['feature_count']}")
        
        if health['issues']:
            logger.warning(f"Issues: {', '.join(health['issues'])}")
        else:
            logger.info("No issues detected")
            
        return health['status'] == 'healthy'
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False


def demo_model_info(client: ChurnPredictionAPIClient):
    """Demonstrate model information retrieval."""
    logger.info("\n" + "="*60)
    logger.info("MODEL INFORMATION DEMO")
    logger.info("="*60)
    
    try:
        info = client.get_model_info()
        
        logger.info(f"Model Name: {info['model_name']}")
        logger.info(f"Model Version: {info['model_version']}")
        logger.info(f"Model Type: {info['model_type']}")
        logger.info(f"Feature Count: {info['feature_count']}")
        logger.info(f"Risk Thresholds: {info['risk_thresholds']}")
        
        if info.get('performance_metrics'):
            logger.info(f"Performance Metrics: {info['performance_metrics']}")
        
        return info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return None


def demo_single_prediction(client: ChurnPredictionAPIClient):
    """Demonstrate single customer prediction."""
    logger.info("\n" + "="*60)
    logger.info("SINGLE PREDICTION DEMO")
    logger.info("="*60)
    
    try:
        # Create sample customer
        customer_data = create_sample_customer_data("API_DEMO_001")
        
        logger.info(f"Making prediction for customer: {customer_data['customer_id']}")
        logger.info(f"Customer profile:")
        logger.info(f"  - Contract: {customer_data['contract']}")
        logger.info(f"  - Tenure: {customer_data['tenure']} months")
        logger.info(f"  - Monthly charges: ${customer_data['monthly_charges']}")
        logger.info(f"  - Internet service: {customer_data['internet_service']}")
        
        # Make prediction
        start_time = time.time()
        result = client.predict_single(customer_data)
        end_time = time.time()
        
        # Display results
        logger.info(f"\nPREDICTION RESULTS (Response time: {end_time - start_time:.3f}s):")
        logger.info(f"  - Churn probability: {result['churn_probability']:.1%}")
        logger.info(f"  - Risk level: {result['risk_level'].upper()}")
        logger.info(f"  - Confidence score: {result['confidence_score']:.1%}")
        
        if result.get('estimated_clv'):
            logger.info(f"  - Estimated CLV: ${result['estimated_clv']:,.2f}")
        
        if result.get('retention_cost_benefit'):
            logger.info(f"  - Retention cost-benefit: ${result['retention_cost_benefit']:,.2f}")
        
        logger.info(f"\nKEY RISK FACTORS:")
        for i, factor in enumerate(result['key_risk_factors'], 1):
            logger.info(f"  {i}. {factor}")
        
        logger.info(f"\nRECOMMENDATIONS:")
        for i, rec in enumerate(result['recommendations'], 1):
            logger.info(f"  {i}. {rec}")
        
        return result
        
    except Exception as e:
        logger.error(f"Single prediction failed: {e}")
        return None


def demo_batch_prediction(client: ChurnPredictionAPIClient):
    """Demonstrate batch prediction."""
    logger.info("\n" + "="*60)
    logger.info("BATCH PREDICTION DEMO")
    logger.info("="*60)
    
    try:
        # Create multiple customers with different profiles
        customers = []
        
        # High-risk customer
        high_risk = create_sample_customer_data("BATCH_HIGH_001")
        high_risk.update({
            "contract": "Month-to-month",
            "tenure": 3,
            "monthly_charges": 110.0,
            "payment_method": "Electronic check",
            "support_interactions_count": 5,
            "satisfaction_score": 4.0
        })
        customers.append(high_risk)
        
        # Medium-risk customer
        medium_risk = create_sample_customer_data("BATCH_MEDIUM_002")
        medium_risk.update({
            "contract": "One year",
            "tenure": 18,
            "monthly_charges": 75.0,
            "support_interactions_count": 1,
            "satisfaction_score": 7.0
        })
        customers.append(medium_risk)
        
        # Low-risk customer
        low_risk = create_sample_customer_data("BATCH_LOW_003")
        low_risk.update({
            "contract": "Two year",
            "tenure": 48,
            "monthly_charges": 55.0,
            "payment_method": "Bank transfer (automatic)",
            "support_interactions_count": 0,
            "satisfaction_score": 9.0
        })
        customers.append(low_risk)
        
        logger.info(f"Making batch predictions for {len(customers)} customers...")
        
        # Make batch prediction
        start_time = time.time()
        result = client.predict_batch(customers)
        end_time = time.time()
        
        # Display results
        logger.info(f"\nBATCH PREDICTION RESULTS:")
        logger.info(f"  - Batch ID: {result['batch_id']}")
        logger.info(f"  - Processed: {result['processed_count']} customers")
        logger.info(f"  - Failed: {result['failed_count']} customers")
        logger.info(f"  - Processing time: {result['processing_time_seconds']:.3f}s")
        logger.info(f"  - API response time: {end_time - start_time:.3f}s")
        
        logger.info(f"\nINDIVIDUAL RESULTS:")
        logger.info("-" * 80)
        
        for prediction in result['predictions']:
            logger.info(f"Customer: {prediction['customer_id']}")
            logger.info(f"  Churn Probability: {prediction['churn_probability']:.1%}")
            logger.info(f"  Risk Level: {prediction['risk_level'].upper()}")
            logger.info(f"  Confidence: {prediction['confidence_score']:.1%}")
            logger.info(f"  Top Risk Factor: {prediction['key_risk_factors'][0] if prediction['key_risk_factors'] else 'None'}")
            logger.info("-" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return None


def demo_feedback_submission(client: ChurnPredictionAPIClient):
    """Demonstrate feedback submission."""
    logger.info("\n" + "="*60)
    logger.info("FEEDBACK SUBMISSION DEMO")
    logger.info("="*60)
    
    try:
        # Create sample feedback
        feedback_data = {
            "customer_id": "API_DEMO_001",
            "prediction_id": "pred_demo_123",
            "actual_churn": False,
            "intervention_applied": True,
            "intervention_type": "loyalty_discount",
            "outcome_date": "2024-02-15",
            "notes": "Customer retained after 20% discount offer"
        }
        
        logger.info(f"Submitting feedback for customer: {feedback_data['customer_id']}")
        logger.info(f"Actual churn: {feedback_data['actual_churn']}")
        logger.info(f"Intervention applied: {feedback_data['intervention_applied']}")
        logger.info(f"Intervention type: {feedback_data['intervention_type']}")
        
        # Submit feedback
        result = client.submit_feedback(feedback_data)
        
        logger.info(f"\nFEEDBACK SUBMISSION RESULT:")
        logger.info(f"  - Status: {result['message']}")
        logger.info(f"  - Customer ID: {result['customer_id']}")
        logger.info(f"  - Timestamp: {result['timestamp']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        return None


def demo_error_handling(client: ChurnPredictionAPIClient):
    """Demonstrate error handling."""
    logger.info("\n" + "="*60)
    logger.info("ERROR HANDLING DEMO")
    logger.info("="*60)
    
    # Test with invalid customer data
    logger.info("Testing with invalid customer data...")
    
    try:
        invalid_customer = create_sample_customer_data("INVALID_001")
        invalid_customer["contract"] = "Invalid Contract"  # Invalid value
        
        result = client.predict_single(invalid_customer)
        logger.warning("Expected validation error but got success")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            logger.info("✓ Validation error handled correctly")
            logger.info(f"  Error details: {e.response.json()}")
        else:
            logger.error(f"Unexpected HTTP error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    
    # Test with missing required fields
    logger.info("\nTesting with missing required fields...")
    
    try:
        incomplete_customer = {"customer_id": "INCOMPLETE_001"}  # Missing required fields
        
        result = client.predict_single(incomplete_customer)
        logger.warning("Expected validation error but got success")
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 422:
            logger.info("✓ Missing field validation handled correctly")
        else:
            logger.error(f"Unexpected HTTP error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


def main():
    """Run all API client demos."""
    logger.info("Starting Customer Churn Prediction API Client Demo")
    logger.info("="*80)
    
    # Initialize client
    client = ChurnPredictionAPIClient()
    
    try:
        # Check if API is available
        logger.info("Checking API availability...")
        if not demo_health_check(client):
            logger.error("API is not healthy. Please start the API server first.")
            logger.info("To start the server, run: python api/run_server.py")
            return
        
        # Run demos
        model_info = demo_model_info(client)
        single_result = demo_single_prediction(client)
        batch_result = demo_batch_prediction(client)
        feedback_result = demo_feedback_submission(client)
        demo_error_handling(client)
        
        logger.info("\n" + "="*80)
        logger.info("API CLIENT DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        # Summary
        logger.info(f"\nSUMMARY:")
        if model_info:
            logger.info(f"  - Model: {model_info['model_name']} v{model_info['model_version']}")
        if single_result:
            logger.info(f"  - Single prediction: {single_result['risk_level']} risk")
        if batch_result:
            logger.info(f"  - Batch predictions: {batch_result['processed_count']} customers processed")
        if feedback_result:
            logger.info(f"  - Feedback: Successfully submitted")
        
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to API server.")
        logger.info("Please make sure the API server is running:")
        logger.info("  python api/run_server.py")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()