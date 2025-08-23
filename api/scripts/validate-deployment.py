#!/usr/bin/env python3
"""
Deployment Validation Script for Churn Prediction API

This script validates that the deployed API is functioning correctly
by running a comprehensive set of tests against the live endpoints.
"""

import requests
import json
import time
import sys
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    response_time: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class APIValidator:
    """Validates API deployment by running comprehensive tests."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'API-Validator/1.0'
        })
        
    def run_all_validations(self) -> List[ValidationResult]:
        """Run all validation tests."""
        logger.info(f"Starting API validation for {self.base_url}")
        
        validations = [
            self.validate_health_check,
            self.validate_model_info,
            self.validate_single_prediction,
            self.validate_batch_prediction,
            self.validate_error_handling,
            self.validate_rate_limiting,
            self.validate_documentation,
            self.validate_performance,
        ]
        
        results = []
        for validation in validations:
            try:
                result = validation()
                results.append(result)
                status = "✅ PASS" if result.passed else "❌ FAIL"
                logger.info(f"{status} {result.test_name} ({result.response_time:.2f}ms)")
                if not result.passed and result.error_message:
                    logger.error(f"   Error: {result.error_message}")
            except Exception as e:
                result = ValidationResult(
                    test_name=validation.__name__,
                    passed=False,
                    response_time=0,
                    error_message=str(e)
                )
                results.append(result)
                logger.error(f"❌ FAIL {validation.__name__} - Exception: {e}")
        
        return results
    
    def validate_health_check(self) -> ValidationResult:
        """Validate health check endpoint."""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/model/health", timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return ValidationResult(
                    test_name="Health Check",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Expected status 200, got {response.status_code}"
                )
            
            data = response.json()
            required_fields = ['status', 'timestamp', 'model_loaded']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return ValidationResult(
                    test_name="Health Check",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Missing required fields: {missing_fields}"
                )
            
            if data['status'] != 'healthy':
                return ValidationResult(
                    test_name="Health Check",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Service status is '{data['status']}', expected 'healthy'"
                )
            
            return ValidationResult(
                test_name="Health Check",
                passed=True,
                response_time=response_time,
                details=data
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Health Check",
                passed=False,
                response_time=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def validate_model_info(self) -> ValidationResult:
        """Validate model info endpoint."""
        start_time = time.time()
        
        try:
            response = self.session.get(f"{self.base_url}/model/info", timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return ValidationResult(
                    test_name="Model Info",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Expected status 200, got {response.status_code}"
                )
            
            data = response.json()
            required_fields = ['model_name', 'model_version', 'feature_count']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return ValidationResult(
                    test_name="Model Info",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Missing required fields: {missing_fields}"
                )
            
            return ValidationResult(
                test_name="Model Info",
                passed=True,
                response_time=response_time,
                details=data
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Model Info",
                passed=False,
                response_time=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def validate_single_prediction(self) -> ValidationResult:
        """Validate single prediction endpoint."""
        start_time = time.time()
        
        sample_customer = {
            "customer_id": "validation_test_001",
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
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=sample_customer,
                timeout=self.timeout
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return ValidationResult(
                    test_name="Single Prediction",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Expected status 200, got {response.status_code}"
                )
            
            data = response.json()
            required_fields = ['customer_id', 'churn_probability', 'risk_level', 'confidence_score']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return ValidationResult(
                    test_name="Single Prediction",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Missing required fields: {missing_fields}"
                )
            
            # Validate data types and ranges
            if not (0 <= data['churn_probability'] <= 1):
                return ValidationResult(
                    test_name="Single Prediction",
                    passed=False,
                    response_time=response_time,
                    error_message=f"churn_probability {data['churn_probability']} not in range [0, 1]"
                )
            
            if data['risk_level'] not in ['low', 'medium', 'high']:
                return ValidationResult(
                    test_name="Single Prediction",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Invalid risk_level: {data['risk_level']}"
                )
            
            return ValidationResult(
                test_name="Single Prediction",
                passed=True,
                response_time=response_time,
                details=data
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Single Prediction",
                passed=False,
                response_time=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def validate_batch_prediction(self) -> ValidationResult:
        """Validate batch prediction endpoint."""
        start_time = time.time()
        
        # Create a small batch of customers
        customers = []
        for i in range(3):
            customer = {
                "customer_id": f"batch_validation_test_{i:03d}",
                "gender": "Female" if i % 2 == 0 else "Male",
                "senior_citizen": False,
                "partner": True,
                "dependents": False,
                "tenure": 12 + i * 6,
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
                "monthly_charges": 70.0 + i * 10,
                "total_charges": 1000.0 + i * 500,
            }
            customers.append(customer)
        
        batch_request = {"customers": customers}
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=batch_request,
                timeout=self.timeout
            )
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code != 200:
                return ValidationResult(
                    test_name="Batch Prediction",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Expected status 200, got {response.status_code}"
                )
            
            data = response.json()
            required_fields = ['predictions', 'batch_id', 'processed_count']
            missing_fields = [field for field in required_fields if field not in data]
            
            if missing_fields:
                return ValidationResult(
                    test_name="Batch Prediction",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Missing required fields: {missing_fields}"
                )
            
            if len(data['predictions']) != len(customers):
                return ValidationResult(
                    test_name="Batch Prediction",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Expected {len(customers)} predictions, got {len(data['predictions'])}"
                )
            
            return ValidationResult(
                test_name="Batch Prediction",
                passed=True,
                response_time=response_time,
                details={"batch_size": len(customers), "processed": data['processed_count']}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Batch Prediction",
                passed=False,
                response_time=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def validate_error_handling(self) -> ValidationResult:
        """Validate error handling with invalid input."""
        start_time = time.time()
        
        invalid_customer = {
            "customer_id": "error_test",
            "gender": "Invalid",  # Invalid gender
            "contract": "Invalid Contract",  # Invalid contract
            "monthly_charges": -100,  # Invalid negative charges
        }
        
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=invalid_customer,
                timeout=self.timeout
            )
            response_time = (time.time() - start_time) * 1000
            
            # Should return 422 for validation error
            if response.status_code not in [400, 422]:
                return ValidationResult(
                    test_name="Error Handling",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Expected status 400 or 422, got {response.status_code}"
                )
            
            # Should return structured error response
            try:
                error_data = response.json()
                if 'error' not in error_data and 'detail' not in error_data:
                    return ValidationResult(
                        test_name="Error Handling",
                        passed=False,
                        response_time=response_time,
                        error_message="Error response missing 'error' or 'detail' field"
                    )
            except json.JSONDecodeError:
                return ValidationResult(
                    test_name="Error Handling",
                    passed=False,
                    response_time=response_time,
                    error_message="Error response is not valid JSON"
                )
            
            return ValidationResult(
                test_name="Error Handling",
                passed=True,
                response_time=response_time,
                details={"status_code": response.status_code}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Error Handling",
                passed=False,
                response_time=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def validate_rate_limiting(self) -> ValidationResult:
        """Validate rate limiting (basic test)."""
        start_time = time.time()
        
        try:
            # Make multiple rapid requests to health endpoint
            responses = []
            for _ in range(5):
                response = self.session.get(f"{self.base_url}/model/health", timeout=5)
                responses.append(response.status_code)
                time.sleep(0.1)  # Small delay between requests
            
            response_time = (time.time() - start_time) * 1000
            
            # All requests should succeed (basic rate limiting test)
            if all(status == 200 for status in responses):
                return ValidationResult(
                    test_name="Rate Limiting",
                    passed=True,
                    response_time=response_time,
                    details={"requests_made": len(responses), "all_successful": True}
                )
            else:
                return ValidationResult(
                    test_name="Rate Limiting",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Some requests failed: {responses}"
                )
            
        except Exception as e:
            return ValidationResult(
                test_name="Rate Limiting",
                passed=False,
                response_time=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def validate_documentation(self) -> ValidationResult:
        """Validate API documentation endpoints."""
        start_time = time.time()
        
        try:
            # Check OpenAPI docs
            docs_response = self.session.get(f"{self.base_url}/docs", timeout=self.timeout)
            openapi_response = self.session.get(f"{self.base_url}/openapi.json", timeout=self.timeout)
            
            response_time = (time.time() - start_time) * 1000
            
            if docs_response.status_code != 200:
                return ValidationResult(
                    test_name="Documentation",
                    passed=False,
                    response_time=response_time,
                    error_message=f"Docs endpoint returned {docs_response.status_code}"
                )
            
            if openapi_response.status_code != 200:
                return ValidationResult(
                    test_name="Documentation",
                    passed=False,
                    response_time=response_time,
                    error_message=f"OpenAPI endpoint returned {openapi_response.status_code}"
                )
            
            # Validate OpenAPI JSON structure
            try:
                openapi_data = openapi_response.json()
                if 'openapi' not in openapi_data or 'paths' not in openapi_data:
                    return ValidationResult(
                        test_name="Documentation",
                        passed=False,
                        response_time=response_time,
                        error_message="Invalid OpenAPI specification structure"
                    )
            except json.JSONDecodeError:
                return ValidationResult(
                    test_name="Documentation",
                    passed=False,
                    response_time=response_time,
                    error_message="OpenAPI response is not valid JSON"
                )
            
            return ValidationResult(
                test_name="Documentation",
                passed=True,
                response_time=response_time,
                details={"endpoints_documented": len(openapi_data.get('paths', {}))}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Documentation",
                passed=False,
                response_time=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
    
    def validate_performance(self) -> ValidationResult:
        """Validate basic performance requirements."""
        start_time = time.time()
        
        sample_customer = {
            "customer_id": "performance_test",
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
        }
        
        try:
            # Make multiple prediction requests and measure response times
            response_times = []
            for _ in range(5):
                req_start = time.time()
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json=sample_customer,
                    timeout=self.timeout
                )
                req_time = (time.time() - req_start) * 1000
                response_times.append(req_time)
                
                if response.status_code != 200:
                    return ValidationResult(
                        test_name="Performance",
                        passed=False,
                        response_time=(time.time() - start_time) * 1000,
                        error_message=f"Request failed with status {response.status_code}"
                    )
            
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            
            # Performance thresholds
            avg_threshold = 1000  # 1 second average
            max_threshold = 2000  # 2 seconds maximum
            
            if avg_response_time > avg_threshold:
                return ValidationResult(
                    test_name="Performance",
                    passed=False,
                    response_time=(time.time() - start_time) * 1000,
                    error_message=f"Average response time {avg_response_time:.2f}ms exceeds threshold {avg_threshold}ms"
                )
            
            if max_response_time > max_threshold:
                return ValidationResult(
                    test_name="Performance",
                    passed=False,
                    response_time=(time.time() - start_time) * 1000,
                    error_message=f"Maximum response time {max_response_time:.2f}ms exceeds threshold {max_threshold}ms"
                )
            
            return ValidationResult(
                test_name="Performance",
                passed=True,
                response_time=(time.time() - start_time) * 1000,
                details={
                    "avg_response_time": avg_response_time,
                    "max_response_time": max_response_time,
                    "requests_tested": len(response_times)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Performance",
                passed=False,
                response_time=(time.time() - start_time) * 1000,
                error_message=str(e)
            )

def main():
    """Main function to run API validation."""
    parser = argparse.ArgumentParser(description='Validate Churn Prediction API deployment')
    parser.add_argument('--url', required=True, help='Base URL of the API to validate')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = APIValidator(args.url, args.timeout)
    
    # Run validations
    results = validator.run_all_validations()
    
    # Calculate summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    failed_tests = total_tests - passed_tests
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"API VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if failed_tests > 0:
        print(f"\n❌ FAILED TESTS:")
        for result in results:
            if not result.passed:
                print(f"  - {result.test_name}: {result.error_message}")
    
    # Save results to file if requested
    if args.output:
        results_data = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests/total_tests)*100
            },
            'results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'response_time': r.response_time,
                    'error_message': r.error_message,
                    'details': r.details
                }
                for r in results
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if failed_tests == 0 else 1)

if __name__ == '__main__':
    main()