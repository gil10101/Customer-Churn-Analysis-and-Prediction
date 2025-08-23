"""
Deployment Tests for Churn Prediction API

These tests validate the deployment infrastructure including:
- Docker container functionality
- Health checks
- Service integration
- Configuration validation
"""

import pytest
import requests
import docker
import time
import subprocess
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Test configuration
API_BASE_URL = "http://localhost:8000"
DEV_API_BASE_URL = "http://localhost:8001"
HEALTH_CHECK_TIMEOUT = 60
REQUEST_TIMEOUT = 30

class TestDockerDeployment:
    """Test Docker deployment functionality."""
    
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Docker client fixture."""
        return docker.from_env()
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists and is valid."""
        dockerfile_path = Path("api/Dockerfile")
        assert dockerfile_path.exists(), "Dockerfile not found"
        
        # Check basic Dockerfile structure
        content = dockerfile_path.read_text()
        assert "FROM python:" in content, "Dockerfile missing Python base image"
        assert "COPY" in content, "Dockerfile missing COPY instructions"
        assert "EXPOSE 8000" in content, "Dockerfile not exposing port 8000"
        assert "CMD" in content or "ENTRYPOINT" in content, "Dockerfile missing CMD/ENTRYPOINT"
    
    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists and is valid."""
        compose_path = Path("api/docker-compose.yml")
        assert compose_path.exists(), "docker-compose.yml not found"
        
        # Check basic compose structure
        content = compose_path.read_text()
        assert "version:" in content, "docker-compose.yml missing version"
        assert "services:" in content, "docker-compose.yml missing services"
        assert "churn-prediction-api:" in content, "docker-compose.yml missing API service"
    
    def test_docker_build(self, docker_client):
        """Test that Docker image builds successfully."""
        try:
            # Build the image
            image, logs = docker_client.images.build(
                path=".",
                dockerfile="api/Dockerfile",
                tag="churn-prediction-api:test",
                rm=True
            )
            
            assert image is not None, "Docker image build failed"
            assert image.tags, "Docker image has no tags"
            
            # Clean up
            docker_client.images.remove(image.id, force=True)
            
        except docker.errors.BuildError as e:
            pytest.fail(f"Docker build failed: {e}")
    
    def test_docker_compose_validation(self):
        """Test docker-compose configuration validation."""
        try:
            # Validate compose file
            result = subprocess.run(
                ["docker-compose", "-f", "api/docker-compose.yml", "config"],
                capture_output=True,
                text=True,
                cwd=".",
                timeout=30
            )
            
            assert result.returncode == 0, f"docker-compose validation failed: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.fail("docker-compose validation timed out")
        except FileNotFoundError:
            pytest.skip("docker-compose not available")

class TestHealthChecks:
    """Test health check functionality."""
    
    def test_health_check_endpoint_structure(self):
        """Test health check endpoint returns proper structure."""
        try:
            response = requests.get(f"{API_BASE_URL}/model/health", timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['status', 'timestamp', 'model_loaded']
                for field in required_fields:
                    assert field in data, f"Health check missing required field: {field}"
                
                # Check field types
                assert isinstance(data['status'], str), "status should be string"
                assert isinstance(data['timestamp'], str), "timestamp should be string"
                assert isinstance(data['model_loaded'], bool), "model_loaded should be boolean"
                
        except requests.exceptions.RequestException:
            pytest.skip("API not available for health check test")
    
    def test_health_check_response_time(self):
        """Test health check response time is acceptable."""
        try:
            start_time = time.time()
            response = requests.get(f"{API_BASE_URL}/model/health", timeout=REQUEST_TIMEOUT)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                # Health check should respond within 500ms
                assert response_time < 500, f"Health check too slow: {response_time}ms"
                
        except requests.exceptions.RequestException:
            pytest.skip("API not available for response time test")

class TestServiceIntegration:
    """Test service integration and dependencies."""
    
    def test_api_model_info_endpoint(self):
        """Test model info endpoint functionality."""
        try:
            response = requests.get(f"{API_BASE_URL}/model/info", timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ['model_name', 'model_version', 'feature_count']
                for field in required_fields:
                    assert field in data, f"Model info missing required field: {field}"
                
        except requests.exceptions.RequestException:
            pytest.skip("API not available for model info test")
    
    def test_api_documentation_endpoints(self):
        """Test API documentation endpoints."""
        endpoints = ["/docs", "/redoc", "/openapi.json"]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=REQUEST_TIMEOUT)
                
                if response.status_code == 200:
                    # Basic validation that content is returned
                    assert len(response.content) > 0, f"Empty response from {endpoint}"
                    
                    if endpoint == "/openapi.json":
                        # Validate JSON structure
                        data = response.json()
                        assert "openapi" in data, "OpenAPI spec missing openapi field"
                        assert "paths" in data, "OpenAPI spec missing paths field"
                        
            except requests.exceptions.RequestException:
                pytest.skip(f"API not available for {endpoint} test")

class TestConfigurationValidation:
    """Test configuration file validation."""
    
    def test_nginx_config_exists(self):
        """Test nginx configuration exists."""
        nginx_config = Path("api/nginx.conf")
        assert nginx_config.exists(), "nginx.conf not found"
        
        content = nginx_config.read_text()
        assert "upstream" in content, "nginx.conf missing upstream configuration"
        assert "server" in content, "nginx.conf missing server configuration"
        assert "proxy_pass" in content, "nginx.conf missing proxy_pass configuration"
    
    def test_redis_config_exists(self):
        """Test Redis configuration exists."""
        redis_config = Path("api/redis.conf")
        assert redis_config.exists(), "redis.conf not found"
        
        content = redis_config.read_text()
        assert "port 6379" in content, "redis.conf missing port configuration"
        assert "maxmemory" in content, "redis.conf missing memory configuration"
    
    def test_monitoring_configs_exist(self):
        """Test monitoring configuration files exist."""
        monitoring_files = [
            "api/monitoring/prometheus.yml",
            "api/monitoring/alert_rules.yml",
            "api/monitoring/grafana/provisioning/datasources/prometheus.yml",
            "api/monitoring/grafana/provisioning/dashboards/dashboard.yml"
        ]
        
        for config_file in monitoring_files:
            config_path = Path(config_file)
            assert config_path.exists(), f"Monitoring config not found: {config_file}"

class TestDeploymentScripts:
    """Test deployment scripts functionality."""
    
    def test_deployment_script_exists(self):
        """Test deployment script exists and is executable."""
        deploy_script = Path("api/scripts/deploy.sh")
        assert deploy_script.exists(), "deploy.sh not found"
        
        # Check script has proper shebang
        content = deploy_script.read_text()
        assert content.startswith("#!/bin/bash"), "deploy.sh missing bash shebang"
        
        # Check for key functions
        assert "main()" in content, "deploy.sh missing main function"
        assert "deploy_services()" in content, "deploy.sh missing deploy_services function"
        assert "wait_for_health()" in content, "deploy.sh missing wait_for_health function"
    
    def test_validation_script_exists(self):
        """Test validation script exists."""
        validation_script = Path("api/scripts/validate-deployment.py")
        assert validation_script.exists(), "validate-deployment.py not found"
        
        # Check script has proper shebang
        content = validation_script.read_text()
        assert content.startswith("#!/usr/bin/env python3"), "validate-deployment.py missing python shebang"
        
        # Check for key classes
        assert "class APIValidator" in content, "validate-deployment.py missing APIValidator class"
        assert "def main()" in content, "validate-deployment.py missing main function"
    
    def test_load_test_script_exists(self):
        """Test load test script exists."""
        load_test_script = Path("api/scripts/load-test.js")
        assert load_test_script.exists(), "load-test.js not found"
        
        # Check script content
        content = load_test_script.read_text()
        assert "export const options" in content, "load-test.js missing options export"
        assert "export default function" in content, "load-test.js missing default export function"

class TestCICDPipeline:
    """Test CI/CD pipeline configuration."""
    
    def test_github_workflow_exists(self):
        """Test GitHub Actions workflow exists."""
        workflow_path = Path("api/.github/workflows/ci-cd.yml")
        assert workflow_path.exists(), "CI/CD workflow not found"
        
        content = workflow_path.read_text()
        assert "name: CI/CD Pipeline" in content, "Workflow missing name"
        assert "jobs:" in content, "Workflow missing jobs"
        assert "code-quality:" in content, "Workflow missing code quality job"
        assert "test:" in content, "Workflow missing test job"
        assert "docker-build:" in content, "Workflow missing docker build job"
    
    def test_workflow_has_required_jobs(self):
        """Test workflow has all required jobs."""
        workflow_path = Path("api/.github/workflows/ci-cd.yml")
        if not workflow_path.exists():
            pytest.skip("CI/CD workflow not found")
        
        content = workflow_path.read_text()
        
        required_jobs = [
            "code-quality",
            "test",
            "api-test",
            "docker-build",
            "load-test",
            "security-scan"
        ]
        
        for job in required_jobs:
            assert f"{job}:" in content, f"Workflow missing required job: {job}"

class TestSecurityConfiguration:
    """Test security configuration."""
    
    def test_dockerfile_security_practices(self):
        """Test Dockerfile follows security best practices."""
        dockerfile_path = Path("api/Dockerfile")
        content = dockerfile_path.read_text()
        
        # Check for non-root user
        assert "USER appuser" in content, "Dockerfile not using non-root user"
        
        # Check for proper file permissions
        assert "chown" in content, "Dockerfile not setting proper file ownership"
        
        # Check for minimal base image
        assert "slim" in content, "Dockerfile not using minimal base image"
    
    def test_nginx_security_headers(self):
        """Test nginx configuration includes security headers."""
        nginx_config = Path("api/nginx.conf")
        content = nginx_config.read_text()
        
        security_headers = [
            "X-Frame-Options",
            "X-Content-Type-Options",
            "X-XSS-Protection",
            "Content-Security-Policy"
        ]
        
        for header in security_headers:
            assert header in content, f"nginx.conf missing security header: {header}"
    
    def test_rate_limiting_configured(self):
        """Test rate limiting is configured."""
        nginx_config = Path("api/nginx.conf")
        content = nginx_config.read_text()
        
        assert "limit_req_zone" in content, "nginx.conf missing rate limiting configuration"
        assert "limit_req" in content, "nginx.conf missing rate limiting application"

@pytest.mark.integration
class TestFullDeployment:
    """Integration tests for full deployment."""
    
    @pytest.fixture(scope="class")
    def deployed_service(self):
        """Deploy service for integration testing."""
        # This would typically start the service using docker-compose
        # For now, we'll assume the service is already running
        yield
        # Cleanup would go here
    
    def test_end_to_end_prediction_flow(self, deployed_service):
        """Test complete prediction flow."""
        try:
            # Test health check
            health_response = requests.get(f"{API_BASE_URL}/model/health", timeout=REQUEST_TIMEOUT)
            if health_response.status_code != 200:
                pytest.skip("API not healthy for end-to-end test")
            
            # Test prediction
            sample_customer = {
                "customer_id": "integration_test_001",
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
            
            prediction_response = requests.post(
                f"{API_BASE_URL}/predict",
                json=sample_customer,
                timeout=REQUEST_TIMEOUT
            )
            
            assert prediction_response.status_code == 200, "Prediction request failed"
            
            prediction_data = prediction_response.json()
            assert "churn_probability" in prediction_data, "Prediction missing churn_probability"
            assert "risk_level" in prediction_data, "Prediction missing risk_level"
            
        except requests.exceptions.RequestException:
            pytest.skip("API not available for end-to-end test")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])