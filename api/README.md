# Customer Churn Prediction API

A FastAPI-based REST API serving the trained churn model: per-customer risk scores, batch predictions, and recommendation payloads.

## Features

- **Single Customer Predictions**: Get churn probability and risk assessment for individual customers
- **Batch Predictions**: Process multiple customers efficiently in a single request
- **Model Information**: Access model metadata, performance metrics, and feature information
- **Health Monitoring**: Built-in health checks and service status monitoring
- **Feedback Loop**: Submit prediction outcomes for model improvement
- **Comprehensive Error Handling**: Robust error handling with detailed error messages
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation
- **Docker Support**: Containerized deployment with Docker and docker-compose

## Quick Start

### Prerequisites

- Python 3.9+
- Required packages (see `requirements.txt`)
- Trained churn prediction model (see model training examples)

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have a trained model in the `models/` directory

3. Start the API server:
```bash
python api/run_server.py
```

The API will be available at `http://localhost:8000`

### Docker Deployment

1. Build and run with docker-compose:
```bash
cd api
docker-compose up --build
```

2. For production with nginx:
```bash
docker-compose --profile production up --build
```

## API Endpoints

### Core Prediction Endpoints

#### Single Customer Prediction
```http
POST /predict
```

Predict churn probability for a single customer.

**Request Body:**
```json
{
  "customer_id": "CUST_001",
  "gender": "Female",
  "senior_citizen": false,
  "partner": true,
  "dependents": false,
  "tenure": 24,
  "contract": "One year",
  "paperless_billing": true,
  "payment_method": "Credit card (automatic)",
  "phone_service": true,
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
```

**Response:**
```json
{
  "customer_id": "CUST_001",
  "churn_probability": 0.65,
  "risk_level": "medium",
  "confidence_score": 0.82,
  "key_risk_factors": [
    "Medium tenure customer",
    "Fiber optic service"
  ],
  "recommendations": [
    "Monitor customer engagement closely",
    "Consider targeted retention campaign"
  ],
  "model_version": "v1.0",
  "prediction_timestamp": "2024-01-15T10:30:00",
  "estimated_clv": 2400.0,
  "retention_cost_benefit": 180.0
}
```

#### Batch Predictions
```http
POST /predict/batch
```

Process multiple customers in a single request (up to 1000 customers).

**Request Body:**
```json
{
  "customers": [
    {
      "customer_id": "CUST_001",
      // ... customer data
    },
    {
      "customer_id": "CUST_002",
      // ... customer data
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "customer_id": "CUST_001",
      "churn_probability": 0.65,
      // ... prediction details
    }
  ],
  "batch_id": "batch_20240115_103000_2",
  "processed_count": 2,
  "failed_count": 0,
  "processing_time_seconds": 0.15,
  "errors": []
}
```

### Model Information Endpoints

#### Model Information
```http
GET /model/info
```

Get information about the currently loaded model.

**Response:**
```json
{
  "model_name": "churn_predictor",
  "model_version": "v1.0",
  "model_type": "RandomForestClassifier",
  "feature_count": 19,
  "feature_names": ["gender", "tenure", "monthly_charges", ...],
  "risk_thresholds": {
    "low": 0.3,
    "medium": 0.7,
    "high": 1.0
  },
  "performance_metrics": {
    "accuracy": 0.85,
    "auc": 0.92
  }
}
```

#### Health Check
```http
GET /model/health
```

Check the health status of the prediction service.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "model_loaded": true,
  "feature_count": 19,
  "uptime_seconds": 3600,
  "issues": []
}
```

### Feedback Endpoint

#### Submit Feedback
```http
POST /model/feedback
```

Submit feedback on prediction accuracy for model improvement.

**Request Body:**
```json
{
  "customer_id": "CUST_001",
  "prediction_id": "pred_123",
  "actual_churn": false,
  "intervention_applied": true,
  "intervention_type": "loyalty_discount",
  "outcome_date": "2024-02-15",
  "notes": "Customer retained after discount offer"
}
```

## Usage Examples

### Python Client

```python
import requests

# Initialize client
base_url = "http://localhost:8000"

# Check health
health = requests.get(f"{base_url}/model/health").json()
print(f"API Status: {health['status']}")

# Make prediction
customer_data = {
    "customer_id": "EXAMPLE_001",
    "gender": "Female",
    "senior_citizen": False,
    # ... other fields
}

response = requests.post(f"{base_url}/predict", json=customer_data)
prediction = response.json()

print(f"Churn Probability: {prediction['churn_probability']:.1%}")
print(f"Risk Level: {prediction['risk_level']}")
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/model/health"

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d @customer_data.json

# Model information
curl -X GET "http://localhost:8000/model/info"
```

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI JSON**: `http://localhost:8000/openapi.json`

## Configuration

### Environment Variables

- `LOG_LEVEL`: Logging level (default: "info")
- `PYTHONPATH`: Python path for imports
- `MODEL_PATH`: Path to model files (default: "models")

### Server Configuration

The server can be configured via command line arguments:

```bash
python api/run_server.py --help
```

Options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--log-level`: Log level (debug, info, warning, error)
- `--workers`: Number of worker processes

## Error Handling

The API provides comprehensive error handling with structured error responses:

### Validation Errors (422)
```json
{
  "error": "Validation Error",
  "detail": "contract must be one of ['Month-to-month', 'One year', 'Two year']",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Service Unavailable (503)
```json
{
  "error": "Prediction service not available. Check service health.",
  "detail": "Prediction service not available. Check service health.",
  "timestamp": "2024-01-15T10:30:00"
}
```

### Internal Server Error (500)
```json
{
  "error": "Internal Server Error",
  "detail": "An unexpected error occurred",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Performance

### Response Times
- Single prediction: ~50-200ms
- Batch prediction (100 customers): ~500ms-2s
- Health check: ~10-50ms
- Model info: ~10-50ms

### Throughput
- Single predictions: ~100-500 requests/second
- Batch predictions: ~1000-5000 customers/second

### Scalability
- Horizontal scaling with multiple workers
- Load balancing with nginx
- Container orchestration with Docker Swarm or Kubernetes

## Security Considerations

### Production Deployment
1. **HTTPS**: Use HTTPS in production
2. **Authentication**: Implement API key or OAuth authentication
3. **Rate Limiting**: Add rate limiting to prevent abuse
4. **Input Validation**: All inputs are validated using Pydantic models
5. **CORS**: Configure CORS appropriately for your domain
6. **Secrets Management**: Use environment variables for sensitive configuration

### Docker Security
- Non-root user in container
- Read-only model files
- Health checks for monitoring
- Resource limits in production

## Monitoring and Logging

### Health Monitoring
- Built-in health check endpoint
- Model loading status
- Feature count validation
- Issue reporting

### Logging
- Structured logging with timestamps
- Request/response logging
- Error tracking
- Performance metrics

### Metrics
- Response times
- Request counts
- Error rates
- Model performance

## Development

### Running Tests
```bash
# Run API integration tests
pytest tests/test_api_integration.py -v

# Run all tests
pytest tests/ -v
```

### Development Server
```bash
# Start with auto-reload
python api/run_server.py --reload --log-level debug
```

### Code Quality
```bash
# Format code
black api/

# Lint code
flake8 api/

# Type checking
mypy api/
```

## Troubleshooting

### Common Issues

1. **Model Not Found**
   - Ensure trained model exists in `models/` directory
   - Check model file naming convention
   - Verify model registry configuration

2. **Import Errors**
   - Check PYTHONPATH environment variable
   - Ensure all dependencies are installed
   - Verify project structure

3. **Port Already in Use**
   - Change port with `--port` argument
   - Kill existing processes on port 8000

4. **Docker Issues**
   - Check Docker daemon is running
   - Verify volume mounts for models directory
   - Check container logs: `docker-compose logs`

### Debug Mode
```bash
# Start with debug logging
python api/run_server.py --log-level debug --reload
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Use type hints
5. Add logging for important operations

## License

This project is part of the Customer Churn Prediction system.