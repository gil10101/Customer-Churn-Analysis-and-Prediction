"""
Prediction Service Demo

This script demonstrates how to use the PredictionService for customer churn prediction.
It shows the complete workflow from creating customer data to getting predictions with
recommendations.
"""

import sys
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

from utils.prediction_service import (
    CustomerInput, PredictionResult, ModelRegistry, PredictionService,
    create_sample_customer, quick_prediction
)
from utils.logging_setup import get_notebook_logger

# Set up logging
logger = get_notebook_logger("prediction_service_demo")

def create_demo_model():
    """Create a simple demo model for testing the prediction service."""
    logger.info("Creating demo model...")
    
    # Create synthetic training data that matches our feature schema
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'gender': np.random.choice([0, 1], n_samples),
        'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'partner': np.random.choice([0, 1], n_samples),
        'dependents': np.random.choice([0, 1], n_samples),
        'tenure': np.random.randint(1, 73, n_samples),
        'contract': np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2]),
        'paperless_billing': np.random.choice([0, 1], n_samples),
        'payment_method': np.random.choice([0, 1, 2, 3], n_samples),
        'phone_service': np.random.choice([0, 1], n_samples, p=[0.1, 0.9]),
        'multiple_lines': np.random.choice([0, 1, 2], n_samples),
        'internet_service': np.random.choice([0, 1, 2], n_samples, p=[0.2, 0.4, 0.4]),
        'online_security': np.random.choice([0, 1, 2], n_samples),
        'online_backup': np.random.choice([0, 1, 2], n_samples),
        'device_protection': np.random.choice([0, 1, 2], n_samples),
        'tech_support': np.random.choice([0, 1, 2], n_samples),
        'streaming_tv': np.random.choice([0, 1, 2], n_samples),
        'streaming_movies': np.random.choice([0, 1, 2], n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(20, 8000, n_samples)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create target variable with some logic to make it realistic
    # Higher churn probability for:
    # - Month-to-month contracts (contract == 0)
    # - Higher monthly charges
    # - Lower tenure
    # - Electronic check payment (payment_method == 0)
    
    churn_prob = (
        0.3 +  # Base probability
        0.4 * (df['contract'] == 0) +  # Month-to-month contract
        0.2 * (df['monthly_charges'] > 80) +  # High charges
        0.3 * (df['tenure'] < 12) +  # New customers
        0.2 * (df['payment_method'] == 0) +  # Electronic check
        0.1 * (df['senior_citizen'] == 1) -  # Senior citizens
        0.2 * (df['partner'] == 1) -  # Has partner
        0.1 * (df['dependents'] == 1)  # Has dependents
    )
    
    # Add some noise and clip to [0, 1]
    churn_prob += np.random.normal(0, 0.1, n_samples)
    churn_prob = np.clip(churn_prob, 0, 1)
    
    # Convert to binary labels
    y = (churn_prob > 0.5).astype(int)
    
    logger.info(f"Generated {n_samples} samples with {y.mean():.1%} churn rate")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    logger.info(f"Model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
    
    # Create metadata
    metadata = {
        'version': 'demo_v1.0',
        'created_date': pd.Timestamp.now().isoformat(),
        'feature_names': list(df.columns),
        'model_type': 'RandomForestClassifier',
        'performance_metrics': {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'n_samples': n_samples,
            'churn_rate': float(y.mean())
        },
        'description': 'Demo model for prediction service testing'
    }
    
    return model, metadata

def save_demo_model():
    """Create and save a demo model."""
    logger.info("Setting up demo model...")
    
    # Create model
    model, metadata = create_demo_model()
    
    # Save model using ModelRegistry
    registry = ModelRegistry()
    registry.save_model(model, "churn_predictor", "demo_v1.0", metadata)
    
    logger.info("Demo model saved successfully!")
    return registry

def demo_single_prediction():
    """Demonstrate single customer prediction."""
    logger.info("\n" + "="*60)
    logger.info("SINGLE CUSTOMER PREDICTION DEMO")
    logger.info("="*60)
    
    # Create prediction service
    service = PredictionService(default_model_name="churn_predictor", default_model_version="demo_v1.0")
    
    # Create sample customer
    customer = create_sample_customer()
    
    # Modify some attributes to make it more interesting
    customer.customer_id = "DEMO_CUSTOMER_001"
    customer.contract = "Month-to-month"
    customer.tenure = 6
    customer.monthly_charges = 95.0
    customer.payment_method = "Electronic check"
    customer.support_interactions_count = 3
    
    logger.info(f"Predicting churn for customer: {customer.customer_id}")
    logger.info(f"Customer profile:")
    logger.info(f"  - Contract: {customer.contract}")
    logger.info(f"  - Tenure: {customer.tenure} months")
    logger.info(f"  - Monthly charges: ${customer.monthly_charges}")
    logger.info(f"  - Payment method: {customer.payment_method}")
    logger.info(f"  - Support interactions: {customer.support_interactions_count}")
    
    # Make prediction
    result = service.predict(customer)
    
    # Display results
    logger.info(f"\nPREDICTION RESULTS:")
    logger.info(f"  - Churn probability: {result.churn_probability:.1%}")
    logger.info(f"  - Risk level: {result.risk_level.upper()}")
    logger.info(f"  - Confidence score: {result.confidence_score:.1%}")
    logger.info(f"  - Estimated CLV: ${result.estimated_clv:,.2f}")
    logger.info(f"  - Retention cost-benefit: ${result.retention_cost_benefit:,.2f}")
    
    logger.info(f"\nKEY RISK FACTORS:")
    for i, factor in enumerate(result.key_risk_factors, 1):
        logger.info(f"  {i}. {factor}")
    
    logger.info(f"\nRECOMMENDATIONS:")
    for i, rec in enumerate(result.recommendations, 1):
        logger.info(f"  {i}. {rec}")
    
    return result

def demo_batch_prediction():
    """Demonstrate batch prediction."""
    logger.info("\n" + "="*60)
    logger.info("BATCH PREDICTION DEMO")
    logger.info("="*60)
    
    # Create prediction service
    service = PredictionService(default_model_name="churn_predictor", default_model_version="demo_v1.0")
    
    # Create multiple customers with different profiles
    customers = []
    
    # High-risk customer
    high_risk = create_sample_customer()
    high_risk.customer_id = "HIGH_RISK_001"
    high_risk.contract = "Month-to-month"
    high_risk.tenure = 3
    high_risk.monthly_charges = 110.0
    high_risk.payment_method = "Electronic check"
    high_risk.support_interactions_count = 5
    high_risk.satisfaction_score = 4.0
    customers.append(high_risk)
    
    # Medium-risk customer
    medium_risk = create_sample_customer()
    medium_risk.customer_id = "MEDIUM_RISK_002"
    medium_risk.contract = "One year"
    medium_risk.tenure = 18
    medium_risk.monthly_charges = 75.0
    medium_risk.payment_method = "Credit card (automatic)"
    medium_risk.support_interactions_count = 1
    medium_risk.satisfaction_score = 7.0
    customers.append(medium_risk)
    
    # Low-risk customer
    low_risk = create_sample_customer()
    low_risk.customer_id = "LOW_RISK_003"
    low_risk.contract = "Two year"
    low_risk.tenure = 48
    low_risk.monthly_charges = 55.0
    low_risk.payment_method = "Bank transfer (automatic)"
    low_risk.support_interactions_count = 0
    low_risk.satisfaction_score = 9.0
    customers.append(low_risk)
    
    logger.info(f"Making batch predictions for {len(customers)} customers...")
    
    # Make batch predictions
    results = service.batch_predict(customers)
    
    # Display results
    logger.info(f"\nBATCH PREDICTION RESULTS:")
    logger.info("-" * 80)
    
    for result in results:
        logger.info(f"Customer: {result.customer_id}")
        logger.info(f"  Churn Probability: {result.churn_probability:.1%}")
        logger.info(f"  Risk Level: {result.risk_level.upper()}")
        logger.info(f"  Confidence: {result.confidence_score:.1%}")
        logger.info(f"  Top Risk Factor: {result.key_risk_factors[0] if result.key_risk_factors else 'None'}")
        logger.info(f"  Top Recommendation: {result.recommendations[0] if result.recommendations else 'None'}")
        logger.info("-" * 80)
    
    return results

def demo_model_info():
    """Demonstrate model information and health check."""
    logger.info("\n" + "="*60)
    logger.info("MODEL INFORMATION DEMO")
    logger.info("="*60)
    
    # Create prediction service
    service = PredictionService(default_model_name="churn_predictor", default_model_version="demo_v1.0")
    
    # Get model info
    model_info = service.get_model_info()
    
    logger.info("MODEL INFORMATION:")
    logger.info(f"  - Model Name: {model_info['model_name']}")
    logger.info(f"  - Model Version: {model_info['model_version']}")
    logger.info(f"  - Model Type: {model_info['model_type']}")
    logger.info(f"  - Feature Count: {model_info['feature_count']}")
    logger.info(f"  - Risk Thresholds: {model_info['risk_thresholds']}")
    
    # Get health check
    health = service.health_check()
    
    logger.info(f"\nHEALTH CHECK:")
    logger.info(f"  - Status: {health['status'].upper()}")
    logger.info(f"  - Model Loaded: {health['model_loaded']}")
    logger.info(f"  - Feature Count: {health['feature_count']}")
    if health['issues']:
        logger.info(f"  - Issues: {', '.join(health['issues'])}")
    else:
        logger.info(f"  - Issues: None")
    
    return model_info, health

def demo_quick_prediction():
    """Demonstrate quick prediction function."""
    logger.info("\n" + "="*60)
    logger.info("QUICK PREDICTION DEMO")
    logger.info("="*60)
    
    # Create a customer
    customer = create_sample_customer()
    customer.customer_id = "QUICK_DEMO_001"
    
    logger.info(f"Using quick_prediction() function for customer: {customer.customer_id}")
    
    # Use quick prediction function - but we need to handle the version issue
    try:
        result = quick_prediction(customer, "churn_predictor")
    except Exception as e:
        logger.warning(f"Quick prediction failed with default version: {e}")
        logger.info("Using direct service call instead...")
        service = PredictionService(default_model_name="churn_predictor", default_model_version="demo_v1.0")
        result = service.predict(customer)
    
    logger.info(f"Quick prediction result:")
    logger.info(f"  - Churn probability: {result.churn_probability:.1%}")
    logger.info(f"  - Risk level: {result.risk_level}")
    logger.info(f"  - Confidence: {result.confidence_score:.1%}")
    
    return result

def main():
    """Run all prediction service demos."""
    logger.info("Starting Prediction Service Demo")
    logger.info("="*80)
    
    try:
        # Set up demo model
        registry = save_demo_model()
        
        # Run demos
        single_result = demo_single_prediction()
        batch_results = demo_batch_prediction()
        model_info, health = demo_model_info()
        quick_result = demo_quick_prediction()
        
        logger.info("\n" + "="*80)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        
        # Summary
        logger.info(f"\nSUMMARY:")
        logger.info(f"  - Single prediction: {single_result.risk_level} risk")
        logger.info(f"  - Batch predictions: {len(batch_results)} customers processed")
        logger.info(f"  - Model health: {health['status']}")
        logger.info(f"  - Quick prediction: {quick_result.risk_level} risk")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise

if __name__ == "__main__":
    main()