"""
Prediction Service for Customer Churn Prediction API.

This module provides the core prediction service with model loading, preprocessing,
churn probability calculation, confidence scoring, and recommendation generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import pickle
import joblib
from datetime import datetime
import json
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

from .config import get_business_config, get_feature_engineering_config
from .logging_setup import get_notebook_logger

logger = get_notebook_logger(__name__)

@dataclass
class CustomerInput:
    """
    Data model for customer input with validation.
    
    Represents all customer data needed for churn prediction including
    demographics, account information, service usage, and billing history.
    """
    customer_id: str
    
    # Demographics
    gender: str
    senior_citizen: bool
    partner: bool
    dependents: bool
    
    # Account information
    tenure: int
    contract: str  # "Month-to-month", "One year", "Two year"
    paperless_billing: bool
    payment_method: str
    
    # Service usage
    phone_service: bool
    multiple_lines: str  # "No", "Yes", "No phone service"
    internet_service: str  # "DSL", "Fiber optic", "No"
    online_security: str  # "No", "Yes", "No internet service"
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    
    # Financial data
    monthly_charges: float
    total_charges: float
    
    # Optional enhanced features (may not be available for all customers)
    usage_minutes_monthly: Optional[float] = None
    data_usage_gb_monthly: Optional[float] = None
    support_interactions_count: Optional[int] = None
    complaint_count: Optional[int] = None
    satisfaction_score: Optional[float] = None
    payment_delay_frequency: Optional[float] = None
    service_change_frequency: Optional[float] = None
    
    # Metadata
    prediction_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Validate input data after initialization."""
        self._validate_input()
        if self.prediction_timestamp is None:
            self.prediction_timestamp = datetime.now().isoformat()
    
    def _validate_input(self):
        """Validate customer input data."""
        # Validate required fields
        if not self.customer_id:
            raise ValueError("customer_id is required")
        
        # Validate categorical fields
        valid_contracts = ["Month-to-month", "One year", "Two year"]
        if self.contract not in valid_contracts:
            raise ValueError(f"contract must be one of {valid_contracts}")
        
        valid_internet_services = ["DSL", "Fiber optic", "No"]
        if self.internet_service not in valid_internet_services:
            raise ValueError(f"internet_service must be one of {valid_internet_services}")
        
        # Validate numeric fields
        if self.tenure < 0:
            raise ValueError("tenure must be non-negative")
        
        if self.monthly_charges < 0:
            raise ValueError("monthly_charges must be non-negative")
        
        if self.total_charges < 0:
            raise ValueError("total_charges must be non-negative")
        
        # Validate optional numeric fields
        if self.usage_minutes_monthly is not None and self.usage_minutes_monthly < 0:
            raise ValueError("usage_minutes_monthly must be non-negative")
        
        if self.data_usage_gb_monthly is not None and self.data_usage_gb_monthly < 0:
            raise ValueError("data_usage_gb_monthly must be non-negative")
        
        if self.support_interactions_count is not None and self.support_interactions_count < 0:
            raise ValueError("support_interactions_count must be non-negative")
        
        if self.complaint_count is not None and self.complaint_count < 0:
            raise ValueError("complaint_count must be non-negative")
        
        if self.satisfaction_score is not None and not (0 <= self.satisfaction_score <= 10):
            raise ValueError("satisfaction_score must be between 0 and 10")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'customer_id': self.customer_id,
            'gender': self.gender,
            'senior_citizen': self.senior_citizen,
            'partner': self.partner,
            'dependents': self.dependents,
            'tenure': self.tenure,
            'contract': self.contract,
            'paperless_billing': self.paperless_billing,
            'payment_method': self.payment_method,
            'phone_service': self.phone_service,
            'multiple_lines': self.multiple_lines,
            'internet_service': self.internet_service,
            'online_security': self.online_security,
            'online_backup': self.online_backup,
            'device_protection': self.device_protection,
            'tech_support': self.tech_support,
            'streaming_tv': self.streaming_tv,
            'streaming_movies': self.streaming_movies,
            'monthly_charges': self.monthly_charges,
            'total_charges': self.total_charges,
            'usage_minutes_monthly': self.usage_minutes_monthly,
            'data_usage_gb_monthly': self.data_usage_gb_monthly,
            'support_interactions_count': self.support_interactions_count,
            'complaint_count': self.complaint_count,
            'satisfaction_score': self.satisfaction_score,
            'payment_delay_frequency': self.payment_delay_frequency,
            'service_change_frequency': self.service_change_frequency,
            'prediction_timestamp': self.prediction_timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomerInput':
        """Create CustomerInput from dictionary."""
        return cls(**data)


@dataclass
class PredictionResult:
    """
    Data model for prediction results with comprehensive risk assessment.
    
    Contains churn probability, risk level classification, confidence scoring,
    key risk factors identification, and actionable recommendations.
    """
    customer_id: str
    churn_probability: float
    risk_level: str  # "low", "medium", "high"
    confidence_score: float
    key_risk_factors: List[str]
    recommendations: List[str]
    model_version: str
    prediction_timestamp: str
    
    # Additional risk metrics
    risk_score: Optional[float] = None
    risk_percentile: Optional[float] = None
    
    # Feature contributions (for explainability)
    feature_contributions: Optional[Dict[str, float]] = None
    
    # Business metrics
    estimated_clv: Optional[float] = None  # Customer Lifetime Value
    retention_cost_benefit: Optional[float] = None
    
    def __post_init__(self):
        """Validate prediction result after initialization."""
        self._validate_result()
    
    def _validate_result(self):
        """Validate prediction result data."""
        if not (0 <= self.churn_probability <= 1):
            raise ValueError("churn_probability must be between 0 and 1")
        
        if not (0 <= self.confidence_score <= 1):
            raise ValueError("confidence_score must be between 0 and 1")
        
        valid_risk_levels = ["low", "medium", "high"]
        if self.risk_level not in valid_risk_levels:
            raise ValueError(f"risk_level must be one of {valid_risk_levels}")
        
        if self.risk_score is not None and not (0 <= self.risk_score <= 100):
            raise ValueError("risk_score must be between 0 and 100")
        
        if self.risk_percentile is not None and not (0 <= self.risk_percentile <= 100):
            raise ValueError("risk_percentile must be between 0 and 100")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'customer_id': self.customer_id,
            'churn_probability': self.churn_probability,
            'risk_level': self.risk_level,
            'confidence_score': self.confidence_score,
            'key_risk_factors': self.key_risk_factors,
            'recommendations': self.recommendations,
            'model_version': self.model_version,
            'prediction_timestamp': self.prediction_timestamp,
            'risk_score': self.risk_score,
            'risk_percentile': self.risk_percentile,
            'feature_contributions': self.feature_contributions,
            'estimated_clv': self.estimated_clv,
            'retention_cost_benefit': self.retention_cost_benefit
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionResult':
        """Create PredictionResult from dictionary."""
        return cls(**data)


class ModelRegistry:
    """
    Simple model registry for loading and managing trained models.
    """
    
    def __init__(self, models_path: Path = Path("models")):
        """
        Initialize model registry.
        
        Args:
            models_path: Path to directory containing saved models
        """
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
        self._model_cache = {}
        self._model_metadata = {}
        
        logger.info(f"ModelRegistry initialized with path: {self.models_path}")
    
    def load_model(self, model_name: str, version: str = "latest") -> Tuple[BaseEstimator, Dict[str, Any]]:
        """
        Load a trained model and its metadata.
        
        Args:
            model_name: Name of the model to load
            version: Version of the model (default: "latest")
            
        Returns:
            Tuple of (model, metadata)
        """
        cache_key = f"{model_name}_{version}"
        
        # Check cache first
        if cache_key in self._model_cache:
            logger.debug(f"Loading model {model_name} v{version} from cache")
            return self._model_cache[cache_key], self._model_metadata[cache_key]
        
        # Construct file paths
        if version == "latest":
            model_files = list(self.models_path.glob(f"{model_name}_*.pkl"))
            if not model_files:
                raise FileNotFoundError(f"No models found for {model_name}")
            
            # Get the most recent model file
            model_file = max(model_files, key=lambda x: x.stat().st_mtime)
            version = model_file.stem.split('_', 1)[1]  # Extract version from filename
        else:
            model_file = self.models_path / f"{model_name}_{version}.pkl"
        
        metadata_file = self.models_path / f"{model_name}_{version}_metadata.json"
        
        # Load model
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Loaded model {model_name} v{version} from {model_file}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name} v{version}: {e}")
            raise
        
        # Load metadata
        metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {model_name} v{version}: {e}")
        
        # Cache the loaded model and metadata
        self._model_cache[cache_key] = model
        self._model_metadata[cache_key] = metadata
        
        return model, metadata
    
    def save_model(
        self,
        model: BaseEstimator,
        model_name: str,
        version: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save a trained model and its metadata.
        
        Args:
            model: Trained model to save
            model_name: Name for the model
            version: Version identifier
            metadata: Optional metadata dictionary
        """
        model_file = self.models_path / f"{model_name}_{version}.pkl"
        metadata_file = self.models_path / f"{model_name}_{version}_metadata.json"
        
        # Save model
        try:
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model {model_name} v{version} to {model_file}")
        except Exception as e:
            logger.error(f"Failed to save model {model_name} v{version}: {e}")
            raise
        
        # Save metadata
        if metadata:
            try:
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"Saved metadata for {model_name} v{version}")
            except Exception as e:
                logger.warning(f"Failed to save metadata for {model_name} v{version}: {e}")
        
        # Update cache
        cache_key = f"{model_name}_{version}"
        self._model_cache[cache_key] = model
        self._model_metadata[cache_key] = metadata or {}
    
    def list_models(self) -> List[Dict[str, str]]:
        """List all available models."""
        model_files = list(self.models_path.glob("*.pkl"))
        models = []
        
        for model_file in model_files:
            name_version = model_file.stem
            if '_' in name_version:
                # Split only on the first underscore to handle versions with underscores
                parts = name_version.split('_')
                if len(parts) >= 2:
                    name = '_'.join(parts[:-1])  # Everything except the last part
                    version = parts[-1]  # Last part is the version
                    models.append({
                        'name': name,
                        'version': version,
                        'file_path': str(model_file),
                        'modified_time': datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                    })
        
        return sorted(models, key=lambda x: x['modified_time'], reverse=True)


class PredictionService:
    """
    Core prediction service for customer churn prediction.
    
    Provides model loading, preprocessing, churn probability calculation,
    confidence scoring, and recommendation generation capabilities.
    """
    
    def __init__(
        self,
        model_registry: Optional[ModelRegistry] = None,
        default_model_name: str = "churn_predictor",
        default_model_version: str = "latest"
    ):
        """
        Initialize prediction service.
        
        Args:
            model_registry: Model registry for loading models
            default_model_name: Default model name to use
            default_model_version: Default model version to use
        """
        self.model_registry = model_registry or ModelRegistry()
        self.default_model_name = default_model_name
        self.default_model_version = default_model_version
        
        # Configuration
        self.business_config = get_business_config()
        self.feature_config = get_feature_engineering_config()
        
        # Model and preprocessing components
        self.current_model = None
        self.current_model_metadata = {}
        self.preprocessor = None
        self.feature_names = []
        
        # Risk thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.7,
            'high': 1.0
        }
        
        # Load default model
        self._load_default_model()
        
        logger.info(f"PredictionService initialized with model: {default_model_name} v{default_model_version}")
    
    def _load_default_model(self):
        """Load the default model and preprocessor."""
        try:
            self.current_model, self.current_model_metadata = self.model_registry.load_model(
                self.default_model_name, self.default_model_version
            )
            
            # Extract feature names from metadata if available
            if 'feature_names' in self.current_model_metadata:
                self.feature_names = self.current_model_metadata['feature_names']
            
            logger.info(f"Loaded default model: {self.default_model_name} v{self.default_model_version}")
            
        except FileNotFoundError:
            logger.warning(f"Default model {self.default_model_name} not found. Service will require explicit model loading.")
            self.current_model = None
            self.current_model_metadata = {}
    
    def load_model(self, model_name: str, version: str = "latest") -> None:
        """
        Load a specific model for predictions.
        
        Args:
            model_name: Name of the model to load
            version: Version of the model to load
        """
        try:
            self.current_model, self.current_model_metadata = self.model_registry.load_model(
                model_name, version
            )
            
            # Update feature names
            if 'feature_names' in self.current_model_metadata:
                self.feature_names = self.current_model_metadata['feature_names']
            
            logger.info(f"Loaded model: {model_name} v{version}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} v{version}: {e}")
            raise
    
    def preprocess_input(self, customer_data: CustomerInput) -> np.ndarray:
        """
        Preprocess customer input data for model prediction.
        
        Args:
            customer_data: Customer input data
            
        Returns:
            Preprocessed feature array
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded. Please load a model first.")
        
        # Convert customer data to DataFrame for easier processing
        data_dict = customer_data.to_dict()
        
        # Remove non-feature fields
        non_feature_fields = ['customer_id', 'prediction_timestamp']
        for field in non_feature_fields:
            data_dict.pop(field, None)
        
        df = pd.DataFrame([data_dict])
        
        # Apply basic preprocessing
        df = self._apply_basic_preprocessing(df)
        
        # Apply feature engineering if configured
        df = self._apply_feature_engineering(df)
        
        # Ensure all required features are present
        df = self._ensure_feature_consistency(df)
        
        # Convert to numpy array
        features = df.values
        
        logger.debug(f"Preprocessed input shape: {features.shape}")
        
        return features
    
    def _apply_basic_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply basic preprocessing steps."""
        df = df.copy()
        
        # Handle missing values with defaults
        if 'usage_minutes_monthly' in df.columns:
            df['usage_minutes_monthly'].fillna(self.feature_config.default_usage_minutes, inplace=True)
        
        if 'data_usage_gb_monthly' in df.columns:
            df['data_usage_gb_monthly'].fillna(self.feature_config.default_data_usage_gb, inplace=True)
        
        # Fill other optional fields with reasonable defaults
        optional_fields = [
            'support_interactions_count', 'complaint_count', 'satisfaction_score',
            'payment_delay_frequency', 'service_change_frequency'
        ]
        
        for field in optional_fields:
            if field in df.columns:
                if field in ['support_interactions_count', 'complaint_count']:
                    df[field].fillna(0, inplace=True)
                elif field == 'satisfaction_score':
                    df[field].fillna(7.0, inplace=True)  # Neutral satisfaction
                else:
                    df[field].fillna(0.0, inplace=True)
        
        # Convert boolean columns to integers
        bool_columns = df.select_dtypes(include=['bool']).columns
        df[bool_columns] = df[bool_columns].astype(int)
        
        # Encode categorical variables
        df = self._encode_categorical_variables(df)
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        df = df.copy()
        
        # Define categorical mappings (these should match training data encoding)
        categorical_mappings = {
            'gender': {'Female': 0, 'Male': 1},
            'contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'payment_method': {
                'Electronic check': 0,
                'Mailed check': 1,
                'Bank transfer (automatic)': 2,
                'Credit card (automatic)': 3
            },
            'internet_service': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
            'multiple_lines': {'No': 0, 'Yes': 1, 'No phone service': 2},
            'online_security': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'online_backup': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'device_protection': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'tech_support': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'streaming_tv': {'No': 0, 'Yes': 1, 'No internet service': 2},
            'streaming_movies': {'No': 0, 'Yes': 1, 'No internet service': 2}
        }
        
        # Apply mappings
        for column, mapping in categorical_mappings.items():
            if column in df.columns:
                df[column] = df[column].map(mapping)
                # Handle unmapped values
                if df[column].isna().any():
                    logger.warning(f"Unmapped values found in {column}, using default encoding")
                    df[column].fillna(0, inplace=True)
        
        return df
    
    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        df = df.copy()
        
        # Create usage ratio features if usage data is available
        if 'usage_minutes_monthly' in df.columns and 'monthly_charges' in df.columns:
            # Avoid division by zero
            df['usage_efficiency_ratio'] = np.where(
                df['monthly_charges'] > self.feature_config.min_charges_threshold,
                df['usage_minutes_monthly'] / df['monthly_charges'],
                0
            )
        
        if 'data_usage_gb_monthly' in df.columns and 'monthly_charges' in df.columns:
            df['cost_per_gb_ratio'] = np.where(
                (df['data_usage_gb_monthly'] > 0) & (df['monthly_charges'] > self.feature_config.min_charges_threshold),
                df['monthly_charges'] / df['data_usage_gb_monthly'],
                0
            )
        
        # Create tenure bands
        if 'tenure' in df.columns:
            df['tenure_band'] = pd.cut(
                df['tenure'],
                bins=self.feature_config.tenure_bins,
                labels=range(len(self.feature_config.tenure_bins) - 1),
                include_lowest=True
            ).astype(int)
        
        # Create service count feature
        service_columns = [
            'phone_service', 'multiple_lines', 'internet_service',
            'online_security', 'online_backup', 'device_protection',
            'tech_support', 'streaming_tv', 'streaming_movies'
        ]
        
        available_service_columns = [col for col in service_columns if col in df.columns]
        if available_service_columns:
            # Count active services (assuming 1 means active, 0 means inactive)
            df['service_count'] = df[available_service_columns].sum(axis=1)
        
        # Create value efficiency ratio
        if 'monthly_charges' in df.columns and 'service_count' in df.columns:
            df['value_per_service_ratio'] = np.where(
                df['service_count'] > 0,
                df['monthly_charges'] / df['service_count'],
                df['monthly_charges']
            )
        
        return df
    
    def _ensure_feature_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure feature consistency with training data."""
        if not self.feature_names:
            logger.warning("No feature names available from model metadata")
            return df
        
        # Add missing features with default values
        for feature in self.feature_names:
            if feature not in df.columns:
                logger.debug(f"Adding missing feature {feature} with default value 0")
                df[feature] = 0
        
        # Remove extra features not in training data
        extra_features = set(df.columns) - set(self.feature_names)
        if extra_features:
            logger.debug(f"Removing extra features: {extra_features}")
            df = df.drop(columns=list(extra_features))
        
        # Reorder columns to match training data
        df = df[self.feature_names]
        
        return df
    
    def predict_churn_probability(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Predict churn probability with confidence scoring.
        
        Args:
            features: Preprocessed feature array
            
        Returns:
            Tuple of (churn_probability, confidence_score)
        """
        if self.current_model is None:
            raise RuntimeError("No model loaded. Please load a model first.")
        
        try:
            # Get prediction probabilities
            if hasattr(self.current_model, 'predict_proba'):
                probabilities = self.current_model.predict_proba(features)
                churn_probability = probabilities[0, 1]  # Probability of churn (class 1)
                
                # Calculate confidence score based on probability distribution
                confidence_score = self._calculate_confidence_score(probabilities[0])
                
            elif hasattr(self.current_model, 'decision_function'):
                # For models with decision function (like SVM)
                decision_score = self.current_model.decision_function(features)[0]
                churn_probability = 1 / (1 + np.exp(-decision_score))  # Sigmoid transformation
                
                # Confidence based on distance from decision boundary
                confidence_score = min(abs(decision_score) / 3.0, 1.0)  # Normalize to [0, 1]
                
            else:
                # Fallback for models without probability prediction
                prediction = self.current_model.predict(features)[0]
                churn_probability = float(prediction)
                confidence_score = 0.5  # Low confidence for binary predictions
            
            logger.debug(f"Predicted churn probability: {churn_probability:.3f}, confidence: {confidence_score:.3f}")
            
            return churn_probability, confidence_score
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _calculate_confidence_score(self, probabilities: np.ndarray) -> float:
        """
        Calculate confidence score based on prediction probabilities.
        
        Args:
            probabilities: Array of class probabilities
            
        Returns:
            Confidence score between 0 and 1
        """
        # Confidence is higher when the model is more certain (probabilities closer to 0 or 1)
        max_prob = np.max(probabilities)
        
        # Transform to confidence score: higher when max_prob is closer to 1
        confidence = 2 * max_prob - 1  # Maps [0.5, 1] to [0, 1]
        confidence = max(0, confidence)  # Ensure non-negative
        
        return confidence
    
    def generate_recommendations(self, prediction_result: PredictionResult, customer_data: CustomerInput) -> List[str]:
        """
        Generate actionable recommendations based on risk factors and customer profile.
        
        Args:
            prediction_result: Prediction result with risk assessment
            customer_data: Original customer data
            
        Returns:
            List of actionable recommendations
        """
        recommendations = []
        
        # Risk-level based recommendations
        if prediction_result.risk_level == "high":
            recommendations.extend([
                "Immediate retention intervention recommended",
                "Consider offering personalized discount or upgrade incentive",
                "Schedule proactive customer service outreach within 48 hours"
            ])
        elif prediction_result.risk_level == "medium":
            recommendations.extend([
                "Monitor customer engagement closely",
                "Consider targeted marketing campaign for service upgrades",
                "Proactive customer satisfaction survey recommended"
            ])
        else:  # low risk
            recommendations.extend([
                "Continue standard customer engagement",
                "Consider upselling opportunities for additional services"
            ])
        
        # Contract-specific recommendations
        if customer_data.contract == "Month-to-month":
            recommendations.append("Offer incentive to upgrade to longer-term contract")
        
        # Service-specific recommendations
        if customer_data.internet_service == "Fiber optic" and prediction_result.churn_probability > 0.5:
            recommendations.append("Review fiber service quality and customer satisfaction")
        
        if not customer_data.phone_service and customer_data.internet_service != "No":
            recommendations.append("Consider bundled phone and internet service offer")
        
        # Payment method recommendations
        if customer_data.payment_method == "Electronic check":
            recommendations.append("Encourage switch to automatic payment method with discount incentive")
        
        # Support-based recommendations
        if customer_data.support_interactions_count and customer_data.support_interactions_count > 3:
            recommendations.append("Review recent support interactions and address recurring issues")
        
        # Tenure-based recommendations
        if customer_data.tenure < 12:
            recommendations.append("Implement new customer retention program")
        elif customer_data.tenure > 60:
            recommendations.append("Recognize customer loyalty with exclusive offers")
        
        # Financial recommendations
        if customer_data.monthly_charges > 80:
            recommendations.append("Review pricing competitiveness and consider retention pricing")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)
        
        return unique_recommendations[:8]  # Limit to top 8 recommendations
    
    def identify_key_risk_factors(self, features: np.ndarray, customer_data: CustomerInput) -> List[str]:
        """
        Identify key risk factors contributing to churn probability.
        
        Args:
            features: Preprocessed feature array
            customer_data: Original customer data
            
        Returns:
            List of key risk factors
        """
        risk_factors = []
        
        # Contract risk factors
        if customer_data.contract == "Month-to-month":
            risk_factors.append("Month-to-month contract (high flexibility risk)")
        
        # Tenure risk factors
        if customer_data.tenure < 6:
            risk_factors.append("Very new customer (< 6 months tenure)")
        elif customer_data.tenure < 12:
            risk_factors.append("New customer (< 1 year tenure)")
        
        # Service risk factors
        if customer_data.internet_service == "Fiber optic":
            risk_factors.append("Fiber optic service (higher churn rate)")
        
        if not customer_data.online_security or customer_data.online_security == "No":
            risk_factors.append("No online security service")
        
        if not customer_data.tech_support or customer_data.tech_support == "No":
            risk_factors.append("No tech support service")
        
        # Payment risk factors
        if customer_data.payment_method == "Electronic check":
            risk_factors.append("Electronic check payment method")
        
        if customer_data.paperless_billing:
            risk_factors.append("Paperless billing enabled")
        
        # Financial risk factors
        if customer_data.monthly_charges > 80:
            risk_factors.append("High monthly charges (> $80)")
        
        # Support interaction risk factors
        if customer_data.support_interactions_count and customer_data.support_interactions_count > 2:
            risk_factors.append("Multiple recent support interactions")
        
        # Demographic risk factors
        if customer_data.senior_citizen:
            risk_factors.append("Senior citizen demographic")
        
        if not customer_data.partner and not customer_data.dependents:
            risk_factors.append("Single customer without dependents")
        
        return risk_factors[:6]  # Limit to top 6 risk factors
    
    def predict(
        self,
        customer_data: CustomerInput,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> PredictionResult:
        """
        Perform complete churn prediction with risk assessment and recommendations.
        
        Args:
            customer_data: Customer input data
            model_name: Optional specific model name to use
            model_version: Optional specific model version to use
            
        Returns:
            Complete prediction result with risk assessment and recommendations
        """
        # Load specific model if requested
        if model_name and (model_name != self.default_model_name or model_version != self.default_model_version):
            self.load_model(model_name, model_version or "latest")
        
        # Preprocess input
        features = self.preprocess_input(customer_data)
        
        # Get prediction and confidence
        churn_probability, confidence_score = self.predict_churn_probability(features)
        
        # Determine risk level
        risk_level = self._determine_risk_level(churn_probability)
        
        # Identify key risk factors
        key_risk_factors = self.identify_key_risk_factors(features, customer_data)
        
        # Create initial prediction result
        prediction_result = PredictionResult(
            customer_id=customer_data.customer_id,
            churn_probability=churn_probability,
            risk_level=risk_level,
            confidence_score=confidence_score,
            key_risk_factors=key_risk_factors,
            recommendations=[],  # Will be filled by generate_recommendations
            model_version=self.current_model_metadata.get('version', 'unknown'),
            prediction_timestamp=datetime.now().isoformat(),
            risk_score=churn_probability * 100,
            estimated_clv=self._estimate_customer_lifetime_value(customer_data),
            retention_cost_benefit=self._calculate_retention_cost_benefit(churn_probability, customer_data)
        )
        
        # Generate recommendations
        prediction_result.recommendations = self.generate_recommendations(prediction_result, customer_data)
        
        logger.info(
            f"Prediction completed for customer {customer_data.customer_id}: "
            f"probability={churn_probability:.3f}, risk_level={risk_level}, confidence={confidence_score:.3f}"
        )
        
        return prediction_result
    
    def _determine_risk_level(self, churn_probability: float) -> str:
        """Determine risk level based on churn probability."""
        if churn_probability <= self.risk_thresholds['low']:
            return "low"
        elif churn_probability <= self.risk_thresholds['medium']:
            return "medium"
        else:
            return "high"
    
    def _estimate_customer_lifetime_value(self, customer_data: CustomerInput) -> float:
        """Estimate customer lifetime value based on current charges and tenure."""
        # Simple CLV estimation: monthly_charges * expected_remaining_months
        base_clv = customer_data.monthly_charges * 24  # Assume 2-year horizon
        
        # Adjust based on contract type
        if customer_data.contract == "Two year":
            base_clv *= 1.2  # Higher value for committed customers
        elif customer_data.contract == "Month-to-month":
            base_clv *= 0.8  # Lower value for flexible customers
        
        # Adjust based on tenure (loyal customers have higher CLV)
        if customer_data.tenure > 36:
            base_clv *= 1.3
        elif customer_data.tenure > 12:
            base_clv *= 1.1
        
        return base_clv
    
    def _calculate_retention_cost_benefit(self, churn_probability: float, customer_data: CustomerInput) -> float:
        """Calculate the cost-benefit of retention intervention."""
        estimated_clv = self._estimate_customer_lifetime_value(customer_data)
        retention_cost = self.business_config.customer_retention_cost
        
        # Expected benefit = probability of churn * CLV * intervention success rate
        expected_benefit = churn_probability * estimated_clv * 0.7  # Assume 70% intervention success rate
        
        # Net benefit = expected benefit - retention cost
        net_benefit = expected_benefit - retention_cost
        
        return net_benefit
    
    def batch_predict(self, customers: List[CustomerInput]) -> List[PredictionResult]:
        """
        Perform batch predictions for multiple customers.
        
        Args:
            customers: List of customer input data
            
        Returns:
            List of prediction results
        """
        logger.info(f"Starting batch prediction for {len(customers)} customers")
        
        results = []
        for customer in customers:
            try:
                result = self.predict(customer)
                results.append(result)
            except Exception as e:
                logger.error(f"Prediction failed for customer {customer.customer_id}: {e}")
                # Create error result
                error_result = PredictionResult(
                    customer_id=customer.customer_id,
                    churn_probability=0.0,
                    risk_level="low",  # Use valid risk level instead of "unknown"
                    confidence_score=0.0,
                    key_risk_factors=["Prediction failed"],
                    recommendations=["Manual review required"],
                    model_version="error",
                    prediction_timestamp=datetime.now().isoformat()
                )
                results.append(error_result)
        
        logger.info(f"Batch prediction completed. {len(results)} results generated")
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the currently loaded model."""
        if self.current_model is None:
            return {"error": "No model loaded"}
        
        return {
            "model_name": self.default_model_name,
            "model_version": self.current_model_metadata.get('version', 'unknown'),
            "model_type": type(self.current_model).__name__,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "metadata": self.current_model_metadata,
            "risk_thresholds": self.risk_thresholds
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the prediction service."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_loaded": self.current_model is not None,
            "model_name": self.default_model_name if self.current_model else None,
            "model_version": self.current_model_metadata.get('version', 'unknown') if self.current_model else None,
            "feature_count": len(self.feature_names),
            "issues": []
        }
        
        # Check for potential issues
        if self.current_model is None:
            health_status["status"] = "unhealthy"
            health_status["issues"].append("No model loaded")
        
        if not self.feature_names:
            health_status["issues"].append("No feature names available")
        
        return health_status


# Convenience functions for quick usage
def create_sample_customer() -> CustomerInput:
    """Create a sample customer for testing purposes."""
    return CustomerInput(
        customer_id="SAMPLE_001",
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
        usage_minutes_monthly=450.0,
        data_usage_gb_monthly=15.5,
        support_interactions_count=1,
        complaint_count=0,
        satisfaction_score=8.0
    )


def quick_prediction(customer_data: CustomerInput, model_name: str = "churn_predictor", model_version: str = "latest") -> PredictionResult:
    """
    Quick prediction function for single customer.
    
    Args:
        customer_data: Customer input data
        model_name: Name of model to use
        model_version: Version of model to use
        
    Returns:
        Prediction result
    """
    kwargs = {"default_model_name": model_name}
    if model_version != "latest":
        kwargs["default_model_version"] = model_version
    service = PredictionService(**kwargs)
    return service.predict(customer_data)