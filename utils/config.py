"""
Comprehensive configuration management system for notebook modernization.
Provides centralized configuration with type hints and dataclasses.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DataConfig:
    """Configuration for data processing and analysis."""
    data_path: Path = field(default_factory=lambda: Path("data"))
    raw_data_file: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    processed_data_path: Path = field(default_factory=lambda: Path("data/processed"))
    random_seed: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    
    def __post_init__(self):
        """Ensure paths exist."""
        self.data_path.mkdir(exist_ok=True)
        self.processed_data_path.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration for machine learning models."""
    model_output_path: Path = field(default_factory=lambda: Path("models"))
    cross_validation_folds: int = 5
    hyperparameter_trials: int = 100
    early_stopping_patience: int = 10
    max_training_epochs: int = 1000
    
    # PyTorch specific
    device: str = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
    batch_size: int = 64
    learning_rate: float = 0.001
    
    # Ensemble configuration
    ensemble_methods: List[str] = field(default_factory=lambda: [
        "VotingClassifier", "StackingClassifier", "BaggingClassifier"
    ])
    
    def __post_init__(self):
        """Ensure model output directory exists."""
        self.model_output_path.mkdir(exist_ok=True)

@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    figure_output_path: Path = field(default_factory=lambda: Path("figures"))
    figure_format: str = "png"
    figure_dpi: int = 300
    figure_size: Tuple[int, int] = (12, 8)
    
    # Plotly configuration
    plotly_theme: str = "plotly_white"
    plotly_color_sequence: List[str] = field(default_factory=lambda: [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ])
    
    # Matplotlib/Seaborn configuration
    matplotlib_style: str = "seaborn-v0_8"
    seaborn_palette: str = "husl"
    
    def __post_init__(self):
        """Ensure figure output directory exists."""
        self.figure_output_path.mkdir(exist_ok=True)

@dataclass
class StatisticalConfig:
    """Configuration for statistical analysis."""
    confidence_level: float = 0.95
    significance_level: float = 0.05
    bootstrap_iterations: int = 1000
    monte_carlo_iterations: int = 10000
    
    # Multiple comparison correction
    multiple_comparison_method: str = "fdr_bh"  # Benjamini-Hochberg
    
    # Survival analysis
    survival_confidence_level: float = 0.95
    
    # A/B testing
    ab_test_power: float = 0.8
    minimum_detectable_effect: float = 0.05

@dataclass
class BusinessConfig:
    """Configuration for business metrics and calculations."""
    # Cost structure (in USD)
    customer_acquisition_cost: float = 100.0
    customer_retention_cost: float = 25.0
    average_customer_value: float = 500.0
    churn_cost_multiplier: float = 5.0
    
    # ROI calculation parameters
    discount_rate: float = 0.1
    time_horizon_months: int = 24
    
    # Thresholds
    high_value_customer_threshold: float = 1000.0
    churn_probability_threshold: float = 0.5

@dataclass
class LoggingConfig:
    """Configuration for logging system."""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file_path: Path = field(default_factory=lambda: Path("logs"))
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    def __post_init__(self):
        """Ensure log directory exists."""
        self.log_file_path.mkdir(exist_ok=True)

@dataclass
class FeatureEngineeringConfig:
    """Configuration for advanced feature engineering."""
    rolling_window_months: List[int] = field(default_factory=lambda: [3, 6, 12])
    tenure_bins: List[int] = field(default_factory=lambda: [0, 6, 12, 24, 36, 72])
    interaction_features: List[str] = field(default_factory=lambda: ["polynomial", "ratio", "difference"])
    encoding_strategy: str = "target_encoding"
    handle_missing: str = "advanced_imputation"
    
    # Usage ratio feature settings
    usage_ratio_features: List[str] = field(default_factory=lambda: [
        "usage_efficiency_ratio", "value_per_service_ratio", "cost_per_gb_ratio"
    ])
    
    # Default values for missing usage data
    default_usage_minutes: float = 0.0
    default_data_usage_gb: float = 0.0
    min_charges_threshold: float = 0.01  # Avoid division by zero

@dataclass
class NotebookConfig:
    """Master configuration class combining all configuration components."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    business: BusinessConfig = field(default_factory=BusinessConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    
    # Notebook-specific settings
    notebook_execution_timeout: int = 3600  # 1 hour
    memory_limit_gb: int = 16
    parallel_jobs: int = -1  # Use all available cores
    
    @classmethod
    def from_environment(cls) -> 'NotebookConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if present
        if seed := os.getenv("RANDOM_SEED"):
            config.data.random_seed = int(seed)
        
        if gpu := os.getenv("USE_GPU"):
            config.model.device = "cuda" if gpu.lower() == "true" else "cpu"
        
        if log_level := os.getenv("LOG_LEVEL"):
            config.logging.log_level = log_level.upper()
        
        if timeout := os.getenv("NOTEBOOK_TIMEOUT"):
            config.notebook_execution_timeout = int(timeout)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "visualization": self.visualization.__dict__,
            "statistical": self.statistical.__dict__,
            "business": self.business.__dict__,
            "logging": self.logging.__dict__,
            "notebook_execution_timeout": self.notebook_execution_timeout,
            "memory_limit_gb": self.memory_limit_gb,
            "parallel_jobs": self.parallel_jobs
        }

# Global configuration instance
CONFIG = NotebookConfig.from_environment()

# Convenience functions for accessing configuration
def get_data_config() -> DataConfig:
    """Get data configuration."""
    return CONFIG.data

def get_model_config() -> ModelConfig:
    """Get model configuration."""
    return CONFIG.model

def get_visualization_config() -> VisualizationConfig:
    """Get visualization configuration."""
    return CONFIG.visualization

def get_statistical_config() -> StatisticalConfig:
    """Get statistical configuration."""
    return CONFIG.statistical

def get_business_config() -> BusinessConfig:
    """Get business configuration."""
    return CONFIG.business

def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return CONFIG.logging

def get_feature_engineering_config() -> FeatureEngineeringConfig:
    """Get feature engineering configuration."""
    return CONFIG.feature_engineering