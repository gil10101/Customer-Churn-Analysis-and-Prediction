"""
Utils package for Customer Churn Analysis and Prediction.
Contains shared utility functions used across the project.
"""

# Configuration management
from .config import (
    NotebookConfig,
    DataConfig,
    ModelConfig,
    VisualizationConfig,
    StatisticalConfig,
    BusinessConfig,
    LoggingConfig,
    CONFIG,
    get_data_config,
    get_model_config,
    get_visualization_config,
    get_statistical_config,
    get_business_config,
    get_logging_config
)

# Logging framework
from .logging_setup import (
    NotebookLogger,
    NotebookLoggerFactory,
    LoggedOperation,
    get_notebook_logger,
    setup_notebook_logging
)

# Notebook initialization
from .notebook_init import (
    initialize_notebook,
    setup_notebook_paths,
    get_notebook_config_summary,
    validate_environment,
    quick_setup
)

# Data quality assessment
from .data_quality import (
    DataQualityMetrics,
    DataQualityAssessor,
    assess_dataframe_quality,
    quick_quality_check
)

# Model evaluation
from .model_evaluation import (
    ModelPerformance,
    BusinessMetrics,
    ModelEvaluator,
    evaluate_classification_model,
    quick_model_comparison
)

# Visualization theme
from .visualization import (
    ColorPalette,
    FontConfiguration,
    FigureConfiguration,
    VisualizationTheme,
    get_professional_theme,
    get_minimal_theme,
    get_dark_theme,
    get_publication_theme,
    apply_theme,
    get_current_theme,
    save_publication_figure
)

# Class imbalance handling
from .imbalance_handler import (
    ImbalanceHandler,
    ImbalanceStrategy,
    ImbalanceResults
)

__all__ = [
    # Configuration classes
    'NotebookConfig',
    'DataConfig', 
    'ModelConfig',
    'VisualizationConfig',
    'StatisticalConfig',
    'BusinessConfig',
    'LoggingConfig',
    'CONFIG',
    
    # Configuration getters
    'get_data_config',
    'get_model_config', 
    'get_visualization_config',
    'get_statistical_config',
    'get_business_config',
    'get_logging_config',
    
    # Logging classes and functions
    'NotebookLogger',
    'NotebookLoggerFactory',
    'LoggedOperation',
    'get_notebook_logger',
    'setup_notebook_logging',
    
    # Notebook initialization
    'initialize_notebook',
    'setup_notebook_paths',
    'get_notebook_config_summary',
    'validate_environment',
    'quick_setup',
    
    # Data quality assessment
    'DataQualityMetrics',
    'DataQualityAssessor',
    'assess_dataframe_quality',
    'quick_quality_check',
    
    # Model evaluation
    'ModelPerformance',
    'BusinessMetrics',
    'ModelEvaluator',
    'evaluate_classification_model',
    'quick_model_comparison',
    
    # Visualization theme
    'ColorPalette',
    'FontConfiguration',
    'FigureConfiguration',
    'VisualizationTheme',
    'get_professional_theme',
    'get_minimal_theme',
    'get_dark_theme',
    'get_publication_theme',
    'apply_theme',
    'get_current_theme',
    'save_publication_figure',
    
    # Class imbalance handling
    'ImbalanceHandler',
    'ImbalanceStrategy',
    'ImbalanceResults'
] 