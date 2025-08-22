"""
Notebook initialization utilities for consistent setup across all notebooks.
Provides standardized initialization for logging, configuration, and environment setup.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
from typing import Optional, Tuple
import sys
from pathlib import Path

from .config import CONFIG, get_visualization_config, get_data_config
from .logging_setup import setup_notebook_logging, NotebookLogger

def initialize_notebook(notebook_name: str, 
                       suppress_warnings: bool = True,
                       set_random_seeds: bool = True,
                       configure_display: bool = True) -> NotebookLogger:
    """
    Initialize notebook with standard configuration.
    
    Args:
        notebook_name: Name of the notebook (e.g., '01_exploratory_data_analysis')
        suppress_warnings: Whether to suppress common warnings
        set_random_seeds: Whether to set random seeds for reproducibility
        configure_display: Whether to configure display settings
        
    Returns:
        Configured NotebookLogger instance
    """
    # Setup logging first
    logger = setup_notebook_logging(notebook_name)
    logger.info(f"Initializing notebook: {notebook_name}")
    
    # Suppress warnings if requested
    if suppress_warnings:
        warnings.filterwarnings('ignore')
        logger.debug("Warnings suppressed")
    
    # Set random seeds for reproducibility
    if set_random_seeds:
        seed = CONFIG.data.random_seed
        np.random.seed(seed)
        
        # Set pandas random seed
        pd.core.common.random_state(seed)
        
        # Set matplotlib random seed
        plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 
            get_visualization_config().plotly_color_sequence)
        
        logger.info(f"Random seeds set to {seed} for reproducibility")
    
    # Configure display settings
    if configure_display:
        _configure_display_settings(logger)
    
    # Log system information
    _log_system_info(logger)
    
    logger.info("Notebook initialization completed successfully")
    return logger

def _configure_display_settings(logger: NotebookLogger) -> None:
    """Configure display settings for notebooks."""
    viz_config = get_visualization_config()
    
    # Configure pandas display
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    
    # Configure matplotlib
    plt.style.use(viz_config.matplotlib_style)
    plt.rcParams['figure.figsize'] = viz_config.figure_size
    plt.rcParams['figure.dpi'] = viz_config.figure_dpi
    plt.rcParams['savefig.dpi'] = viz_config.figure_dpi
    plt.rcParams['savefig.format'] = viz_config.figure_format
    
    # Configure seaborn
    sns.set_palette(viz_config.seaborn_palette)
    
    # Configure plotly
    pio.templates.default = viz_config.plotly_theme
    
    logger.debug("Display settings configured")

def _log_system_info(logger: NotebookLogger) -> None:
    """Log system information for debugging and reproducibility."""
    import platform
    import psutil
    
    system_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "cpu_count": psutil.cpu_count(),
    }
    
    logger.log_data_operation(
        operation="system_info_collection",
        status="completed",
        metrics=system_info
    )

def setup_notebook_paths() -> Tuple[Path, Path, Path, Path]:
    """
    Setup and return standard notebook paths.
    
    Returns:
        Tuple of (data_path, models_path, figures_path, logs_path)
    """
    data_path = CONFIG.data.data_path
    models_path = CONFIG.model.model_output_path
    figures_path = CONFIG.visualization.figure_output_path
    logs_path = CONFIG.logging.log_file_path
    
    # Ensure all paths exist
    for path in [data_path, models_path, figures_path, logs_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    return data_path, models_path, figures_path, logs_path

def get_notebook_config_summary() -> dict:
    """
    Get a summary of current notebook configuration.
    
    Returns:
        Dictionary with configuration summary
    """
    return {
        "random_seed": CONFIG.data.random_seed,
        "test_size": CONFIG.data.test_size,
        "cv_folds": CONFIG.model.cross_validation_folds,
        "hyperparameter_trials": CONFIG.model.hyperparameter_trials,
        "confidence_level": CONFIG.statistical.confidence_level,
        "significance_level": CONFIG.statistical.significance_level,
        "device": CONFIG.model.device,
        "parallel_jobs": CONFIG.parallel_jobs
    }

def validate_environment() -> dict:
    """
    Validate that the environment is properly configured.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "python_version_ok": sys.version_info >= (3, 8),
        "required_paths_exist": True,
        "memory_sufficient": True,
        "dependencies_available": True
    }
    
    # Check required paths
    required_paths = [
        CONFIG.data.data_path,
        CONFIG.model.model_output_path,
        CONFIG.visualization.figure_output_path,
        CONFIG.logging.log_file_path
    ]
    
    for path in required_paths:
        if not path.exists():
            validation_results["required_paths_exist"] = False
            break
    
    # Check memory
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        validation_results["memory_sufficient"] = available_memory_gb >= 4.0
        validation_results["available_memory_gb"] = round(available_memory_gb, 2)
    except ImportError:
        validation_results["memory_sufficient"] = None
    
    # Check key dependencies
    try:
        import pandas, numpy, matplotlib, seaborn, sklearn, plotly
        validation_results["dependencies_available"] = True
    except ImportError as e:
        validation_results["dependencies_available"] = False
        validation_results["missing_dependency"] = str(e)
    
    return validation_results

# Convenience function for quick notebook setup
def quick_setup(notebook_name: str) -> Tuple[NotebookLogger, dict]:
    """
    Quick setup function that initializes notebook and validates environment.
    
    Args:
        notebook_name: Name of the notebook
        
    Returns:
        Tuple of (logger, validation_results)
    """
    logger = initialize_notebook(notebook_name)
    validation_results = validate_environment()
    
    if not all(validation_results.values()):
        logger.warning("Environment validation failed", validation_results=validation_results)
    else:
        logger.info("Environment validation passed")
    
    return logger, validation_results