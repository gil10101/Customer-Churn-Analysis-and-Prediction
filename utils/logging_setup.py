"""
Centralized logging framework for all notebooks.
Provides structured logging with consistent formatting and output management.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import structlog
from dataclasses import dataclass

from .config import get_logging_config, LoggingConfig

@dataclass
class LogEntry:
    """Structured log entry for data science operations."""
    timestamp: datetime
    notebook: str
    operation: str
    status: str
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class NotebookLogger:
    """Enhanced logger for notebook operations with structured logging."""
    
    def __init__(self, notebook_name: str, log_level: Optional[str] = None):
        """
        Initialize notebook logger.
        
        Args:
            notebook_name: Name of the notebook (e.g., '01_exploratory_data_analysis')
            log_level: Optional log level override
        """
        self.notebook_name = notebook_name
        self.config = get_logging_config()
        self.log_level = log_level or self.config.log_level
        
        # Setup structured logger
        self.logger = self._setup_structured_logger()
        
        # Setup standard logger for compatibility
        self.std_logger = self._setup_standard_logger()
    
    def _setup_structured_logger(self) -> structlog.BoundLogger:
        """Setup structured logger with consistent configuration."""
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        return structlog.get_logger(self.notebook_name)
    
    def _setup_standard_logger(self) -> logging.Logger:
        """Setup standard Python logger for compatibility."""
        logger = logging.getLogger(f"notebook.{self.notebook_name}")
        logger.setLevel(getattr(logging, self.log_level))
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, self.log_level))
        console_formatter = logging.Formatter(self.config.log_format)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with rotation
        log_file = self.config.log_file_path / f"{self.notebook_name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.config.max_log_file_size,
            backupCount=self.config.backup_count
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self.std_logger.info(message)
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self.std_logger.debug(message)
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self.std_logger.warning(message)
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self.std_logger.error(message)
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self.std_logger.critical(message)
        self.logger.critical(message, **kwargs)
    
    def log_data_operation(self, operation: str, status: str, 
                          metrics: Optional[Dict[str, Any]] = None,
                          error: Optional[str] = None) -> None:
        """
        Log data science operation with structured information.
        
        Args:
            operation: Name of the operation (e.g., 'data_loading', 'model_training')
            status: Status of operation ('started', 'completed', 'failed')
            metrics: Optional metrics dictionary
            error: Optional error message if status is 'failed'
        """
        log_entry = LogEntry(
            timestamp=datetime.now(),
            notebook=self.notebook_name,
            operation=operation,
            status=status,
            metrics=metrics,
            error=error
        )
        
        log_data = {
            "operation": operation,
            "status": status,
            "notebook": self.notebook_name
        }
        
        if metrics:
            log_data["metrics"] = metrics
        
        if error:
            log_data["error"] = error
        
        if status == "failed":
            self.logger.error("Operation failed", **log_data)
        elif status == "completed":
            self.logger.info("Operation completed", **log_data)
        else:
            self.logger.info("Operation status", **log_data)
    
    def log_model_performance(self, model_name: str, metrics: Dict[str, float]) -> None:
        """Log model performance metrics."""
        self.log_data_operation(
            operation=f"model_evaluation_{model_name}",
            status="completed",
            metrics=metrics
        )
        
        # Also log to standard logger for visibility
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Model {model_name} performance: {metrics_str}")
    
    def log_data_quality(self, quality_metrics: Dict[str, float]) -> None:
        """Log data quality assessment results."""
        self.log_data_operation(
            operation="data_quality_assessment",
            status="completed",
            metrics=quality_metrics
        )
        
        # Log summary to standard logger
        avg_quality = sum(quality_metrics.values()) / len(quality_metrics)
        self.info(f"Data quality assessment completed. Average quality score: {avg_quality:.3f}")
    
    def log_business_impact(self, impact_metrics: Dict[str, float]) -> None:
        """Log business impact calculations."""
        self.log_data_operation(
            operation="business_impact_calculation",
            status="completed",
            metrics=impact_metrics
        )
        
        # Log key business metrics
        if "roi_percentage" in impact_metrics:
            self.info(f"Calculated ROI: {impact_metrics['roi_percentage']:.2f}%")
        if "revenue_impact" in impact_metrics:
            self.info(f"Revenue impact: ${impact_metrics['revenue_impact']:,.2f}")

class NotebookLoggerFactory:
    """Factory for creating notebook loggers with consistent configuration."""
    
    _loggers: Dict[str, NotebookLogger] = {}
    
    @classmethod
    def get_logger(cls, notebook_name: str, log_level: Optional[str] = None) -> NotebookLogger:
        """
        Get or create a logger for the specified notebook.
        
        Args:
            notebook_name: Name of the notebook
            log_level: Optional log level override
            
        Returns:
            NotebookLogger instance
        """
        if notebook_name not in cls._loggers:
            cls._loggers[notebook_name] = NotebookLogger(notebook_name, log_level)
        
        return cls._loggers[notebook_name]
    
    @classmethod
    def setup_notebook_logging(cls, notebook_name: str) -> NotebookLogger:
        """
        Setup logging for a notebook and return the logger.
        Convenience method for notebook initialization.
        
        Args:
            notebook_name: Name of the notebook
            
        Returns:
            Configured NotebookLogger instance
        """
        logger = cls.get_logger(notebook_name)
        logger.info(f"Logging initialized for notebook: {notebook_name}")
        return logger

# Convenience functions for easy import
def get_notebook_logger(notebook_name: str, log_level: Optional[str] = None) -> NotebookLogger:
    """Get a logger for the specified notebook."""
    return NotebookLoggerFactory.get_logger(notebook_name, log_level)

def setup_notebook_logging(notebook_name: str) -> NotebookLogger:
    """Setup logging for a notebook."""
    return NotebookLoggerFactory.setup_notebook_logging(notebook_name)

# Context manager for operation logging
class LoggedOperation:
    """Context manager for logging operations with automatic status tracking."""
    
    def __init__(self, logger: NotebookLogger, operation_name: str):
        """
        Initialize logged operation.
        
        Args:
            logger: NotebookLogger instance
            operation_name: Name of the operation to log
        """
        self.logger = logger
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        """Start the operation."""
        self.start_time = datetime.now()
        self.logger.log_data_operation(self.operation_name, "started")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the operation."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.log_data_operation(
                self.operation_name, 
                "completed",
                metrics={"duration_seconds": duration}
            )
        else:
            self.logger.log_data_operation(
                self.operation_name,
                "failed",
                metrics={"duration_seconds": duration},
                error=str(exc_val)
            )
        
        return False  # Don't suppress exceptions