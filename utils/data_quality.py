"""
Data Quality Assessment Module for Customer Churn Analysis.

This module provides comprehensive data quality assessment capabilities including
completeness, consistency, validity, and uniqueness scoring with automated
missing value analysis and outlier detection methods.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """
    Comprehensive data quality assessment metrics dataclass.
    
    Provides structured assessment of data quality across multiple dimensions
    including completeness, consistency, validity, and uniqueness.
    """
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    validity_score: float = 0.0
    uniqueness_score: float = 0.0
    overall_score: float = 0.0
    
    # Detailed breakdowns
    missing_patterns: Dict[str, float] = field(default_factory=dict)
    outlier_counts: Dict[str, int] = field(default_factory=dict)
    data_types_validation: Dict[str, bool] = field(default_factory=dict)
    consistency_violations: Dict[str, List[str]] = field(default_factory=dict)
    uniqueness_violations: Dict[str, float] = field(default_factory=dict)
    
    # Statistical summaries
    column_statistics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    correlation_issues: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate overall score after initialization."""
        self._calculate_overall_score()
    
    def _calculate_overall_score(self) -> None:
        """Calculate weighted overall data quality score."""
        weights = {
            'completeness': 0.3,
            'consistency': 0.25,
            'validity': 0.25,
            'uniqueness': 0.2
        }
        
        self.overall_score = (
            weights['completeness'] * self.completeness_score +
            weights['consistency'] * self.consistency_score +
            weights['validity'] * self.validity_score +
            weights['uniqueness'] * self.uniqueness_score
        )
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get comprehensive quality assessment summary."""
        return {
            'overall_score': self.overall_score,
            'dimension_scores': {
                'completeness': self.completeness_score,
                'consistency': self.consistency_score,
                'validity': self.validity_score,
                'uniqueness': self.uniqueness_score
            },
            'total_columns_assessed': len(self.column_statistics),
            'columns_with_missing_data': len([col for col, pct in self.missing_patterns.items() if pct > 0]),
            'columns_with_outliers': len([col for col, count in self.outlier_counts.items() if count > 0]),
            'data_type_compliance': sum(self.data_types_validation.values()) / len(self.data_types_validation) if self.data_types_validation else 0
        }


class DataQualityAssessor:
    """
    Comprehensive data quality assessment engine.
    
    Provides automated analysis of data quality across multiple dimensions
    with configurable thresholds and detailed reporting capabilities.
    """
    
    def __init__(
        self,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 1.5,
        missing_threshold: float = 0.05,
        uniqueness_threshold: float = 0.95,
        correlation_threshold: float = 0.95
    ):
        """
        Initialize data quality assessor.
        
        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'isolation_forest')
            outlier_threshold: Threshold for outlier detection
            missing_threshold: Threshold for acceptable missing data percentage
            uniqueness_threshold: Threshold for uniqueness violations
            correlation_threshold: Threshold for high correlation detection
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.missing_threshold = missing_threshold
        self.uniqueness_threshold = uniqueness_threshold
        self.correlation_threshold = correlation_threshold
        
        logger.info(f"Initialized DataQualityAssessor with {outlier_method} outlier detection")
    
    def assess_data_quality(self, df: pd.DataFrame, expected_schema: Optional[Dict[str, str]] = None) -> DataQualityMetrics:
        """
        Perform comprehensive data quality assessment.
        
        Args:
            df: DataFrame to assess
            expected_schema: Optional dictionary mapping column names to expected data types
            
        Returns:
            DataQualityMetrics object with comprehensive assessment results
        """
        logger.info(f"Starting data quality assessment for DataFrame with shape {df.shape}")
        
        # Initialize metrics
        metrics = DataQualityMetrics()
        
        # Assess each dimension
        metrics.completeness_score = self._assess_completeness(df, metrics)
        metrics.consistency_score = self._assess_consistency(df, metrics, expected_schema)
        metrics.validity_score = self._assess_validity(df, metrics)
        metrics.uniqueness_score = self._assess_uniqueness(df, metrics)
        
        # Calculate column statistics
        metrics.column_statistics = self._calculate_column_statistics(df)
        
        # Detect correlation issues
        metrics.correlation_issues = self._detect_correlation_issues(df)
        
        # Recalculate overall score
        metrics._calculate_overall_score()
        
        logger.info(f"Data quality assessment completed. Overall score: {metrics.overall_score:.3f}")
        
        return metrics
    
    def _assess_completeness(self, df: pd.DataFrame, metrics: DataQualityMetrics) -> float:
        """Assess data completeness dimension."""
        missing_percentages = df.isnull().sum() / len(df)
        metrics.missing_patterns = missing_percentages.to_dict()
        
        # Calculate completeness score (1 - average missing percentage)
        completeness_score = 1 - missing_percentages.mean()
        
        logger.debug(f"Completeness assessment: {completeness_score:.3f}")
        return max(0.0, completeness_score)
    
    def _assess_consistency(self, df: pd.DataFrame, metrics: DataQualityMetrics, expected_schema: Optional[Dict[str, str]]) -> float:
        """Assess data consistency dimension."""
        consistency_violations = {}
        data_type_validation = {}
        
        for column in df.columns:
            violations = []
            
            # Check data type consistency
            if expected_schema and column in expected_schema:
                expected_type = expected_schema[column]
                actual_type = str(df[column].dtype)
                is_valid_type = self._validate_data_type(df[column], expected_type)
                data_type_validation[column] = is_valid_type
                
                if not is_valid_type:
                    violations.append(f"Expected {expected_type}, got {actual_type}")
            else:
                data_type_validation[column] = True
            
            # Check for mixed data types in object columns
            if df[column].dtype == 'object':
                unique_types = set(type(val).__name__ for val in df[column].dropna().unique())
                if len(unique_types) > 1:
                    violations.append(f"Mixed data types: {unique_types}")
            
            # Check for inconsistent formatting (e.g., date formats, case sensitivity)
            if df[column].dtype == 'object':
                inconsistencies = self._detect_format_inconsistencies(df[column])
                violations.extend(inconsistencies)
            
            if violations:
                consistency_violations[column] = violations
        
        metrics.consistency_violations = consistency_violations
        metrics.data_types_validation = data_type_validation
        
        # Calculate consistency score
        total_columns = len(df.columns)
        consistent_columns = sum(data_type_validation.values())
        format_penalty = len(consistency_violations) / total_columns
        
        consistency_score = (consistent_columns / total_columns) - format_penalty
        
        logger.debug(f"Consistency assessment: {consistency_score:.3f}")
        return max(0.0, consistency_score)
    
    def _assess_validity(self, df: pd.DataFrame, metrics: DataQualityMetrics) -> float:
        """Assess data validity dimension."""
        # Detect outliers for numerical columns
        outlier_counts = {}
        
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numerical_columns:
            outliers = self._detect_outliers(df[column].dropna())
            outlier_counts[column] = len(outliers)
        
        metrics.outlier_counts = outlier_counts
        
        # Calculate validity score based on outlier percentage
        total_numerical_values = df[numerical_columns].count().sum()
        total_outliers = sum(outlier_counts.values())
        
        if total_numerical_values > 0:
            outlier_percentage = total_outliers / total_numerical_values
            validity_score = 1 - outlier_percentage
        else:
            validity_score = 1.0
        
        logger.debug(f"Validity assessment: {validity_score:.3f}")
        return max(0.0, validity_score)
    
    def _assess_uniqueness(self, df: pd.DataFrame, metrics: DataQualityMetrics) -> float:
        """Assess data uniqueness dimension."""
        uniqueness_violations = {}
        
        for column in df.columns:
            unique_ratio = df[column].nunique() / len(df[column].dropna())
            
            # Flag columns that should be unique but aren't
            if column.lower() in ['id', 'customer_id', 'user_id', 'email']:
                if unique_ratio < self.uniqueness_threshold:
                    uniqueness_violations[column] = unique_ratio
            
            # Store all uniqueness ratios for analysis
            uniqueness_violations[f"{column}_ratio"] = unique_ratio
        
        metrics.uniqueness_violations = uniqueness_violations
        
        # Calculate uniqueness score
        # Focus on columns that should be unique
        id_columns = [col for col in df.columns if any(id_term in col.lower() for id_term in ['id', 'email'])]
        
        if id_columns:
            id_uniqueness_scores = [
                df[col].nunique() / len(df[col].dropna()) 
                for col in id_columns
            ]
            uniqueness_score = np.mean(id_uniqueness_scores)
        else:
            # If no ID columns, use average uniqueness across all columns
            uniqueness_score = np.mean([
                df[col].nunique() / len(df[col].dropna()) 
                for col in df.columns
            ])
        
        logger.debug(f"Uniqueness assessment: {uniqueness_score:.3f}")
        return max(0.0, uniqueness_score)
    
    def _detect_outliers(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using specified method."""
        if self.outlier_method == 'iqr':
            return self._detect_outliers_iqr(series)
        elif self.outlier_method == 'zscore':
            return self._detect_outliers_zscore(series)
        elif self.outlier_method == 'isolation_forest':
            return self._detect_outliers_isolation_forest(series)
        else:
            raise ValueError(f"Unknown outlier detection method: {self.outlier_method}")
    
    def _detect_outliers_iqr(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using Interquartile Range method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - self.outlier_threshold * IQR
        upper_bound = Q3 + self.outlier_threshold * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return outliers.index.values
    
    def _detect_outliers_zscore(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series))
        outliers = series[z_scores > self.outlier_threshold]
        return outliers.index.values
    
    def _detect_outliers_isolation_forest(self, series: pd.Series) -> np.ndarray:
        """Detect outliers using Isolation Forest method."""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Reshape for sklearn
            X = series.values.reshape(-1, 1)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(X)
            
            # Return indices of outliers (labeled as -1)
            outlier_indices = series.index[outlier_labels == -1]
            return outlier_indices.values
            
        except ImportError:
            logger.warning("sklearn not available, falling back to IQR method")
            return self._detect_outliers_iqr(series)
    
    def _validate_data_type(self, series: pd.Series, expected_type: str) -> bool:
        """Validate if series matches expected data type."""
        actual_type = str(series.dtype)
        
        # Define type mappings
        type_mappings = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32', 'float16'],
            'string': ['object', 'string'],
            'datetime': ['datetime64[ns]', 'datetime64'],
            'bool': ['bool'],
            'category': ['category']
        }
        
        if expected_type in type_mappings:
            return actual_type in type_mappings[expected_type]
        else:
            return actual_type == expected_type
    
    def _detect_format_inconsistencies(self, series: pd.Series) -> List[str]:
        """Detect formatting inconsistencies in object columns."""
        inconsistencies = []
        
        if series.dtype != 'object':
            return inconsistencies
        
        # Check for mixed case patterns
        string_values = series.dropna().astype(str)
        
        if len(string_values) > 0:
            # Check case consistency
            has_upper = any(val.isupper() for val in string_values if val.isalpha())
            has_lower = any(val.islower() for val in string_values if val.isalpha())
            has_mixed = any(val != val.upper() and val != val.lower() for val in string_values if val.isalpha())
            
            if sum([has_upper, has_lower, has_mixed]) > 1:
                inconsistencies.append("Inconsistent case formatting")
            
            # Check for leading/trailing whitespace
            has_whitespace = any(val != val.strip() for val in string_values)
            if has_whitespace:
                inconsistencies.append("Leading/trailing whitespace detected")
            
            # Check for multiple date formats (basic check)
            if any(char in ''.join(string_values.head(100)) for char in ['/', '-', '.']):
                date_patterns = set()
                for val in string_values.head(100):
                    if any(char in val for char in ['/', '-', '.']):
                        # Simple pattern detection
                        if '/' in val:
                            date_patterns.add('slash_separated')
                        if '-' in val:
                            date_patterns.add('dash_separated')
                        if '.' in val:
                            date_patterns.add('dot_separated')
                
                if len(date_patterns) > 1:
                    inconsistencies.append("Multiple date/time formats detected")
        
        return inconsistencies
    
    def _calculate_column_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Calculate comprehensive statistics for each column."""
        column_stats = {}
        
        for column in df.columns:
            stats_dict = {
                'dtype': str(df[column].dtype),
                'non_null_count': df[column].count(),
                'null_count': df[column].isnull().sum(),
                'null_percentage': df[column].isnull().sum() / len(df),
                'unique_count': df[column].nunique(),
                'unique_percentage': df[column].nunique() / len(df[column].dropna()) if len(df[column].dropna()) > 0 else 0
            }
            
            # Add numerical statistics for numeric columns
            if df[column].dtype in ['int64', 'int32', 'float64', 'float32']:
                numeric_stats = df[column].describe()
                stats_dict.update({
                    'mean': numeric_stats['mean'],
                    'std': numeric_stats['std'],
                    'min': numeric_stats['min'],
                    'max': numeric_stats['max'],
                    'q25': numeric_stats['25%'],
                    'q50': numeric_stats['50%'],
                    'q75': numeric_stats['75%'],
                    'skewness': df[column].skew(),
                    'kurtosis': df[column].kurtosis()
                })
            
            # Add categorical statistics for object columns
            elif df[column].dtype == 'object':
                value_counts = df[column].value_counts()
                stats_dict.update({
                    'most_frequent_value': value_counts.index[0] if len(value_counts) > 0 else None,
                    'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                    'least_frequent_value': value_counts.index[-1] if len(value_counts) > 0 else None,
                    'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0
                })
            
            column_stats[column] = stats_dict
        
        return column_stats
    
    def _detect_correlation_issues(self, df: pd.DataFrame) -> Dict[str, float]:
        """Detect high correlation issues between numerical columns."""
        correlation_issues = {}
        
        # Get numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        if len(numerical_df.columns) > 1:
            # Calculate correlation matrix
            corr_matrix = numerical_df.corr().abs()
            
            # Find high correlations (excluding diagonal)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if corr_value > self.correlation_threshold:
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        high_corr_pairs.append((col1, col2, corr_value))
                        correlation_issues[f"{col1}_vs_{col2}"] = corr_value
        
        return correlation_issues
    
    def generate_quality_report(self, metrics: DataQualityMetrics, output_path: Optional[Path] = None) -> str:
        """Generate comprehensive data quality report."""
        report_lines = [
            "=" * 80,
            "DATA QUALITY ASSESSMENT REPORT",
            "=" * 80,
            "",
            f"Overall Quality Score: {metrics.overall_score:.3f}",
            "",
            "DIMENSION SCORES:",
            f"  Completeness: {metrics.completeness_score:.3f}",
            f"  Consistency:  {metrics.consistency_score:.3f}",
            f"  Validity:     {metrics.validity_score:.3f}",
            f"  Uniqueness:   {metrics.uniqueness_score:.3f}",
            "",
            "DETAILED FINDINGS:",
            ""
        ]
        
        # Missing data analysis
        if metrics.missing_patterns:
            report_lines.extend([
                "Missing Data Analysis:",
                "-" * 25
            ])
            for col, pct in sorted(metrics.missing_patterns.items(), key=lambda x: x[1], reverse=True):
                if pct > 0:
                    report_lines.append(f"  {col}: {pct:.1%} missing")
            report_lines.append("")
        
        # Outlier analysis
        if metrics.outlier_counts:
            report_lines.extend([
                "Outlier Analysis:",
                "-" * 17
            ])
            for col, count in sorted(metrics.outlier_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0:
                    report_lines.append(f"  {col}: {count} outliers detected")
            report_lines.append("")
        
        # Consistency violations
        if metrics.consistency_violations:
            report_lines.extend([
                "Consistency Issues:",
                "-" * 19
            ])
            for col, violations in metrics.consistency_violations.items():
                report_lines.append(f"  {col}:")
                for violation in violations:
                    report_lines.append(f"    - {violation}")
            report_lines.append("")
        
        # Correlation issues
        if metrics.correlation_issues:
            report_lines.extend([
                "High Correlation Issues:",
                "-" * 24
            ])
            for pair, corr_value in sorted(metrics.correlation_issues.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"  {pair}: {corr_value:.3f}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Quality report saved to {output_path}")
        
        return report_text


# Convenience functions for quick assessment
def assess_dataframe_quality(
    df: pd.DataFrame,
    expected_schema: Optional[Dict[str, str]] = None,
    outlier_method: str = 'iqr',
    generate_report: bool = False,
    report_path: Optional[Path] = None
) -> DataQualityMetrics:
    """
    Convenience function for quick data quality assessment.
    
    Args:
        df: DataFrame to assess
        expected_schema: Optional schema validation
        outlier_method: Method for outlier detection
        generate_report: Whether to generate text report
        report_path: Path to save report
        
    Returns:
        DataQualityMetrics object with assessment results
    """
    assessor = DataQualityAssessor(outlier_method=outlier_method)
    metrics = assessor.assess_data_quality(df, expected_schema)
    
    if generate_report:
        report = assessor.generate_quality_report(metrics, report_path)
        if not report_path:
            print(report)
    
    return metrics


def quick_quality_check(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick quality check returning summary statistics.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary with key quality indicators
    """
    metrics = assess_dataframe_quality(df)
    return metrics.get_quality_summary()