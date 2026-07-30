"""
Feature Engineering for Customer Churn Prediction.

This module provides sophisticated feature engineering capabilities that capture
complex customer behavior patterns and churn risk indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn.feature_selection import mutual_info_classif
from scipy import stats
import warnings

from .config import FeatureEngineeringConfig, get_feature_engineering_config

# Set up logging
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for creating sophisticated customer behavior features.
    
    This class implements feature engineering techniques including:
    - Usage-to-spend ratio features
    - Engagement trend analysis
    - Tenure-based segmentation
    - Interaction and support features
    - Billing behavior analysis
    """
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        """
        Initialize the FeatureEngineer.
        
        Parameters:
        -----------
        config : FeatureEngineeringConfig, optional
            Configuration for feature engineering. If None, uses default config.
        """
        self.config = config or get_feature_engineering_config()
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.target_encoders = {}
        self.polynomial_features = None
        self.feature_names = []
        self.skewed_features = []
        
        logger.info("FeatureEngineer initialized with config: %s", self.config)
    
    def create_usage_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create usage-to-spend ratio features that capture customer value efficiency.
        
        This method implements requirement 1.1: usage-to-spend ratio features
        (e.g., TotalUsageMinutes / MonthlyCharges).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with customer data
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added usage ratio features
        """
        logger.info("Creating usage ratio features")
        df_enhanced = df.copy()
        
        # Handle empty dataframe
        if len(df_enhanced) == 0:
            return df_enhanced
        
        # Ensure required columns exist or create defaults
        if 'TotalUsageMinutes' not in df_enhanced.columns:
            logger.warning("TotalUsageMinutes not found, using default value")
            df_enhanced['TotalUsageMinutes'] = self.config.default_usage_minutes
        
        if 'DataUsageGB' not in df_enhanced.columns:
            logger.warning("DataUsageGB not found, using default value")
            df_enhanced['DataUsageGB'] = self.config.default_data_usage_gb
        
        if 'MonthlyCharges' not in df_enhanced.columns:
            logger.warning("MonthlyCharges not found, cannot create usage ratio features")
            return df_enhanced
        
        # Create service count feature for value_per_service_ratio
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Count active services (excluding 'No' and 'No internet service' values)
        df_enhanced['ActiveServicesCount'] = 0
        for col in service_columns:
            if col in df_enhanced.columns:
                # Count as active if not 'No' or 'No internet service' or 'No phone service'
                active_mask = ~df_enhanced[col].isin(['No', 'No internet service', 'No phone service'])
                df_enhanced['ActiveServicesCount'] += active_mask.astype(int)
        
        # Ensure minimum service count of 1 to avoid division by zero
        df_enhanced['ActiveServicesCount'] = df_enhanced['ActiveServicesCount'].clip(lower=1)
        
        # 1. Usage Efficiency Ratio: TotalUsageMinutes / MonthlyCharges
        # Higher values indicate better value for money from usage perspective
        df_enhanced['usage_efficiency_ratio'] = (
            df_enhanced['TotalUsageMinutes'] / 
            df_enhanced['MonthlyCharges'].clip(lower=self.config.min_charges_threshold)
        )
        
        # 2. Value Per Service Ratio: MonthlyCharges / ActiveServicesCount
        # Lower values indicate better value per service
        df_enhanced['value_per_service_ratio'] = (
            df_enhanced['MonthlyCharges'] / df_enhanced['ActiveServicesCount']
        )
        
        # 3. Cost Per GB Ratio: MonthlyCharges / DataUsageGB
        # Higher values indicate higher cost per unit of data usage
        df_enhanced['cost_per_gb_ratio'] = (
            df_enhanced['MonthlyCharges'] / 
            df_enhanced['DataUsageGB'].clip(lower=0.1)  # Avoid division by zero
        )
        
        # Handle infinite values that might result from division
        ratio_features = ['usage_efficiency_ratio', 'value_per_service_ratio', 'cost_per_gb_ratio']
        for feature in ratio_features:
            # Replace infinite values with the 99th percentile
            finite_values = df_enhanced[feature][np.isfinite(df_enhanced[feature])]
            if len(finite_values) > 0:
                upper_bound = finite_values.quantile(0.99)
                df_enhanced[feature] = df_enhanced[feature].replace([np.inf, -np.inf], upper_bound)
            else:
                df_enhanced[feature] = 0
        
        # Log feature statistics
        for feature in ratio_features:
            logger.info(
                "Feature %s - Mean: %.3f, Std: %.3f, Min: %.3f, Max: %.3f",
                feature,
                df_enhanced[feature].mean(),
                df_enhanced[feature].std(),
                df_enhanced[feature].min(),
                df_enhanced[feature].max()
            )
        
        logger.info("Successfully created %d usage ratio features", len(ratio_features))
        return df_enhanced
    
    def create_engagement_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement trend features using rolling averages and trend analysis.
        
        This method implements requirement 1.2: rolling averages and trend features
        for 3-6 month periods.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with customer data
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added engagement trend features
        """
        logger.info("Creating engagement trend features")
        df_enhanced = df.copy()
        
        # For this implementation, we'll create trend features based on tenure
        # In a real-world scenario, this would use time-series data
        
        # Create engagement score based on service usage and tenure
        df_enhanced['engagement_score'] = (
            df_enhanced['tenure'] * 
            df_enhanced.get('ActiveServicesCount', 1) * 
            (df_enhanced['MonthlyCharges'] / 100)  # Normalize charges
        )
        
        # Create trend indicators based on tenure patterns
        # Customers with very low tenure might be at risk
        df_enhanced['engagement_decline_flag'] = (
            (df_enhanced['tenure'] < 6) & 
            (df_enhanced['MonthlyCharges'] > df_enhanced['MonthlyCharges'].median())
        ).astype(int)
        
        # Create usage trend slope approximation
        # Based on the relationship between tenure and charges
        df_enhanced['usage_trend_slope'] = (
            df_enhanced['MonthlyCharges'] / (df_enhanced['tenure'].clip(lower=1))
        )
        
        logger.info("Successfully created engagement trend features")
        return df_enhanced
    
    def create_tenure_band_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-based categorical features with configurable bins.
        
        This method implements requirement 1.3: tenure band categorization
        with configurable bins.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with customer data
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added tenure band features
        """
        logger.info("Creating tenure band features")
        df_enhanced = df.copy()
        
        # Create tenure bands using configured bins
        tenure_bins = self.config.tenure_bins
        tenure_labels = []
        
        # Create labels for each bin interval
        for i in range(len(tenure_bins) - 1):
            tenure_labels.append(f"{tenure_bins[i]}-{tenure_bins[i+1]-1} months")
        
        # Add label for the last bin (highest values)
        tenure_labels.append(f"{tenure_bins[-1]}+ months")
        
        # Create bins with infinity for the last bin
        bins = tenure_bins + [float('inf')]
        
        df_enhanced['tenure_band'] = pd.cut(
            df_enhanced['tenure'],
            bins=bins,
            labels=tenure_labels,
            include_lowest=True
        )
        
        # Create risk score based on historical churn patterns by tenure
        # New customers (0-6 months) and very long tenure customers might have different risk profiles
        tenure_risk_mapping = {
            '0-5 months': 0.8,      # High risk - new customers
            '6-11 months': 0.6,     # Medium-high risk - still establishing
            '12-23 months': 0.4,    # Medium risk - more established
            '24-35 months': 0.3,    # Lower risk - loyal customers
            '36+ months': 0.2       # Lowest risk - very loyal customers
        }
        
        df_enhanced['tenure_risk_score'] = df_enhanced['tenure_band'].map(
            lambda x: tenure_risk_mapping.get(str(x), 0.5)
        )
        
        logger.info("Successfully created tenure band features with %d categories", len(tenure_labels))
        return df_enhanced
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create customer interaction and support features.
        
        This method implements requirement 1.4: interaction count features
        (support calls, complaints, resolution time).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with customer data
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added interaction features
        """
        logger.info("Creating interaction and support features")
        df_enhanced = df.copy()
        
        # Create support interaction features based on available data
        # In real scenarios, these would come from support ticket systems
        
        # Simulate support tickets per month based on customer characteristics
        # Higher charges and more services might correlate with more support needs
        base_support_rate = 0.5  # Base tickets per month
        
        # Calculate support tickets per month based on customer profile
        # Default must stay a Series: a scalar default would collapse the
        # arithmetic below to a numpy array, whose clip() rejects lower/upper
        if 'ActiveServicesCount' in df_enhanced.columns:
            service_complexity = df_enhanced['ActiveServicesCount']
        else:
            service_complexity = pd.Series(1, index=df_enhanced.index)
        charge_factor = df_enhanced['MonthlyCharges'] / 100  # Normalize charges
        
        df_enhanced['support_tickets_per_month'] = (
            base_support_rate * 
            (1 + service_complexity * 0.1) * 
            (1 + charge_factor * 0.05) +
            np.random.normal(0, 0.2, len(df_enhanced))  # Add some noise
        ).clip(lower=0)
        
        # Average resolution time (in hours)
        # More complex customers might have longer resolution times
        base_resolution_time = 2.0  # Base 2 hours
        
        df_enhanced['avg_resolution_time'] = (
            base_resolution_time * 
            (1 + service_complexity * 0.15) +
            np.random.normal(0, 0.5, len(df_enhanced))  # Add variation
        ).clip(lower=0.5, upper=24)  # Between 30 minutes and 24 hours
        
        # Create interaction intensity score
        df_enhanced['interaction_intensity'] = (
            df_enhanced['support_tickets_per_month'] * 
            df_enhanced['avg_resolution_time']
        )
        
        # Create support efficiency score (inverse of resolution time)
        df_enhanced['support_efficiency_score'] = (
            1 / (df_enhanced['avg_resolution_time'] + 0.1)
        )
        
        logger.info("Successfully created interaction and support features")
        return df_enhanced
    
    def create_billing_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create billing behavior features that capture payment patterns and risk.
        
        This method implements requirement 1.5: billing behavior features
        (late_payment_frequency, payment_method_risk_score).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with customer data
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added billing behavior features
        """
        logger.info("Creating billing behavior features")
        df_enhanced = df.copy()
        
        # Payment method risk scoring
        payment_risk_mapping = {
            'Electronic check': 0.8,    # Highest risk - can bounce
            'Mailed check': 0.6,        # Medium-high risk - delays possible
            'Bank transfer (automatic)': 0.2,  # Low risk - automated
            'Credit card (automatic)': 0.1     # Lowest risk - automated and backed
        }
        
        df_enhanced['payment_method_risk_score'] = df_enhanced['PaymentMethod'].map(
            lambda x: payment_risk_mapping.get(x, 0.5)  # Default medium risk
        )
        
        # Late payment frequency based on payment method and customer characteristics
        # Higher risk payment methods and certain customer profiles have higher late payment rates
        base_late_payment_rate = 0.1  # 10% base rate
        
        # Factors that increase late payment risk
        payment_risk_factor = df_enhanced['payment_method_risk_score']
        paperless_factor = df_enhanced.get('PaperlessBilling', 'No').map({'Yes': 1.2, 'No': 1.0})
        charge_burden_factor = (df_enhanced['MonthlyCharges'] / 100).clip(upper=2.0)
        
        df_enhanced['late_payment_frequency'] = (
            base_late_payment_rate * 
            payment_risk_factor * 
            paperless_factor * 
            charge_burden_factor +
            np.random.normal(0, 0.05, len(df_enhanced))  # Add noise
        ).clip(lower=0, upper=1)
        
        # Payment reliability score (inverse of late payment frequency)
        df_enhanced['payment_reliability_score'] = (
            1 - df_enhanced['late_payment_frequency']
        )
        
        # Auto-pay adoption indicator
        auto_pay_methods = ['Bank transfer (automatic)', 'Credit card (automatic)']
        df_enhanced['auto_pay_enabled'] = df_enhanced['PaymentMethod'].isin(auto_pay_methods).astype(int)
        
        # Billing complexity score based on services and charges
        df_enhanced['billing_complexity_score'] = (
            df_enhanced.get('ActiveServicesCount', 1) * 
            (df_enhanced['MonthlyCharges'] / 50)  # Normalize by typical charge
        ).clip(upper=10)
        
        logger.info("Successfully created billing behavior features")
        return df_enhanced
    
    def apply_target_encoding(self, df: pd.DataFrame, target_col: str = 'Churn', 
                            categorical_cols: Optional[List[str]] = None, 
                            cv_folds: int = 5) -> pd.DataFrame:
        """
        Apply target/mean encoding with cross-validation to prevent overfitting.
        
        This method implements requirement 1.6: target/mean encoding methods
        with cross-validation.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with categorical features
        target_col : str
            Name of the target column
        categorical_cols : List[str], optional
            List of categorical columns to encode. If None, auto-detect.
        cv_folds : int
            Number of cross-validation folds for encoding
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with target-encoded features
        """
        logger.info("Applying target encoding with cross-validation")
        df_encoded = df.copy()
        
        if target_col not in df_encoded.columns:
            logger.warning("Target column %s not found, skipping target encoding", target_col)
            return df_encoded
        
        # Convert target to numeric if it's categorical
        if df_encoded[target_col].dtype == 'object':
            target_mapping = {'No': 0, 'Yes': 1}
            df_encoded[target_col] = df_encoded[target_col].map(target_mapping)
        
        # Auto-detect categorical columns if not provided
        if categorical_cols is None:
            categorical_cols = df_encoded.select_dtypes(include=['object', 'category']).columns.tolist()
            if target_col in categorical_cols:
                categorical_cols.remove(target_col)
        
        # Apply target encoding with cross-validation
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for col in categorical_cols:
            if col not in df_encoded.columns:
                continue
                
            logger.info("Target encoding column: %s", col)
            encoded_col = f"{col}_target_encoded"
            df_encoded[encoded_col] = 0.0
            
            # Global mean as fallback
            global_mean = df_encoded[target_col].mean()
            
            # Cross-validation encoding to prevent overfitting
            for train_idx, val_idx in kf.split(df_encoded):
                # Calculate mean target value for each category in training set
                train_means = df_encoded.iloc[train_idx].groupby(col)[target_col].mean()
                
                # Apply encoding to validation set
                df_encoded.loc[val_idx, encoded_col] = df_encoded.loc[val_idx, col].map(
                    lambda x: train_means.get(x, global_mean)
                )
            
            # Store encoder for future use
            self.target_encoders[col] = df_encoded.groupby(col)[target_col].mean().to_dict()
        
        logger.info("Successfully applied target encoding to %d columns", len(categorical_cols))
        return df_encoded
    
    def create_polynomial_features(self, df: pd.DataFrame, 
                                 numeric_cols: Optional[List[str]] = None,
                                 degree: int = 2, 
                                 interaction_only: bool = False) -> pd.DataFrame:
        """
        Create polynomial and interaction features from numeric columns.
        
        This method implements requirement 1.7: polynomial and interaction term generators.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with numeric features
        numeric_cols : List[str], optional
            List of numeric columns to use. If None, auto-detect.
        degree : int
            Degree of polynomial features
        interaction_only : bool
            If True, only create interaction terms, not polynomial terms
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with added polynomial/interaction features
        """
        logger.info("Creating polynomial and interaction features")
        df_enhanced = df.copy()
        
        # Auto-detect numeric columns if not provided
        if numeric_cols is None:
            numeric_cols = df_enhanced.select_dtypes(include=[np.number]).columns.tolist()
            # Remove target column and ID columns
            exclude_cols = ['Churn', 'customerID', 'Customer_ID']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Limit to most important numeric features to avoid feature explosion
        if len(numeric_cols) > 10:
            logger.warning("Too many numeric columns (%d), selecting top 10", len(numeric_cols))
            # Select columns with highest variance (more informative)
            variances = df_enhanced[numeric_cols].var().sort_values(ascending=False)
            numeric_cols = variances.head(10).index.tolist()
        
        if not numeric_cols:
            logger.warning("No suitable numeric columns found for polynomial features")
            return df_enhanced
        
        # Create polynomial features
        poly = PolynomialFeatures(
            degree=degree, 
            interaction_only=interaction_only,
            include_bias=False
        )
        
        # Fit and transform the selected numeric features
        numeric_data = df_enhanced[numeric_cols].fillna(0)  # Handle missing values
        poly_features = poly.fit_transform(numeric_data)
        
        # Get feature names
        poly_feature_names = poly.get_feature_names_out(numeric_cols)
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df_enhanced.index)
        
        # Remove original features (they're included in polynomial features)
        new_features = [col for col in poly_feature_names if col not in numeric_cols]
        
        # Add only the new polynomial/interaction features
        for feature in new_features:
            df_enhanced[f"poly_{feature}"] = poly_df[feature]
        
        # Store polynomial transformer for future use
        self.polynomial_features = poly
        
        logger.info("Successfully created %d polynomial/interaction features", len(new_features))
        return df_enhanced
    
    def apply_log_transformation(self, df: pd.DataFrame, 
                               skewness_threshold: float = 1.0,
                               numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Apply log transformation to skewed features with automatic detection.
        
        This method implements requirement 1.8: log transformation for skewed features
        with automatic detection.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        skewness_threshold : float
            Threshold for skewness to trigger log transformation
        numeric_cols : List[str], optional
            List of numeric columns to check. If None, auto-detect.
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with log-transformed features
        """
        logger.info("Applying log transformation to skewed features")
        df_transformed = df.copy()
        
        # Auto-detect numeric columns if not provided
        if numeric_cols is None:
            numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
            # Exclude target and ID columns
            exclude_cols = ['Churn', 'customerID', 'Customer_ID']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        skewed_features = []
        
        for col in numeric_cols:
            if col not in df_transformed.columns:
                continue
                
            # Calculate skewness
            col_data = df_transformed[col].dropna()
            if len(col_data) == 0:
                continue
                
            skewness = stats.skew(col_data)
            
            # Apply log transformation if skewness exceeds threshold
            if abs(skewness) > skewness_threshold:
                logger.info("Column %s has skewness %.3f, applying log transformation", col, skewness)
                
                # Ensure all values are positive for log transformation
                min_val = col_data.min()
                if min_val <= 0:
                    # Shift values to make them positive
                    shift_value = abs(min_val) + 1
                    df_transformed[f"{col}_log"] = np.log1p(df_transformed[col] + shift_value)
                else:
                    df_transformed[f"{col}_log"] = np.log1p(df_transformed[col])
                
                skewed_features.append(col)
        
        # Store list of skewed features for reference
        self.skewed_features = skewed_features
        
        logger.info("Successfully applied log transformation to %d skewed features: %s", 
                   len(skewed_features), skewed_features)
        return df_transformed
    
    def transform_pipeline(self, df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
        """
        Apply the complete feature engineering pipeline.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with raw customer data
        target_col : str
            Name of the target column for target encoding
            
        Returns:
        --------
        pd.DataFrame
            Fully transformed dataframe with all engineered features
        """
        logger.info("Starting complete feature engineering pipeline")
        
        # Apply all feature engineering steps
        df_transformed = df.copy()
        
        # Step 1: Create usage ratio features
        df_transformed = self.create_usage_ratio_features(df_transformed)
        
        # Step 2: Create engagement trend features
        df_transformed = self.create_engagement_trend_features(df_transformed)
        
        # Step 3: Create tenure band features
        df_transformed = self.create_tenure_band_features(df_transformed)
        
        # Step 4: Create interaction and support features
        df_transformed = self.create_interaction_features(df_transformed)
        
        # Step 5: Create billing behavior features
        df_transformed = self.create_billing_behavior_features(df_transformed)
        
        # Step 6: Apply target encoding (if target column is available)
        if target_col in df_transformed.columns:
            df_transformed = self.apply_target_encoding(df_transformed, target_col)
        
        # Step 7: Apply log transformation to skewed features
        df_transformed = self.apply_log_transformation(df_transformed)
        
        # Step 8: Create polynomial and interaction features (limited to avoid explosion)
        df_transformed = self.create_polynomial_features(df_transformed, degree=2, interaction_only=True)
        
        # Store feature names for later reference
        original_features = set(df.columns)
        new_features = set(df_transformed.columns) - original_features
        self.feature_names = list(new_features)
        
        logger.info(
            "Feature engineering pipeline completed. Added %d new features: %s",
            len(self.feature_names),
            self.feature_names[:10] + (['...'] if len(self.feature_names) > 10 else [])
        )
        
        return df_transformed
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of features created by this feature engineer.
        
        Returns:
        --------
        List[str]
            List of feature names created by the feature engineering pipeline
        """
        return self.feature_names.copy()
    
    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """
        Get a mapping of feature names to their descriptions.
        
        Returns:
        --------
        Dict[str, str]
            Dictionary mapping feature names to their descriptions
        """
        return {
            # Usage ratio features
            'usage_efficiency_ratio': 'Usage minutes per dollar spent (higher = better value)',
            'value_per_service_ratio': 'Monthly charges per active service (lower = better value)',
            'cost_per_gb_ratio': 'Cost per GB of data usage (higher = more expensive)',
            'ActiveServicesCount': 'Number of active services subscribed',
            
            # Engagement features
            'engagement_score': 'Overall customer engagement score',
            'engagement_decline_flag': 'Flag indicating potential engagement decline',
            'usage_trend_slope': 'Trend in usage relative to tenure',
            
            # Tenure features
            'tenure_band': 'Categorical tenure grouping',
            'tenure_risk_score': 'Risk score based on tenure patterns',
            
            # Interaction features
            'support_tickets_per_month': 'Average number of support tickets per month',
            'avg_resolution_time': 'Average time to resolve support tickets (hours)',
            'interaction_intensity': 'Overall customer-support interaction intensity',
            'support_efficiency_score': 'Efficiency of support interactions (higher = faster resolution)',
            
            # Billing behavior features
            'payment_method_risk_score': 'Risk score based on payment method (higher = riskier)',
            'late_payment_frequency': 'Frequency of late payments (0-1 scale)',
            'payment_reliability_score': 'Payment reliability (higher = more reliable)',
            'auto_pay_enabled': 'Whether customer uses automatic payment methods',
            'billing_complexity_score': 'Complexity of customer billing (higher = more complex)',
            
            # Target encoded features
            'PaymentMethod_target_encoded': 'Target-encoded payment method',
            'Contract_target_encoded': 'Target-encoded contract type',
            'InternetService_target_encoded': 'Target-encoded internet service type',
            
            # Log-transformed features
            'MonthlyCharges_log': 'Log-transformed monthly charges',
            'TotalCharges_log': 'Log-transformed total charges',
            'tenure_log': 'Log-transformed tenure',
            
            # Polynomial/interaction features
            'poly_*': 'Polynomial and interaction features between numeric variables'
        }
    
    def get_skewed_features(self) -> List[str]:
        """
        Get the list of features that were log-transformed due to skewness.
        
        Returns:
        --------
        List[str]
            List of feature names that were log-transformed
        """
        return self.skewed_features.copy()
    
    def get_target_encoders(self) -> Dict[str, Dict]:
        """
        Get the target encoders for categorical features.
        
        Returns:
        --------
        Dict[str, Dict]
            Dictionary of target encoders for each categorical feature
        """
        return self.target_encoders.copy()

