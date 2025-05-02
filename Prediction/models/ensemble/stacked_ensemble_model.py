import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
import time

# Custom data loading functions
def load_processed_data():
    """Load preprocessed customer data."""
    data_path = os.path.join(os.path.dirname(__file__), '../../../Analysis/data/telco_churn_cleaned.csv')
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        # Try alternative path
        data_path = os.path.join(os.path.dirname(__file__), '../../../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Could not find data file at {data_path}")
        print(f"Using original data file at {data_path}")
    
    return pd.read_csv(data_path)

def prepare_features_target(data):
    """Prepare features and target variable from data."""
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Define target variable
    if 'Churn' in df.columns:
        # Convert 'Yes'/'No' to 1/0 if needed
        if df['Churn'].dtype == 'object':
            df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
        y = df['Churn'].values
        df = df.drop('Churn', axis=1)
    else:
        # If no Churn column, use a dummy variable for testing
        print("Warning: No 'Churn' column found. Creating dummy target variable.")
        y = np.random.randint(0, 2, size=len(df))
    
    # Check for and handle missing values
    if df.isnull().sum().sum() > 0:
        print("Warning: Dataset contains missing values. Filling with appropriate values.")
        # Fill numeric columns with median
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical columns with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Handle categorical variables
    cat_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    # Get final feature names
    feature_names = df_encoded.columns.tolist()
    
    # Convert to numpy array for machine learning
    X = df_encoded.values
    
    return X, y, feature_names

class StackedEnsembleChurnModel:
    """
    Stacked ensemble model for customer churn prediction.
    Uses a combination of base models and a meta-model to combine their predictions.
    """
    def __init__(self):
        # Define base models
        self.base_models = [
            ('random_forest', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1
            )),
            ('gradient_boosting', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                subsample=0.8
            )),
            ('xgboost', XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                scale_pos_weight=2
            ))
        ]
        
        # Define meta-model
        self.meta_model = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            solver='liblinear',
            random_state=42
        )
        
        # Scaler for standardizing features
        self.scaler = StandardScaler()
        
        # Store trained models
        self.trained_base_models = []
        self.feature_names = None
        
    def _generate_meta_features(self, X, y, base_models, n_splits=5):
        """Generate meta-features using cross-validation."""
        meta_features = np.zeros((X.shape[0], len(base_models)))
        
        # Define cross-validation strategy
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # For each base model
        for i, (name, model) in enumerate(base_models):
            print(f"Generating meta-features for {name}...")
            # For each fold in cross-validation
            for train_idx, val_idx in kf.split(X, y):
                # Split data
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold = y[train_idx]
                
                # Train model on training fold
                model.fit(X_train_fold, y_train_fold)
                
                # Predict probabilities for validation fold
                meta_features[val_idx, i] = model.predict_proba(X_val_fold)[:, 1]
        
        return meta_features
    
    def fit(self, X, y):
        """Train the ensemble model."""
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)
        
        print("Generating meta-features...")
        meta_features = self._generate_meta_features(X_scaled, y, self.base_models)
        
        print("Training base models on full dataset...")
        self.trained_base_models = []
        for name, model in self.base_models:
            print(f"Training {name} on full dataset...")
            model.fit(X_scaled, y)
            self.trained_base_models.append((name, model))
        
        print("Training meta-model...")
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """Generate predictions from the ensemble."""
        X_scaled = self.scaler.transform(X)
        
        # Generate base model predictions
        meta_features = np.zeros((X.shape[0], len(self.trained_base_models)))
        for i, (name, model) in enumerate(self.trained_base_models):
            meta_features[:, i] = model.predict_proba(X_scaled)[:, 1]
        
        # Use meta-model to make final predictions
        return self.meta_model.predict(meta_features)
    
    def predict_proba(self, X):
        """Generate probability predictions from the ensemble."""
        X_scaled = self.scaler.transform(X)
        
        # Generate base model predictions
        meta_features = np.zeros((X.shape[0], len(self.trained_base_models)))
        for i, (name, model) in enumerate(self.trained_base_models):
            meta_features[:, i] = model.predict_proba(X_scaled)[:, 1]
        
        # Use meta-model to make final probability predictions
        return self.meta_model.predict_proba(meta_features)
    
    def save(self, filepath):
        """Save the ensemble model to disk."""
        joblib.dump(self, filepath)
        
    @staticmethod
    def load(filepath):
        """Load an ensemble model from disk."""
        return joblib.load(filepath)

def train_ensemble_model():
    """Train and evaluate the stacked ensemble model."""
    print("Loading and preparing data...")
    # Load preprocessed data
    data = load_processed_data()
    
    # Prepare features and target
    X, y, feature_names = prepare_features_target(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create and train the ensemble model
    print("\nTraining stacked ensemble model...")
    start_time = time.time()
    
    ensemble = StackedEnsembleChurnModel()
    ensemble.feature_names = feature_names
    ensemble.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Model trained in {training_time:.2f} seconds")
    
    # Make predictions
    y_pred = ensemble.predict(X_test)
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'log_loss': log_loss(y_test, y_pred_proba),
        'execution_time': training_time
    }
    
    # Print metrics
    print("\nStacked Ensemble Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save the model and metrics
    save_dir = os.path.dirname(__file__)
    model_path = os.path.join(save_dir, 'stacked_ensemble_model.joblib')
    ensemble.save(model_path)
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, 'ensemble_metrics.csv'), index=False)
    
    # Save meta-model coefficients (importance of each base model)
    meta_coefs = pd.DataFrame({
        'Model': [name for name, _ in ensemble.base_models],
        'Coefficient': ensemble.meta_model.coef_[0]
    })
    meta_coefs.to_csv(os.path.join(save_dir, 'meta_model_coefficients.csv'), index=False)
    
    print(f"\nModel and metrics saved to {save_dir}")
    
    return ensemble, metrics

if __name__ == "__main__":
    train_ensemble_model() 