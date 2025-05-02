import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
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

def train_baseline_model():
    """
    Train a baseline logistic regression model for customer churn prediction.
    This serves as a benchmark for more complex models.
    """
    print("Loading and preparing data...")
    # Load preprocessed data
    data = load_processed_data()
    
    # Prepare features and target
    X, y, feature_names = prepare_features_target(data)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create a pipeline with scaling and logistic regression
    print("Creating and training baseline logistic regression model...")
    start_time = time.time()
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=1.0,
            class_weight='balanced',
            solver='liblinear',
            random_state=42,
            max_iter=1000
        ))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Model trained in {training_time:.2f} seconds")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
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
    print("\nBaseline Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save feature importance (coefficients)
    coefficients = model.named_steps['classifier'].coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': np.abs(coefficients),
        'Sign': np.sign(coefficients)
    })
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Save the model and metrics
    save_dir = os.path.dirname(__file__)
    joblib.dump(model, os.path.join(save_dir, 'baseline_logistic_regression.joblib'))
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(save_dir, 'baseline_feature_importance.csv'), index=False)
    
    # Save metrics
    pd.DataFrame([metrics]).to_csv(os.path.join(save_dir, 'baseline_metrics.csv'), index=False)
    
    print(f"\nModel and metrics saved to {save_dir}")
    
    return model, metrics, feature_importance

if __name__ == "__main__":
    train_baseline_model() 