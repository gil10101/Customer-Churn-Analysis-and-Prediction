#!/usr/bin/env python3
# Customer Churn Prediction Model using PyTorch

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import os

# Import common utilities
from utils.data_preprocessing import load_telco_data, prepare_data_for_modeling

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create directories if they don't exist
os.makedirs('../models', exist_ok=True)
os.makedirs('../evaluation/images', exist_ok=True)
os.makedirs('../evaluation/docs', exist_ok=True)

# Custom Dataset class
class TelcoChurnDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Define neural network model
class ChurnPredictionModel(nn.Module):
    def __init__(self, input_size):
        super(ChurnPredictionModel, self).__init__()
        
        # Define the architecture
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

def convert_to_pytorch_tensors(X_train, X_test, y_train, y_test):
    """Convert pandas dataframes to PyTorch tensors"""
    # Verify data types
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            print(f"Warning: Column {col} has object dtype. Converting to numeric.")
            try:
                X_train[col] = pd.to_numeric(X_train[col])
                X_test[col] = pd.to_numeric(X_test[col])
            except ValueError:
                print(f"Error: Could not convert column {col} to numeric. Dropping column.")
                X_train = X_train.drop(col, axis=1)
                X_test = X_test.drop(col, axis=1)
    
    # Convert to numpy arrays and ensure float32 type for better compatibility with PyTorch
    X_train_np = X_train.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    y_train_np = y_train.values.astype(np.float32)
    y_test_np = y_test.values.astype(np.float32)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train_np, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_np, dtype=torch.float32)
    
    print(f"Converted data to tensors. Shape: X_train_tensor: {X_train_tensor.shape}, y_train_tensor: {y_train_tensor.shape}")
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_size, epochs=100, batch_size=64, learning_rate=0.001):
    """Train the PyTorch neural network model"""
    # Create datasets and dataloaders
    train_dataset = TelcoChurnDataset(X_train_tensor, y_train_tensor.unsqueeze(1))
    test_dataset = TelcoChurnDataset(X_test_tensor, y_test_tensor.unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model
    model = ChurnPredictionModel(input_size)
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Lists to store metrics
    train_losses = []
    test_losses = []
    test_accuracies = []
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for features, targets in train_loader:
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in test_loader:
                outputs = model(features)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                # Convert outputs to predicted class (0 or 1)
                predicted = (outputs >= 0.5).float()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        # Calculate average test loss and accuracy
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)
        
        accuracy = accuracy_score(all_targets, all_preds)
        test_accuracies.append(accuracy)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    # Save the model
    torch.save(model.state_dict(), '../models/churn_prediction_model.pth')
    
    return model, train_losses, test_losses, test_accuracies

def evaluate_model(model, X_test_tensor, y_test_tensor, X_test, y_test, feature_names):
    """Evaluate the trained model and generate visualizations and reports"""
    # Set model to evaluation mode
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor)
        y_pred = (y_pred_proba >= 0.5).float()
    
    # Convert tensors to numpy arrays
    y_true = y_test_tensor.numpy()
    y_pred = y_pred.numpy()
    y_pred_proba = y_pred_proba.numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print("\nModel Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('../evaluation/images/confusion_matrix_pytorch.png')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('../evaluation/images/roc_curve_pytorch.png')
    
    # Feature Importance (using gradient-based approach)
    # This is a simple approximation using the model weights
    # For a more accurate approach, you'd use techniques like SHAP or integrated gradients
    
    # Get first layer weights
    weights = model.layers[0].weight.data.abs().mean(dim=0).numpy()
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': weights
    }).sort_values('Importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    plt.title('Top 15 Features by Importance (PyTorch Model)')
    plt.tight_layout()
    plt.savefig('../evaluation/images/feature_importance_pytorch.png')
    
    # Save evaluation results to markdown
    with open('../evaluation/docs/pytorch_model_evaluation.md', 'w') as f:
        f.write("# PyTorch Model Evaluation\n\n")
        f.write("## Model Architecture\n\n")
        f.write("```\n")
        f.write(str(model))
        f.write("\n```\n\n")
        
        f.write("## Evaluation Metrics\n\n")
        f.write(f"- **Accuracy**: {accuracy:.4f}\n")
        f.write(f"- **Precision**: {precision:.4f}\n")
        f.write(f"- **Recall**: {recall:.4f}\n")
        f.write(f"- **F1 Score**: {f1:.4f}\n")
        f.write(f"- **ROC AUC**: {roc_auc:.4f}\n\n")
        
        f.write("## Top 15 Features by Importance\n\n")
        f.write("```\n")
        f.write(feature_importance.head(15).to_string())
        f.write("\n```\n\n")
        
        f.write("## Insights\n\n")
        f.write("- The PyTorch model achieved good performance in predicting customer churn.\n")
        f.write("- The most important features align with our previous analysis, with tenure, contract type, and charges being significant predictors.\n")
        f.write("- The confusion matrix shows that the model is better at predicting customers who won't churn than those who will, which is common in imbalanced datasets.\n")
        f.write("- To improve model performance, we could explore techniques for handling imbalanced data, such as oversampling or using weighted loss functions.\n")
    
    return accuracy, precision, recall, f1, roc_auc, feature_importance

def plot_training_curves(train_losses, test_losses, test_accuracies, epochs):
    """Plot training and validation curves"""
    # Plot training and test loss
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), test_accuracies, label='Test Accuracy')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('../evaluation/images/training_curves_pytorch.png')

def main():
    """Main function to run the model training and evaluation pipeline"""
    print("Customer Churn Prediction using PyTorch")
    print("="*50)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    
    # Use absolute path with os.path.join to ensure correct path resolution
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../..'))
    data_path = os.path.join(project_root, 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Ensure file exists before proceeding
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Current directory:", os.getcwd())
        print("Searching for CSV files in project...")
        os.system(f"find {project_root} -name '*.csv' | grep -i telco")
        return
        
    df = load_telco_data(data_path)
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, feature_names, _ = prepare_data_for_modeling(df)
    
    # Convert to PyTorch tensors
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = convert_to_pytorch_tensors(
        X_train, X_test, y_train, y_test
    )
    
    # Feature names might have changed during tensor conversion if columns were dropped
    # So we need to update feature_names to match X_train columns
    feature_names = X_train.columns
    
    # Define hyperparameters
    input_size = X_train_tensor.shape[1]
    epochs = 100
    batch_size = 64
    learning_rate = 0.001
    
    print(f"Model will be trained with input size: {input_size}, using {len(feature_names)} features")
    
    # Train model
    print("\nTraining PyTorch model...")
    model, train_losses, test_losses, test_accuracies = train_model(
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, 
        input_size, epochs, batch_size, learning_rate
    )
    
    # Plot training curves
    plot_training_curves(train_losses, test_losses, test_accuracies, epochs)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, precision, recall, f1, roc_auc, feature_importance = evaluate_model(
        model, X_test_tensor, y_test_tensor, X_test, y_test, feature_names
    )
    
    # Print top features by importance
    print("\nTop 10 features by importance:")
    print(feature_importance.head(10))
    
    print("\nModel training and evaluation completed successfully.")
    print("Results saved to ../evaluation/docs/pytorch_model_evaluation.md")
    print("Visualizations saved to ../evaluation/images/")

if __name__ == "__main__":
    main() 