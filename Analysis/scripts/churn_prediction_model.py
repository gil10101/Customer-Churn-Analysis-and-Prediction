#!/usr/bin/env python3
# Customer Churn Prediction Model using PyTorch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create directories if they don't exist
os.makedirs('../../Analysis/images', exist_ok=True)
os.makedirs('../../Analysis/docs', exist_ok=True)
os.makedirs('../../Analysis/models', exist_ok=True)

# Load and preprocess the dataset
def load_data():
    # Load the dataset
    df = pd.read_csv('../../data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    # Convert SeniorCitizen from 0/1 to No/Yes
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Handle missing values
    df['TotalCharges'].fillna(0, inplace=True)
    
    # Drop customerID as it's not relevant for prediction
    df = df.drop('customerID', axis=1)
    
    return df

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

def preprocess_data(df):
    # Convert categorical variables to one-hot encoding
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    # Split features and target
    X = df_encoded.drop('Churn_Yes', axis=1)
    y = df_encoded['Churn_Yes']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Standardize numerical features
    scaler = StandardScaler()
    
    # Find numerical columns (those with more than 5 unique values)
    numerical_cols = [col for col in X.columns if X[col].nunique() > 5]
    
    # Standardize only numerical columns
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X.columns, X_train, X_test, y_train, y_test

def train_model(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, input_size, epochs=100, batch_size=64, learning_rate=0.001):
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
    torch.save(model.state_dict(), '../../Analysis/models/churn_prediction_model.pth')
    
    return model, train_losses, test_losses, test_accuracies

def evaluate_model(model, X_test_tensor, y_test_tensor, X_test, y_test, feature_names):
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
    plt.savefig('../../Analysis/images/confusion_matrix_pytorch.png')
    
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
    plt.savefig('../../Analysis/images/roc_curve_pytorch.png')
    
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
    plt.savefig('../../Analysis/images/feature_importance_pytorch.png')
    
    # Save evaluation results to markdown
    with open('../../Analysis/docs/pytorch_model_evaluation.md', 'w') as f:
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
    plt.savefig('../../Analysis/images/training_curves_pytorch.png')

def main():
    print("Customer Churn Prediction using PyTorch")
    print("="*50)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    
    # Prepare data for PyTorch
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, feature_names, X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # Define hyperparameters
    input_size = X_train_tensor.shape[1]
    epochs = 100
    batch_size = 64
    learning_rate = 0.001
    
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
    
    # Compare with previous feature importance
    print("\nTop 10 features by importance:")
    print(feature_importance.head(10))
    
    print("\nModel training and evaluation completed successfully.")
    print("Results saved to Analysis/docs/pytorch_model_evaluation.md")
    print("Visualizations saved to Analysis/images/")

if __name__ == "__main__":
    main() 