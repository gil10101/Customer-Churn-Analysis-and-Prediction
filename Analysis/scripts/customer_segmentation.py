#!/usr/bin/env python3
# Customer Segmentation using Clustering Techniques

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import umap.umap_ as umap
import joblib
import os

# Import common utilities
from utils.data_preprocessing import load_telco_data, get_numerical_categorical_columns

# Get the absolute path to the Analysis directory
script_dir = os.path.dirname(os.path.abspath(__file__))
analysis_dir = os.path.dirname(script_dir)

# Create directories if they don't exist
models_dir = os.path.join(analysis_dir, 'models')
images_dir = os.path.join(analysis_dir, 'images')
docs_dir = os.path.join(analysis_dir, 'docs')
data_dir = os.path.join(analysis_dir, 'data')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(docs_dir, exist_ok=True)

def prepare_data_for_clustering(df):
    """
    Prepare the data for clustering analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe
        
    Returns:
    --------
    tuple
        (features_scaled, feature_names, scaler)
    """
    # Make a copy to avoid modifying the original
    df_cluster = df.copy()
    
    # Drop customerID 
    if 'customerID' in df_cluster.columns:
        df_cluster = df_cluster.drop('customerID', axis=1)
    
    # Convert 'TotalCharges' to numeric
    df_cluster['TotalCharges'] = pd.to_numeric(df_cluster['TotalCharges'], errors='coerce')
    df_cluster['TotalCharges'] = df_cluster['TotalCharges'].fillna(0)
    
    # Handle 'Churn' column - extract before one-hot encoding
    churn_column = None
    if 'Churn' in df_cluster.columns:
        churn_values = df_cluster['Churn'].copy()
        churn_column = pd.Series(np.where(churn_values == 'Yes', 1, 0), index=df_cluster.index)
        # Remove Churn from the dataframe to prevent it from being one-hot encoded
        df_cluster = df_cluster.drop('Churn', axis=1)
    
    # Identify numerical and categorical columns
    numerical_cols, categorical_cols = get_numerical_categorical_columns(df_cluster)
    
    # One-hot encode categorical columns
    df_encoded = pd.get_dummies(df_cluster, columns=categorical_cols)
    
    # Verify that we have only numerical data before scaling
    for col in df_encoded.columns:
        if not pd.api.types.is_numeric_dtype(df_encoded[col]):
            print(f"Warning: Column {col} is not numeric. Converting to numeric.")
            df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
            df_encoded[col] = df_encoded[col].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_encoded)
    
    return features_scaled, df_encoded.columns, scaler, churn_column, df_encoded

def determine_optimal_clusters(features, max_clusters=10):
    """
    Determine the optimal number of clusters using the Elbow Method and Silhouette Score
    
    Parameters:
    -----------
    features : numpy.ndarray
        Scaled features for clustering
    max_clusters : int
        Maximum number of clusters to evaluate
        
    Returns:
    --------
    tuple
        (optimal_k_elbow, optimal_k_silhouette)
    """
    # Elbow Method
    inertia = []
    silhouette_scores = []
    
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)
        
        # Compute silhouette score
        labels = kmeans.labels_
        silhouette_scores.append(silhouette_score(features, labels))
    
    # Plot Elbow Method
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.title('Silhouette Score Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'optimal_clusters.png'))
    
    # Find optimal K using both methods
    # For elbow method - find the "elbow" point using second derivative
    diffs = np.diff(inertia)
    diffs_of_diffs = np.diff(diffs)
    optimal_k_elbow = np.argmax(diffs_of_diffs) + 2
    
    # For silhouette score - find the maximum score
    optimal_k_silhouette = np.argmax(silhouette_scores) + 2
    
    print(f"Optimal number of clusters: Elbow Method = {optimal_k_elbow}, Silhouette Score = {optimal_k_silhouette}")
    
    return optimal_k_elbow, optimal_k_silhouette

def perform_kmeans_clustering(features, n_clusters, feature_names=None):
    """
    Perform KMeans clustering
    
    Parameters:
    -----------
    features : numpy.ndarray
        Scaled features for clustering
    n_clusters : int
        Number of clusters to use
    feature_names : list, optional
        Names of features
        
    Returns:
    --------
    tuple
        (kmeans_model, labels, cluster_centers)
    """
    # Initialize and fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(features)
    
    # Get cluster labels and centers
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Save the model
    joblib.dump(kmeans, os.path.join(models_dir, 'kmeans_customer_segments.joblib'))
    
    # If feature names are provided, create a DataFrame with cluster centers
    if feature_names is not None:
        centers_df = pd.DataFrame(centers, columns=feature_names)
        centers_df.to_csv(os.path.join(data_dir, 'cluster_centers.csv'), index=False)
    
    return kmeans, labels, centers

def visualize_clusters_pca(features, labels, n_components=2):
    """
    Visualize clusters using PCA dimensionality reduction
    
    Parameters:
    -----------
    features : numpy.ndarray
        Scaled features for clustering
    labels : numpy.ndarray
        Cluster labels from KMeans
    n_components : int
        Number of PCA components
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)
    
    # Create DataFrame for visualization
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    pca_df['Cluster'] = labels
    
    # Visualize clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis', s=50, alpha=0.7)
    plt.title('Customer Segments - PCA Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'pca_clusters.png'))
    
    # Calculate explained variance ratio
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Explained variance by {n_components} principal components: {explained_variance:.2f}%")
    
    return pca, principal_components

def visualize_clusters_umap(features, labels):
    """
    Visualize clusters using UMAP dimensionality reduction
    
    Parameters:
    -----------
    features : numpy.ndarray
        Scaled features for clustering
    labels : numpy.ndarray
        Cluster labels from KMeans
    """
    # Apply UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(features)
    
    # Create DataFrame for visualization
    umap_df = pd.DataFrame(
        data=embedding,
        columns=['UMAP1', 'UMAP2']
    )
    umap_df['Cluster'] = labels
    
    # Visualize clusters
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='UMAP1', y='UMAP2', hue='Cluster', data=umap_df, palette='viridis', s=50, alpha=0.7)
    plt.title('Customer Segments - UMAP Visualization')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'umap_clusters.png'))
    
    return reducer, embedding

def analyze_clusters(df, labels, feature_names, churn_column=None):
    """
    Analyze the characteristics of each cluster
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe with features
    labels : numpy.ndarray
        Cluster labels from KMeans
    feature_names : list
        Names of features
    churn_column : pd.Series, optional
        Churn indicator for each customer
    """
    # Add cluster labels to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = labels
    
    # Add churn column back if provided
    if churn_column is not None:
        df_with_clusters['Churn'] = churn_column
    
    # Create cluster profiles
    cluster_profiles = []
    
    for cluster in range(len(np.unique(labels))):
        # Get subset of data for this cluster
        cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
        
        # Calculate key metrics for the cluster
        profile = {
            'Cluster': cluster,
            'Size': len(cluster_data),
            'Percentage': len(cluster_data) / len(df_with_clusters) * 100
        }
        
        # Add churn rate if churn column is available
        if churn_column is not None:
            profile['Churn_Rate'] = cluster_data['Churn'].mean() * 100
        
        # Add numerical feature averages
        numerical_cols = [col for col in feature_names if df_with_clusters[col].dtype in [np.int64, np.float64]]
        for col in numerical_cols:
            profile[f'Avg_{col}'] = cluster_data[col].mean()
        
        cluster_profiles.append(profile)
    
    # Create profiles DataFrame
    profiles_df = pd.DataFrame(cluster_profiles)
    
    # Save profiles to CSV
    profiles_df.to_csv(os.path.join(data_dir, 'cluster_profiles.csv'), index=False)
    
    # Create visualizations for each cluster
    
    # Bar chart of cluster sizes
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster', y='Size', data=profiles_df)
    plt.title('Cluster Sizes')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'cluster_sizes.png'))
    
    # If churn column is available, visualize churn rate by cluster
    if churn_column is not None:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Cluster', y='Churn_Rate', data=profiles_df)
        plt.title('Churn Rate by Cluster')
        plt.ylabel('Churn Rate (%)')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'cluster_churn_rates.png'))
    
    # Heatmap of cluster centers (for numerical features)
    numerical_features = [f'Avg_{col}' for col in numerical_cols]
    
    if len(numerical_features) > 0:
        plt.figure(figsize=(12, 10))
        heatmap_data = profiles_df.set_index('Cluster')[numerical_features]
        sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Cluster Centers Heatmap (Numerical Features)')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'cluster_centers_heatmap.png'))
    
    # Generate detailed report
    with open(os.path.join(docs_dir, 'cluster_analysis.md'), 'w') as f:
        f.write("# Customer Segmentation Analysis\n\n")
        
        f.write("## Cluster Profiles\n\n")
        f.write(profiles_df.to_markdown())
        
        f.write("\n\n## Cluster Descriptions\n\n")
        
        # Generate descriptions for each cluster
        for cluster in range(len(np.unique(labels))):
            cluster_data = df_with_clusters[df_with_clusters['Cluster'] == cluster]
            cluster_size = len(cluster_data)
            cluster_percentage = cluster_size / len(df_with_clusters) * 100
            
            f.write(f"### Cluster {cluster}\n\n")
            f.write(f"- **Size**: {cluster_size} customers ({cluster_percentage:.2f}% of total)\n")
            
            if churn_column is not None:
                churn_rate = cluster_data['Churn'].mean() * 100
                f.write(f"- **Churn Rate**: {churn_rate:.2f}%\n")
            
            f.write("- **Key Characteristics**:\n")
            
            # Find top 5 distinctive features
            distinctive_features = []
            
            for col in numerical_cols:
                cluster_mean = cluster_data[col].mean()
                overall_mean = df_with_clusters[col].mean()
                
                # Calculate how distinctive this feature is
                distinctiveness = abs((cluster_mean - overall_mean) / overall_mean) if overall_mean != 0 else 0
                
                distinctive_features.append({
                    'Feature': col,
                    'Cluster_Mean': cluster_mean,
                    'Overall_Mean': overall_mean,
                    'Distinctiveness': distinctiveness
                })
            
            # Sort by distinctiveness
            distinctive_features.sort(key=lambda x: x['Distinctiveness'], reverse=True)
            
            # Output top 5 distinctive features
            for i, feature in enumerate(distinctive_features[:5]):
                direction = "higher" if feature['Cluster_Mean'] > feature['Overall_Mean'] else "lower"
                f.write(f"  - {feature['Feature']}: {feature['Cluster_Mean']:.2f} ({direction} than average of {feature['Overall_Mean']:.2f})\n")
            
            f.write("\n")
    
    return profiles_df

def generate_marketing_strategies(cluster_profiles):
    """
    Generate targeted marketing strategies for each customer segment
    
    Parameters:
    -----------
    cluster_profiles : pd.DataFrame
        DataFrame containing profiles of each cluster
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with marketing strategies for each cluster
    """
    # Create a list to store strategies
    strategies = []
    
    for _, row in cluster_profiles.iterrows():
        cluster = int(row['Cluster'])
        
        # Base strategy object
        strategy = {
            'Cluster': cluster,
            'Segment_Name': f"Segment {cluster}",
            'Target_Description': "",
            'Retention_Strategy': "",
            'Cross_Sell_Opportunity': "",
            'Communication_Channel': "",
            'Priority': ""
        }
        
        # Determine segment characteristics and strategies based on data
        # This is a simplified example - in practice, would be more sophisticated
        
        # Example logic (customize based on actual data patterns):
        if 'Churn_Rate' in row:
            churn_rate = row['Churn_Rate']
            
            # High value, high churn risk
            if churn_rate > 30 and ('Avg_MonthlyCharges' in row and row['Avg_MonthlyCharges'] > 70):
                strategy['Segment_Name'] = "High-Value At-Risk"
                strategy['Target_Description'] = "High-spending customers with elevated churn risk"
                strategy['Retention_Strategy'] = "Premium retention offers, dedicated account manager, loyalty rewards"
                strategy['Cross_Sell_Opportunity'] = "Premium services with extended free trial periods"
                strategy['Communication_Channel'] = "Personal phone call, personalized email"
                strategy['Priority'] = "High"
            
            # High value, low churn risk
            elif churn_rate < 15 and ('Avg_MonthlyCharges' in row and row['Avg_MonthlyCharges'] > 70):
                strategy['Segment_Name'] = "Loyal High-Value"
                strategy['Target_Description'] = "High-spending loyal customers"
                strategy['Retention_Strategy'] = "Recognition program, early access to new features, status benefits"
                strategy['Cross_Sell_Opportunity'] = "Premium services, partner products"
                strategy['Communication_Channel'] = "Personalized email, app notifications"
                strategy['Priority'] = "Medium"
            
            # Low value, high churn risk
            elif churn_rate > 30 and ('Avg_MonthlyCharges' in row and row['Avg_MonthlyCharges'] < 50):
                strategy['Segment_Name'] = "Price-Sensitive Churners"
                strategy['Target_Description'] = "Budget-conscious customers with high churn risk"
                strategy['Retention_Strategy'] = "Economical plan options, bundled discounts"
                strategy['Cross_Sell_Opportunity'] = "Basic add-ons with clear cost-benefit messaging"
                strategy['Communication_Channel'] = "Email, SMS"
                strategy['Priority'] = "Medium-High"
            
            # Low value, low churn risk
            elif churn_rate < 15 and ('Avg_MonthlyCharges' in row and row['Avg_MonthlyCharges'] < 50):
                strategy['Segment_Name'] = "Stable Budget"
                strategy['Target_Description'] = "Loyal customers with lower spending"
                strategy['Retention_Strategy'] = "Maintain service quality, occasional rewards"
                strategy['Cross_Sell_Opportunity'] = "Gradual upgrade path with clear value proposition"
                strategy['Communication_Channel'] = "Email newsletters, app notifications"
                strategy['Priority'] = "Low"
            
            # Default case
            else:
                strategy['Segment_Name'] = f"Segment {cluster}"
                strategy['Target_Description'] = "Mixed customer group"
                strategy['Retention_Strategy'] = "General satisfaction monitoring, standard offers"
                strategy['Cross_Sell_Opportunity'] = "Targeted based on usage patterns"
                strategy['Communication_Channel'] = "Standard mix of channels"
                strategy['Priority'] = "Medium"
        
        strategies.append(strategy)
    
    # Create strategies DataFrame
    strategies_df = pd.DataFrame(strategies)
    
    # Save strategies to CSV
    strategies_df.to_csv(os.path.join(data_dir, 'marketing_strategies.csv'), index=False)
    
    # Generate detailed strategies document
    with open(os.path.join(docs_dir, 'marketing_strategies.md'), 'w') as f:
        f.write("# Targeted Marketing Strategies by Customer Segment\n\n")
        
        for _, strategy in strategies_df.iterrows():
            f.write(f"## {strategy['Segment_Name']}\n\n")
            f.write(f"**Target Description**: {strategy['Target_Description']}\n\n")
            f.write(f"**Priority**: {strategy['Priority']}\n\n")
            f.write("### Retention Strategy\n\n")
            f.write(f"{strategy['Retention_Strategy']}\n\n")
            f.write("### Cross-Sell Opportunities\n\n")
            f.write(f"{strategy['Cross_Sell_Opportunity']}\n\n")
            f.write("### Recommended Communication Channels\n\n")
            f.write(f"{strategy['Communication_Channel']}\n\n")
            f.write("---\n\n")
    
    return strategies_df

def perform_advanced_segmentation_dbscan(features, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering for advanced segmentation
    
    Parameters:
    -----------
    features : numpy.ndarray
        Scaled features for clustering
    eps : float
        The maximum distance between two samples for one to be considered as 
        in the neighborhood of the other
    min_samples : int
        The number of samples in a neighborhood for a point to be considered as a core point
        
    Returns:
    --------
    tuple
        (dbscan_model, labels)
    """
    # Initialize and fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(features)
    
    # Get cluster labels
    labels = dbscan.labels_
    
    # Count number of clusters (excluding noise if present)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Number of noise points: {n_noise} ({n_noise/len(labels)*100:.2f}% of data)")
    
    # Save the model
    joblib.dump(dbscan, os.path.join(models_dir, 'dbscan_customer_segments.joblib'))
    
    return dbscan, labels

def main():
    """Main function to run the customer segmentation pipeline"""
    print("Customer Segmentation using Clustering Techniques")
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
        return
        
    df = load_telco_data(data_path)
    
    # Prepare data for clustering
    features_scaled, feature_names, scaler, churn_column, df_encoded = prepare_data_for_clustering(df)
    
    # Determine optimal number of clusters
    optimal_k_elbow, optimal_k_silhouette = determine_optimal_clusters(features_scaled)
    
    # Choose the optimal k (you can use either method, here we'll use silhouette score)
    optimal_k = optimal_k_silhouette
    
    # Perform KMeans clustering
    kmeans_model, labels, centers = perform_kmeans_clustering(features_scaled, optimal_k, feature_names)
    
    # Visualize clusters using PCA
    pca, principal_components = visualize_clusters_pca(features_scaled, labels)
    
    # Visualize clusters using UMAP
    try:
        umap_reducer, embedding = visualize_clusters_umap(features_scaled, labels)
    except ImportError:
        print("UMAP not installed. Skipping UMAP visualization.")
    
    # Analyze clusters
    cluster_profiles = analyze_clusters(df_encoded, labels, feature_names, churn_column)
    
    # Generate marketing strategies
    marketing_strategies = generate_marketing_strategies(cluster_profiles)
    
    # Try advanced segmentation with DBSCAN
    print("\nPerforming advanced segmentation with DBSCAN...")
    
    # Find appropriate eps parameter using k-distance graph
    from sklearn.neighbors import NearestNeighbors
    
    # Compute the k-distance graph
    k = min(20, len(features_scaled)-1)
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(features_scaled)
    distances, indices = neigh.kneighbors(features_scaled)
    
    # Sort distances in ascending order
    distances = np.sort(distances[:, k-1])
    
    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-distance Graph')
    plt.xlabel('Data Points (sorted by distance)')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'k_distance_graph.png'))
    
    # Find the "elbow" in the k-distance graph
    # For simplicity, we'll choose a reasonable eps value based on visual inspection
    # In practice, you would look at the plot and choose the eps value at the "elbow"
    eps_value = np.percentile(distances, 10)  # Choose a value near the elbow
    
    # Perform DBSCAN clustering
    dbscan_model, dbscan_labels = perform_advanced_segmentation_dbscan(features_scaled, eps=eps_value, min_samples=5)
    
    # If DBSCAN produced meaningful clusters, analyze them
    if len(set(dbscan_labels)) > 1:
        print("\nAnalyzing DBSCAN clusters...")
        
        # Analyze DBSCAN clusters
        dbscan_profiles = analyze_clusters(df_encoded, dbscan_labels, feature_names, churn_column)
        
        # Generate marketing strategies for DBSCAN clusters
        dbscan_strategies = generate_marketing_strategies(dbscan_profiles)
    
    print("\nCustomer segmentation analysis completed successfully.")
    print("Results and visualizations saved to the 'Analysis' directory.")

if __name__ == "__main__":
    main() 