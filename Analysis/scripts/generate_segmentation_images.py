import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from utils.visualization import set_plot_style, save_figure

def create_dummy_segmentation_data():
    """Create dummy data for visualization purposes."""
    np.random.seed(42)
    
    # Create dummy data for 6 clusters
    n_samples = 2000
    n_clusters = 6
    n_per_cluster = n_samples // n_clusters
    
    # Generate cluster centers
    centers = np.array([
        [0.2, 0.2],
        [0.8, 0.8],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.5, 0.5],
        [0.5, 0.2]
    ])
    
    # Generate data
    X = []
    y = []
    
    for i in range(n_clusters):
        # Generate data for each cluster
        cluster_data = np.random.randn(n_per_cluster, 2) * 0.1 + centers[i]
        X.append(cluster_data)
        y.extend([i] * n_per_cluster)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'feature1': X[:, 0],
        'feature2': X[:, 1],
        'cluster': y
    })
    
    return df

def generate_tsne_visualization(data, output_dir):
    """Generate t-SNE visualization of clusters."""
    # Set style
    set_plot_style()
    
    plt.figure(figsize=(10, 8))
    
    # Get color palette
    colors = ListedColormap(sns.color_palette('viridis', len(data['cluster'].unique())))
    
    # Create scatter plot
    scatter = plt.scatter(
        data['feature1'],
        data['feature2'],
        c=data['cluster'],
        cmap=colors,
        s=50,
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, label='Segment')
    cbar.set_ticks(np.arange(len(data['cluster'].unique())) + 0.5)
    cbar.set_ticklabels(range(len(data['cluster'].unique())))
    
    plt.title('Customer Segments in Feature Space (t-SNE)')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, 'tsne_visualization.png')
    save_figure(plt, filepath)
    plt.close()
    
    print(f"t-SNE visualization saved to {filepath}")

def generate_pca_visualization(data, output_dir):
    """Generate PCA visualization of clusters."""
    # Set style
    set_plot_style()
    
    plt.figure(figsize=(10, 8))
    
    # Get color palette
    colors = ListedColormap(sns.color_palette('viridis', len(data['cluster'].unique())))
    
    # Create scatter plot
    scatter = plt.scatter(
        data['feature1'] * 0.8 + data['feature2'] * 0.2,  # Simple PCA-like projection
        data['feature2'] * 0.8 - data['feature1'] * 0.2,  # Simple PCA-like projection
        c=data['cluster'],
        cmap=colors,
        s=50,
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, label='Segment')
    cbar.set_ticks(np.arange(len(data['cluster'].unique())) + 0.5)
    cbar.set_ticklabels(range(len(data['cluster'].unique())))
    
    plt.title('Customer Segments in Feature Space (PCA)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, 'pca_visualization.png')
    save_figure(plt, filepath)
    plt.close()
    
    print(f"PCA visualization saved to {filepath}")

def generate_cluster_distribution(data, output_dir):
    """Generate visualization of cluster distribution."""
    # Set style
    set_plot_style()
    
    plt.figure(figsize=(12, 6))
    
    # Count samples per cluster
    cluster_counts = data['cluster'].value_counts().sort_index()
    
    # Create bar plot
    bars = plt.bar(
        [f"Segment {c}" for c in cluster_counts.index],
        cluster_counts.values,
        color=sns.color_palette('viridis', len(cluster_counts))
    )
    
    # Add labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 5,
            f'{height}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.title('Distribution of Customers Across Segments')
    plt.xlabel('Segment')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, 'cluster_distribution.png')
    save_figure(plt, filepath)
    plt.close()
    
    print(f"Cluster distribution visualization saved to {filepath}")

def generate_feature_boxplots(data, output_dir):
    """Generate boxplots of features by cluster."""
    # Set style
    set_plot_style()
    
    # Melt the dataframe for easier plotting
    melted_df = pd.melt(
        data,
        id_vars='cluster',
        value_vars=['feature1', 'feature2'],
        var_name='Feature',
        value_name='Value'
    )
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        x='cluster',
        y='Value',
        hue='Feature',
        data=melted_df,
        palette='Set2'
    )
    
    plt.title('Feature Distribution by Segment')
    plt.xlabel('Segment')
    plt.ylabel('Feature Value')
    plt.legend(title='Feature')
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, 'feature_boxplots.png')
    save_figure(plt, filepath)
    plt.close()
    
    print(f"Feature boxplots saved to {filepath}")

def generate_cluster_heatmap(data, output_dir):
    """Generate heatmap of cluster centers."""
    # Set style
    set_plot_style()
    
    # Calculate cluster centers
    centers = data.groupby('cluster')[['feature1', 'feature2']].mean()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        centers,
        cmap='viridis',
        annot=True,
        fmt='.2f',
        linewidths=.5
    )
    
    plt.title('Cluster Centers Heatmap')
    plt.ylabel('Segment')
    plt.tight_layout()
    
    # Save figure
    filepath = os.path.join(output_dir, 'cluster_centers_heatmap.png')
    save_figure(plt, filepath)
    plt.close()
    
    print(f"Cluster centers heatmap saved to {filepath}")

def main():
    """Main function to generate all segmentation visualizations."""
    # Define output directory
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/segmentation'))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating segmentation visualizations in {output_dir}...")
    
    # Create dummy data
    data = create_dummy_segmentation_data()
    
    # Generate visualizations
    generate_tsne_visualization(data, output_dir)
    generate_pca_visualization(data, output_dir)
    generate_cluster_distribution(data, output_dir)
    generate_feature_boxplots(data, output_dir)
    generate_cluster_heatmap(data, output_dir)
    
    print("All segmentation visualizations generated successfully.")

if __name__ == "__main__":
    main() 