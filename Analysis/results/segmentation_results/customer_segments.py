import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.visualization import set_plot_style, save_figure

def load_clustered_data():
    """Load data with cluster assignments."""
    # Load the cleaned telco data
    data_path = os.path.join(os.path.dirname(__file__), '../../data/telco_churn_cleaned.csv')
    df = pd.read_csv(data_path)
    
    # Load the cluster profiles
    profiles_path = os.path.join(os.path.dirname(__file__), 'cluster_profiles.csv')
    profiles = pd.read_csv(profiles_path)
    
    # Assign clusters based on customer characteristics
    # This is a simplified approach for demo purposes
    df['Cluster'] = 0  # Default cluster
    
    # Use contract type, internet service, and payment method to assign clusters
    for idx, row in profiles.iterrows():
        cluster = row['Cluster']
        contract = row['Contract']
        internet = row['InternetService']
        payment = row['PaymentMethod']
        
        # Find matching customers and assign to this cluster
        mask = ((df['Contract'] == contract) & 
                (df['InternetService'] == internet) & 
                (df['PaymentMethod'] == payment))
        df.loc[mask, 'Cluster'] = cluster
    
    # Handle any remaining unassigned customers (assign to largest cluster)
    largest_cluster = profiles.loc[profiles['Size'].idxmax()]['Cluster']
    df.loc[df['Cluster'] == 0, 'Cluster'] = largest_cluster
    
    return df

def load_clustering_metrics():
    """Load clustering evaluation metrics."""
    metrics_path = os.path.join(os.path.dirname(__file__), 'clustering_metrics.json')
    with open(metrics_path, 'r') as f:
        return json.load(f)

def load_cluster_profiles():
    """Load cluster profile information."""
    profiles_path = os.path.join(os.path.dirname(__file__), 'cluster_profiles.csv')
    return pd.read_csv(profiles_path)

def analyze_segments(data, profiles):
    """Analyze segments and their characteristics."""
    print("\n===== CUSTOMER SEGMENT ANALYSIS =====\n")
    
    # Overview of segments
    print("SEGMENT OVERVIEW")
    print("-" * 80)
    for _, row in profiles.iterrows():
        print(f"Segment {row['Cluster']} ({row['Size']} customers, {row['ChurnRate']:.1%} churn rate):")
        print(f"  {row['Description']}")
        print(f"  Avg Monthly: ${row['AvgMonthlyCharges']:.2f}, Avg Total: ${row['AvgTotalCharges']:.2f}, Avg Tenure: {row['AvgTenure']:.1f} months")
        print(f"  Common services: {row['InternetService']}, {row['Contract']} contract, {row['PaymentMethod']}")
        print()
    
    # Churn rate by segment
    print("CHURN RATE BY SEGMENT")
    print("-" * 80)
    sorted_profiles = profiles.sort_values('ChurnRate', ascending=False)
    for _, row in sorted_profiles.iterrows():
        print(f"Segment {row['Cluster']}: {row['ChurnRate']:.1%} churn rate - {row['Description']}")
    
    # Create visualizations
    create_segment_visualizations(data, profiles)
    
    return profiles

def create_segment_visualizations(data, profiles):
    """Create visualizations for segment analysis."""
    # Set visualization style
    set_plot_style()
    
    # Directory to save visualizations
    output_dir = os.path.join(os.path.dirname(__file__), '../../images/segmentation')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Churn rate by segment
    plt.figure(figsize=(12, 6))
    sorted_profiles = profiles.sort_values('ChurnRate', ascending=False)
    bars = plt.bar(
        sorted_profiles['Cluster'].astype(str), 
        sorted_profiles['ChurnRate'], 
        color=sns.color_palette("viridis", len(profiles))
    )
    
    # Add labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01,
            f'{height:.1%}', 
            ha='center', va='bottom', fontsize=10
        )
    
    plt.title('Churn Rate by Customer Segment')
    plt.xlabel('Segment')
    plt.ylabel('Churn Rate')
    plt.ylim(0, max(sorted_profiles['ChurnRate']) * 1.15)  # Add some space for labels
    plt.tight_layout()
    save_figure(plt, os.path.join(output_dir, 'segment_churn_rates.png'))
    plt.close()
    
    # 2. Segment sizes
    plt.figure(figsize=(12, 6))
    plt.pie(
        profiles['Size'], 
        labels=profiles['Cluster'].astype(str),
        autopct='%1.1f%%', 
        startangle=90, 
        colors=sns.color_palette("viridis", len(profiles))
    )
    plt.title('Distribution of Customers Across Segments')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    save_figure(plt, os.path.join(output_dir, 'segment_sizes.png'))
    plt.close()
    
    # 3. Monthly charges vs. tenure by segment
    plt.figure(figsize=(14, 8))
    scatter = plt.scatter(
        data['tenure'], 
        data['MonthlyCharges'], 
        c=data['Cluster'], 
        cmap='viridis', 
        alpha=0.6, 
        s=50
    )
    plt.colorbar(scatter, label='Segment')
    plt.title('Monthly Charges vs. Tenure by Segment')
    plt.xlabel('Tenure (months)')
    plt.ylabel('Monthly Charges ($)')
    plt.grid(True, alpha=0.3)
    save_figure(plt, os.path.join(output_dir, 'segment_charges_tenure.png'))
    plt.close()
    
    # 4. Heatmap of average feature values by segment
    # Select numerical features
    numerical_features = ['MonthlyCharges', 'TotalCharges', 'tenure']
    
    # Calculate means by cluster
    segment_means = data.groupby('Cluster')[numerical_features].mean()
    
    # Normalize the data for better visualization
    scaler = StandardScaler()
    segment_means_scaled = pd.DataFrame(
        scaler.fit_transform(segment_means),
        index=segment_means.index,
        columns=segment_means.columns
    )
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        segment_means_scaled.T, 
        cmap='coolwarm',
        annot=True,
        fmt='.2f',
        linewidths=.5
    )
    plt.title('Normalized Average Values of Features by Segment')
    plt.xlabel('Segment')
    plt.tight_layout()
    save_figure(plt, os.path.join(output_dir, 'segment_features_heatmap.png'))
    plt.close()
    
    # 5. Contract type distribution by segment
    contract_segments = pd.crosstab(data['Cluster'], data['Contract'])
    contract_segments_pct = contract_segments.div(contract_segments.sum(axis=1), axis=0)
    
    plt.figure(figsize=(14, 7))
    contract_segments_pct.T.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title('Contract Type Distribution by Segment')
    plt.xlabel('Contract Type')
    plt.ylabel('Percentage')
    plt.tight_layout()
    save_figure(plt, os.path.join(output_dir, 'segment_contract_distribution.png'))
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    """Main function to run segment analysis."""
    # Load data
    data = load_clustered_data()
    profiles = load_cluster_profiles()
    
    # Analyze segments
    analyze_segments(data, profiles)
    
    print("Segment analysis completed successfully.")

if __name__ == "__main__":
    main() 