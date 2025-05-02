import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from utils.visualization import set_plot_style, save_figure

def load_performance_metrics():
    """Load model performance metrics from CSV file."""
    metrics_path = os.path.join(os.path.dirname(__file__), 'model_performance_metrics.csv')
    return pd.read_csv(metrics_path)

def create_comparison_barplot(metrics_df, metric_name):
    """Create a bar plot comparing models on a specific metric."""
    plt.figure(figsize=(12, 6))
    
    # Sort by the metric value in descending order
    sorted_df = metrics_df.sort_values(by=metric_name, ascending=False)
    
    # Create bar plot
    bars = plt.bar(sorted_df['Model'], sorted_df[metric_name], color=sns.color_palette("viridis", len(sorted_df)))
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.title(f'Model Comparison: {metric_name}')
    plt.ylabel(metric_name)
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, max(sorted_df[metric_name]) * 1.15)  # Add some space for the labels
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), f'../images/model_comparison_{metric_name.lower()}.png')
    save_figure(plt, output_path)
    plt.close()

def create_radar_plot(metrics_df):
    """Create a radar plot comparing key metrics across models."""
    # Select metrics for the radar plot
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']
    
    # Prepare the data
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Add the metrics labels to the plot
    plt.xticks(angles[:-1], metrics, size=12)
    
    # Draw y-axis labels (values)
    ax.set_rlabel_position(0)
    ax.set_ylim(0.4, 1)
    plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"], size=10)
    
    # Plot each model
    for idx, row in metrics_df.iterrows():
        values = row[metrics].values.tolist()
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2, label=row['Model'])
        ax.fill(angles, values, alpha=0.1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Model Comparison: Radar Plot of Key Metrics', size=15, y=1.1)
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '../images/model_comparison_radar.png')
    save_figure(plt, output_path)
    plt.close()

def create_tradeoff_plot(metrics_df):
    """Create a scatter plot showing precision-recall tradeoff."""
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    scatter = plt.scatter(
        metrics_df['Precision'], 
        metrics_df['Recall'], 
        s=metrics_df['Execution_Time'] * 5,  # Size based on execution time
        c=metrics_df['F1_Score'],  # Color based on F1 score
        cmap='viridis',
        alpha=0.7
    )
    
    # Add labels for each point
    for i, model in enumerate(metrics_df['Model']):
        plt.annotate(
            model, 
            (metrics_df['Precision'].iloc[i], metrics_df['Recall'].iloc[i]),
            xytext=(7, 0),
            textcoords='offset points',
            fontsize=9
        )
    
    # Add a colorbar and legend
    cbar = plt.colorbar(scatter)
    cbar.set_label('F1 Score')
    
    # Add reference lines for perfect precision and recall
    plt.axhline(y=1, color='gray', linestyle='--', alpha=0.3)
    plt.axvline(x=1, color='gray', linestyle='--', alpha=0.3)
    
    plt.title('Precision-Recall Tradeoff with F1 Score and Execution Time')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), '../images/model_comparison_tradeoff.png')
    save_figure(plt, output_path)
    plt.close()

def main():
    """Main function to run model comparison analysis."""
    # Set the style for all plots
    set_plot_style()
    
    # Load metrics data
    metrics_df = load_performance_metrics()
    
    # Create individual bar plots for each metric
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC_ROC']:
        create_comparison_barplot(metrics_df, metric)
    
    # Create radar plot
    create_radar_plot(metrics_df)
    
    # Create precision-recall tradeoff plot
    create_tradeoff_plot(metrics_df)
    
    print("Model comparison analysis completed successfully.")
    print(f"Results saved to {os.path.abspath(os.path.join(os.path.dirname(__file__), '../images/'))}")

if __name__ == "__main__":
    main() 