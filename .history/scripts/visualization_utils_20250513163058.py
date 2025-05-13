"""
Visualization utilities for App Longevity Prediction models
This module provides functions to set up the visualization environment and create consistent visualizations.
"""

import os
import sys
import platform
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.font_manager import FontProperties

def setup_visualization_environment():
    """
    Set up the visualization environment with proper font configuration
    and style settings to avoid warnings and ensure consistent visuals across platforms.
    """
    # Set the visual style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    
    # Use Agg backend if not already set
    if matplotlib.get_backend() != 'Agg':
        matplotlib.use('Agg')
    
    # Platform-specific font handling
    system = platform.system()
    
    # Create fonts directory if it doesn't exist
    os.makedirs('reports/fonts', exist_ok=True)
    
    if system == 'Linux':
        # Check if running in Colab
        is_colab = 'google.colab' in sys.modules
        
        if is_colab:
            # Install fonts in Colab
            try:
                !apt-get update -qq
                !apt-get install -y fonts-dejavu
                # Clear font cache
                matplotlib.font_manager._rebuild()
                print("Installed DejaVu fonts in Colab")
            except:
                print("Could not install fonts via apt-get. Using fallback.")
                
        # Set font family to a widely available font
        plt.rcParams['font.family'] = 'DejaVu Sans'
        
    elif system == 'Windows':
        # Windows has Arial by default
        plt.rcParams['font.family'] = 'Arial'
        
    elif system == 'Darwin':  # macOS
        plt.rcParams['font.family'] = 'Helvetica'
    
    # Adjust other parameters to avoid warnings
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['figure.autolayout'] = False  # Disable tight_layout to avoid warnings
    
    # Add padding to avoid cut-off labels
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.1
    plt.rcParams['figure.constrained_layout.w_pad'] = 0.1
    
    # Increase font sizes for better readability
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    return True

def create_feature_importance_plot(model_name, feature_names, importance_values):
    """
    Create a feature importance plot with improved styling and saved to the reports directory.
    
    Args:
        model_name (str): Name of the model
        feature_names (list): List of feature names
        importance_values (list): List of importance values
    """
    # Create figure with sufficient size for many features
    plt.figure(figsize=(12, max(8, len(feature_names) * 0.3)))
    
    # Sort features by importance
    indices = np.argsort(importance_values)
    
    # Get top 20 features or all if less than 20
    num_features = min(20, len(feature_names))
    top_indices = indices[-num_features:]
    
    # Create horizontal bar plot
    bars = plt.barh(
        range(num_features), 
        [importance_values[i] for i in top_indices],
        align='center',
        color='#5975a4',
        edgecolor='none',
        alpha=0.8
    )
    
    # Set labels and title with sufficient padding
    plt.yticks(range(num_features), [feature_names[i] for i in top_indices])
    plt.xlabel('Importance')
    plt.title(f'{model_name.upper()} Feature Importance')
    
    # Add grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add value labels to bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height()/2,
            f'{importance_values[top_indices[i]]:.3f}',
            va='center'
        )
    
    # Set tight layout with adjustments to avoid warnings
    fig = plt.gcf()
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    
    # Save figure with high quality and tight bounding box
    plt.savefig(f'reports/{model_name}_feature_importance.png', 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0.2)
    plt.close()

def create_model_comparison_plot(model_metrics):
    """
    Create a comparison plot of model metrics.
    
    Args:
        model_metrics (dict): Dictionary of model metrics
    """
    # Extract metrics for plotting
    models = list(model_metrics.keys())
    rmse_values = [model_metrics[model]['rmse'] for model in models]
    r2_values = [model_metrics[model]['r2'] for model in models]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Plot RMSE (lower is better)
    bars1 = ax1.bar(models, rmse_values, color='#d65f5f', alpha=0.8)
    ax1.set_title('RMSE by Model (Lower is Better)')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('RMSE')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels to bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Plot R² (higher is better)
    bars2 = ax2.bar(models, r2_values, color='#5975a4', alpha=0.8)
    ax2.set_title('R² by Model (Higher is Better)')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('R²')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels to bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.4f}', ha='center', va='bottom')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle('Model Performance Comparison', fontsize=16, y=0.98)
    
    # Save figure
    plt.savefig('reports/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_predictions_plot(model_name, y_true, y_pred):
    """
    Create a visualization of model predictions vs actual values.
    
    Args:
        model_name (str): Name of the model
        y_true (array): True values
        y_pred (array): Predicted values
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Set up 2x2 subplot grid
    gs = plt.GridSpec(2, 2, height_ratios=[3, 1])
    
    # Scatter plot of predicted vs actual
    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(y_true, y_pred, alpha=0.6, color='#5975a4', edgecolor='k', s=50)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    ax1.set_xlabel('Actual Longevity')
    ax1.set_ylabel('Predicted Longevity')
    ax1.set_title('Predicted vs Actual Values')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Add annotations for a few outliers
    residuals_abs = np.abs(residuals)
    outlier_indices = np.argsort(residuals_abs)[-3:]  # Top 3 outliers
    
    for i in outlier_indices:
        ax1.annotate(
            f'Residual: {residuals[i]:.3f}',
            xy=(y_true[i], y_pred[i]),
            xytext=(10, 10),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color='black')
        )
    
    # Histogram of residuals
    ax2 = plt.subplot(gs[0, 1])
    ax2.hist(residuals, bins=10, color='#d65f5f', alpha=0.7, edgecolor='k')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Residuals')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Residuals plot
    ax3 = plt.subplot(gs[1, :])
    ax3.scatter(y_pred, residuals, alpha=0.6, color='#5975a4', edgecolor='k', s=50)
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted Longevity')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residuals vs Predicted Values')
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # Add overall title
    plt.suptitle(f'{model_name.upper()} Prediction Analysis', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save figure
    plt.savefig(f'reports/{model_name}_predictions.png', dpi=300, bbox_inches='tight')
    plt.close() 
