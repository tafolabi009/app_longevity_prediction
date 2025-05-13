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
        plt_fonts = ['Helvetica', 'Arial', 'Tahoma']
    else:  # Linux and others
        plt_fonts = ['DejaVu Sans', 'Liberation Sans', 'FreeSans', 'Nimbus Sans L']
        
    # Find available font
    available_font = None
    for font in plt_fonts:
        try:
            if any(font.lower() in f.lower() for f in mpl.font_manager.findSystemFonts()):
                available_font = font
                break
        except:
            continue
            
    # Set font properties
    if available_font:
        plt.rcParams['font.family'] = available_font
    
    # Set better default styles
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("notebook")
    
    # Set color palette
    sns.set_palette("deep")

def create_feature_importance_plot(model_name, feature_names, importance_values, top_n=20):
    """
    Create a better feature importance plot with improved layout and font handling.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    feature_names : list
        List of feature names
    importance_values : array-like
        Feature importance values
    top_n : int
        Number of top features to show
    """
    # Get top features
    if len(importance_values) > top_n:
        indices = np.argsort(importance_values)[-top_n:]
        feature_names = [feature_names[i] for i in indices]
        importance_values = importance_values[indices]
    
    # Create plot with adequate margins
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importance_values, align='center')
    
    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feature_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top Features - {model_name}')
    
    # Add values as text
    for i, v in enumerate(importance_values):
        ax.text(v + 0.01, i, f"{v:.3f}", va='center')
    
    # Add grid
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.1)
    
    # Save the plot
    plt.savefig(f'reports/{model_name}_feature_importance.png')
    plt.close()

def create_model_comparison_plot(model_metrics):
    """
    Create an improved model comparison visualization.
    
    Parameters:
    -----------
    model_metrics : dict
        Dictionary of model metrics
    """
    metrics_to_plot = ['rmse', 'r2', 'mae']
    metric_labels = {
        'rmse': 'RMSE (lower is better)',
        'r2': 'R² Score (higher is better)',
        'mae': 'MAE (lower is better)'
    }
    
    # Create the figure
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 12), sharex=True)
    
    # Get model names
    models = list(model_metrics.keys())
    
    # Colors
    colors = sns.color_palette("deep", len(models))
    
    # Plot each metric
    for i, metric in enumerate(metrics_to_plot):
        values = [m[metric] for m in model_metrics.values()]
        
        # Sort models by metric performance (ascending for rmse/mae, descending for r2)
        sorted_indices = np.argsort(values)
        if metric == 'r2':
            sorted_indices = sorted_indices[::-1]  # Reverse for R²
            
        sorted_models = [models[j] for j in sorted_indices]
        sorted_values = [values[j] for j in sorted_indices]
        sorted_colors = [colors[j] for j in sorted_indices]
        
        # Create horizontal bar chart
        bars = axes[i].barh(sorted_models, sorted_values, color=sorted_colors)
        
        # Add values as text
        for j, bar in enumerate(bars):
            text_x = bar.get_width() * (1.02 if metric != 'r2' else 1.01)
            text_y = bar.get_y() + bar.get_height()/2
            axes[i].text(text_x, text_y, f"{sorted_values[j]:.4f}", 
                         va='center', ha='left', fontsize=10)
        
        # Set title and labels
        axes[i].set_title(metric_labels[metric])
        axes[i].grid(axis='x', linestyle='--', alpha=0.7)
        
        # Highlight the best model
        if metric == 'r2':
            best_idx = 0  # First one is best (highest R²)
        else:
            best_idx = 0  # First one is best (lowest RMSE/MAE)
            
        bars[best_idx].set_color('green')
        
    # Adjust layout and spacing
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    
    # Add overall title
    fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
    
    # Save figure
    plt.savefig('reports/model_comparison.png', bbox_inches='tight')
    plt.close()

def create_predictions_plot(model_name, y_true, y_pred):
    """
    Create an improved actual vs predicted plot.
    
    Parameters:
    -----------
    model_name : str
        Name of the model
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    """
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    plt.scatter(y_true, y_pred, alpha=0.6, color='#1f77b4', edgecolor='k', s=80)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), 'b-', linewidth=2, label=f'Trend Line (y = {z[0]:.3f}x + {z[1]:.3f})')
    
    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - Actual vs Predicted Values')
    plt.legend()
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add performance metrics text
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    metrics_text = f"RMSE: {rmse:.4f}\nR²: {r2:.4f}\nMAE: {mae:.4f}"
    plt.annotate(metrics_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                verticalalignment='top', fontsize=12)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f'reports/{model_name}_predictions.png')
    plt.close() 
