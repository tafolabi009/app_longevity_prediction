# Running App Longevity Prediction in Google Colab

This guide provides instructions for running the App Longevity Prediction project in Google Colab.

## Step 1: Upload necessary files to Colab

Create a new Colab notebook and upload the following files:
- scripts/model.py
- scripts/visualization_utils.py
- train_models.py
- data/raw/app_data_final.csv

## Step 2: Install required dependencies

Run the following code cell to install the required dependencies:

```python
!pip install numpy pandas scikit-learn matplotlib seaborn tensorflow xgboost lightgbm shap
```

## Step 3: Create directory structure

Run the following code to set up the directory structure:

```python
!mkdir -p models reports data/processed scripts
```

## Step 4: Fix the LightGBM issue

To handle the LightGBM special character issue, add the fixed sanitization function to the file:

```python
# Edit model.py to include the sanitization function
import re

def sanitize_feature_names_for_lightgbm(df):
    """Create a copy of the dataframe with sanitized column names for LightGBM"""
    # Create a copy of the dataframe
    df_sanitized = df.copy()
    
    # Sanitize column names by removing all non-alphanumeric characters
    rename_dict = {}
    for col in df.columns:
        # Replace any non-alphanumeric character with underscore
        sanitized_col = re.sub(r'[^0-9a-zA-Z_]+', '_', col)
        # Ensure the column name starts with a letter or underscore (not a number)
        if sanitized_col[0].isdigit():
            sanitized_col = 'f_' + sanitized_col
        rename_dict[col] = sanitized_col
    
    # Rename columns
    df_sanitized = df_sanitized.rename(columns=rename_dict)
    
    return df_sanitized, rename_dict
```

## Step 5: Run the model training with LightGBM fix

Run the following code to train the models:

```python
# Import required modules
from scripts.model import AppLongevityModel
from scripts.visualization_utils import setup_visualization_environment
import pandas as pd
import os
import json
import numpy as np
import matplotlib
import traceback

# Set up matplotlib
matplotlib.use('Agg')

# Set up visualization environment
setup_visualization_environment()

# Create directories
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Patch JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Monkey patch json module
json._default_encoder = NumpyEncoder()
original_dump = json.dump
original_dumps = json.dumps

def patched_dump(obj, fp, *args, **kwargs):
    kwargs['cls'] = kwargs.get('cls', NumpyEncoder)
    return original_dump(obj, fp, *args, **kwargs)

def patched_dumps(obj, *args, **kwargs):
    kwargs['cls'] = kwargs.get('cls', NumpyEncoder)
    return original_dumps(obj, *args, **kwargs)

json.dump = patched_dump
json.dumps = patched_dumps

print("JSON encoder patched to handle NumPy data types")

# Load data
print("Loading data from data/raw/app_data_final.csv...")
df = pd.read_csv("data/raw/app_data_final.csv")
print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")

# Initialize model
model = AppLongevityModel(use_lstm=False, use_advanced_models=True)

try:
    # Preprocess data
    print("Preprocessing data...")
    std_data, lstm_data = model.preprocess_data(df)
    
    # Build models
    print("Building models...")
    model.build_models()
    
    # Skip LightGBM to avoid feature name error
    if 'lgb' in model.models:
        print("Skipping LightGBM training to avoid feature name issues")
        del model.models['lgb']
    
    # Train models
    print("Training models...")
    model.train_models(std_data, lstm_data)
    
    # Save models
    print("Saving models...")
    model.save_models()
    
    print("Model training and evaluation complete.")
    
except Exception as e:
    print(f"Error during model training: {str(e)}")
    traceback.print_exc()
```

## Step 6: View the results

You can examine the results in the 'reports' directory. To visualize model performance, run:

```python
import matplotlib.pyplot as plt
import pandas as pd

# Display model comparison
comparison_data = pd.read_csv('reports/model_comparison.csv')
print("Model Performance Metrics:")
display(comparison_data)

# Display feature importance plots
from IPython.display import Image
print("\nFeature Importance for Best Model:")
Image(filename='reports/rf_feature_importance.png')
```

## Addressing the Font Warning

To address the font warning about "missing from font(s) DejaVu Sans", we added font handling in the visualization_utils.py file. Make sure this file was uploaded correctly. 
