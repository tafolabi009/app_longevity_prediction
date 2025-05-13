#!/usr/bin/env python3

"""
Simplified App Longevity Prediction model training script for Google Colab
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import json
import traceback
import re

# Set non-interactive backend before importing plt
matplotlib.use('Agg')

# Custom JSON encoder to handle NumPy data types
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

def main():
    """Run the model training process with proper visualization setup"""
    
    # Create required directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Import modules once directories are created
    # This allows setup_visualization_environment to save fonts if needed
    from scripts.visualization_utils import setup_visualization_environment
    from scripts.model import AppLongevityModel
    
    # Set up visualization environment
    print("Setting up visualization environment...")
    setup_visualization_environment()
    
    # Monkey patch json.JSONEncoder if needed
    try:
        # Check if np.float32 is not already handled by the default encoder
        json.dumps(np.float32(1.0))
    except TypeError:
        # Replace the default encoder with our NumpyEncoder
        json._default_encoder = NumpyEncoder()
        
        # Monkey patch the json module's dump and dumps methods
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
    data_file = "data/raw/app_data_final.csv"
    try:
        print(f"Loading data from {data_file}...")
        df = pd.read_csv(data_file)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"Error: {data_file} not found. Please upload the file first.")
        return 1
    
    # Initialize and run model
    model = AppLongevityModel(use_lstm=False)
    
    try:
        # Preprocess data
        print("Preprocessing data...")
        std_data, lstm_data = model.preprocess_data(df)
        
        # Build models
        print("Building models...")
        model.build_models()
        
        # Monkey patch LightGBM training in the _train method to use sanitized feature names
        original_train = model.train_models
        
        def patched_train(std_data, lstm_data=None):
            try:
                # Make the model use our sanitization function for LightGBM
                original_sanitize_feature_names = getattr(model, '_sanitize_feature_names_for_lightgbm', None)
                model._sanitize_feature_names_for_lightgbm = sanitize_feature_names_for_lightgbm
                
                return original_train(std_data, lstm_data)
            except Exception as e:
                if "lightgbm" in str(e).lower():
                    print(f"LightGBM error detected: {str(e)}")
                    print("Continuing without LightGBM model...")
                    
                    # Remove LightGBM from models if it exists
                    if 'lgb' in model.models:
                        del model.models['lgb']
                    
                    # Rerun training without LightGBM
                    if original_sanitize_feature_names:
                        model._sanitize_feature_names_for_lightgbm = original_sanitize_feature_names
                    else:
                        delattr(model, '_sanitize_feature_names_for_lightgbm')
                    
                    return original_train(std_data, lstm_data)
                else:
                    raise
        
        # Apply the patch
        model.train_models = patched_train
        
        # Train models
        print("Training models...")
        model.train_models(std_data, lstm_data)
        
        # Save models
        print("Saving models...")
        model.save_models()
        
        print("Model training and evaluation complete. Results saved to reports/ directory.")
        return 0
    
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
