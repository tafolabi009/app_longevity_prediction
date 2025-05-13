#!/usr/bin/env python3

"""
Train App Longevity Prediction Models

This script provides a more robust wrapper around the model training process,
setting up the visualization environment properly to avoid font issues.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
import json

# Set non-interactive backend before importing plt
matplotlib.use('Agg')  

# First import visualization_utils to setup the environment
from scripts.visualization_utils import setup_visualization_environment
from scripts.model import AppLongevityModel, NumpyEncoder

def main():
    """Run the model training process with proper visualization setup"""
    
    parser = argparse.ArgumentParser(description="Train app longevity prediction models")
    parser.add_argument("--use-lstm", action="store_true", help="Use LSTM in the model")
    parser.add_argument("--disable-advanced", action="store_true", help="Disable advanced models like XGBoost")
    parser.add_argument("--data-file", type=str, default="data/raw/app_data_final.csv", 
                       help="Path to the data file")
    
    args = parser.parse_args()
    
    # Create required directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    # Set up visualization environment properly
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
    try:
        print(f"Loading data from {args.data_file}...")
        df = pd.read_csv(args.data_file)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"Error: {args.data_file} not found. Please run data_collection.py first.")
        return 1
    
    # Initialize and run model
    model = AppLongevityModel(use_lstm=args.use_lstm, use_advanced_models=not args.disable_advanced)
    
    try:
        # Preprocess data
        print("Preprocessing data...")
        std_data, lstm_data = model.preprocess_data(df)
        
        # Determine input shape for LSTM
        input_shape = None
        if lstm_data is not None:
            X_lstm_train = lstm_data[0]
            input_shape = X_lstm_train.shape[1:]
        
        # Build models
        print("Building models...")
        model.build_models(input_shape)
        
        # Train models
        print("Training models...")
        model.train_models(std_data, lstm_data)
        
        # Save models
        print("Saving models...")
        model.save_models()
        
        print("Model training and evaluation complete. Results saved to reports/ directory.")
        return 0
    
    except Exception as e:
        import traceback
        print(f"Error during model training: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
