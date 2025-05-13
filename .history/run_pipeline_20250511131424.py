#!/usr/bin/env python
"""
App Longevity Prediction Pipeline
=================================
This script runs the complete pipeline:
1. Data collection from multiple sources
2. Feature engineering and preprocessing
3. Model training and evaluation
4. Results visualization and reporting
"""

import os
import argparse
import time
from datetime import datetime

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the App Longevity Prediction pipeline")
    parser.add_argument("--skip-collection", action="store_true", help="Skip data collection step")
    parser.add_argument("--skip-model", action="store_true", help="Skip model training step")
    parser.add_argument("--use-lstm", action="store_true", help="Use LSTM in the model")
    parser.add_argument("--disable-advanced", action="store_true", help="Disable advanced models like XGBoost")
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/interim", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Track timing
    start_time = time.time()
    print(f"=== Starting App Longevity Pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Step 1: Data Collection
    if not args.skip_collection:
        print("\n=== Running Data Collection ===")
        os.system("python scripts/data_collection.py")
    else:
        print("\n=== Skipping Data Collection ===")
    
    # Step 2: Model Training
    if not args.skip_model:
        print("\n=== Running Model Training ===")
        cmd = "python scripts/model.py"
        if args.use_lstm:
            cmd += " --use-lstm"
        if args.disable_advanced:
            cmd += " --disable-advanced"
        os.system(cmd)
    else:
        print("\n=== Skipping Model Training ===")
    
    # Report execution time
    execution_time = time.time() - start_time
    print(f"\n=== Pipeline completed in {execution_time:.2f} seconds ===")
    print(f"=== Results available in reports/ directory ===")

if __name__ == "__main__":
    main() 
