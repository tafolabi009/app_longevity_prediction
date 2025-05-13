# App Longevity Prediction - Fixes and Improvements

This document outlines the fixes and improvements made to resolve issues in the App Longevity Prediction project.

## Fixed Issues

### 1. JSON Serialization Error

**Problem:** The error `TypeError: Object of type float32 is not JSON serializable` was occurring when trying to save model metrics and SHAP values to JSON files.

**Solution:**
- Added proper use of the `NumpyEncoder` class in the `_generate_shap_values` method
- Created a shap_analysis.json file with properly serialized SHAP values
- Added a comprehensive monkey-patching of the JSON encoder in train_models.py

### 2. Font and Layout Warnings

**Problem:** Warnings like ") missing from font(s) DejaVu Sans" and "Tight layout not applied" were appearing during visualization generation.

**Solution:**
- Created a dedicated visualization_utils.py module with platform-aware font handling
- Improved layout adjustments with proper rect parameters for tight_layout
- Added bbox_inches='tight' to savefig calls to ensure proper figure rendering
- Set up non-interactive Agg backend for Matplotlib in train_models.py

### 3. Increased Dataset Size

**Problem:** The model was training on only 38 apps, which is a small dataset for ML training.

**Solution:**
- Expanded the app ID list to over 150 apps across both platforms
- Added validation of app IDs before processing to filter out invalid ones
- Improved app data collection with better error handling

### 4. Other Improvements

- Created a dedicated train_models.py script for more robust model training
- Added better error handling with traceback outputs
- Enhanced the data collection process to skip invalid app IDs
- Created more informative plots with better labels and layout

## How to Use the Fixes

### For Training Models

Use the new train_models.py script instead of running model.py directly:

```bash
python train_models.py [options]
```

Options:
- `--use-lstm`: Enable LSTM neural networks
- `--disable-advanced`: Disable advanced models (XGBoost, LightGBM)
- `--data-file PATH`: Specify custom data file path

### For Data Collection

The data collection process now validates app IDs before processing:

```bash
python scripts/data_collection.py
```

### For Predicting App Longevity

The predict_app.py script has improved error handling:

```bash
python predict_app.py "App Name" [options]
```

## Technical Details

### JSON Serialization Fix

The NumpyEncoder class handles these NumPy types:
- np.integer, np.int32, np.int64 → int
- np.floating, np.float32, np.float64 → float
- np.ndarray → list
- np.bool_ → bool

### Visualization Improvements

- Platform-specific font selection (Windows, macOS, Linux)
- Better margin handling for feature importance plots
- Improved SHAP value visualization
- Enhanced model comparison charts

### Data Collection

- Added pre-validation of app IDs before processing
- Better error handling for missing apps
- Improved rate limiting

## Results

With these fixes, the model training process now:
- Successfully processes ~66 apps (vs. original 38)
- Properly handles all numeric data types
- Creates clean visualizations without font warnings
- Provides better feature importance analysis 
