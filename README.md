# App Longevity Prediction Project

This project predicts app longevity based on various features and metrics, using machine learning models.

## Recent Fixes

The following issues have been addressed:

1. **JSON Serialization Fix**:
   - Added proper NumpyEncoder usage in the `_generate_shap_values` method
   - Implemented JSON monkey-patching to handle NumPy data types properly
   - Created a robust serialization mechanism for consistent JSON output

2. **Visualization Improvements**:
   - Fixed font warnings ("missing from font(s) DejaVu Sans") with platform-specific font handling
   - Added improved layout parameters to avoid "Tight layout not applied" warnings
   - Enhanced visualization quality with better formatting and increased readability

3. **LightGBM Feature Name Error**:
   - Fixed "Do not support special JSON characters in feature name" error
   - Added sanitization of feature names for LightGBM compatibility
   - Implemented a graceful fallback mechanism when LightGBM fails

4. **Error Handling**:
   - Added more descriptive error messages with traceback information
   - Implemented more robust error handling in data processing
   - Created a skip mechanism for models that encounter errors

## Running the Code

### Local Environment

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the model training script:
   ```
   python train_models.py
   ```

3. To skip LightGBM if you encounter errors:
   ```
   python train_models.py --skip-lightgbm
   ```

### Google Colab

If you encounter dependency issues on your local machine, you can run this in Google Colab:

1. Upload the following files to Colab:
   - `scripts/model.py`
   - `scripts/visualization_utils.py`
   - `colab_train.py`
   - `data/raw/app_data_final.csv`

2. Run the notebook code as per the instructions in `run_in_colab.md`

## Model Performance

The current best model is Gradient Boosting with:
- RMSE: 0.0493
- R²: 0.8722

The model was trained on a dataset of 66 apps, with features including:
- App ratings and reviews
- Update frequency
- User engagement metrics
- Monetization strategy
- Market category

## Next Steps

1. **Expand the Dataset**: Collect more app data to improve model generalization
2. **Feature Engineering**: Add more derived features from existing data
3. **Hyperparameter Tuning**: Optimize model parameters for better performance
4. **Interpretability**: Enhance SHAP visualization for better model understanding

## Overview

The App Longevity Prediction project analyzes app metadata, user engagement metrics, and marketplace positioning to predict how long an app is likely to remain viable in the market. The system collects data from app stores, processes it, and applies machine learning models to generate predictions.

## Features

- **Predict any app's longevity** by simply providing its name
- **Analyze apps across both iOS and Android** platforms
- **Identify key factors** contributing to an app's success or failure
- **Compare with competitors** in the same category
- **Generate visualizations** for better understanding of app metrics
- **Receive actionable recommendations** to improve app longevity
- **Train multiple ML models** including Random Forest, Gradient Boosting, XGBoost, and more

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/app_longevity_prediction.git
   cd app_longevity_prediction
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Predicting App Longevity

Use the dedicated command-line tool to predict any app's longevity:

```bash
python predict_app.py "Spotify"
```

Additional options:
- `--compare-competitors`: Compare the app with competitors in the same category
- `--no-visualize`: Disable visualization generation
- `--simple`: Generate simplified output without detailed explanations
- `--save-results`: Save analysis results to a JSON file

Examples:
```bash
# Compare Instagram with competitor apps
python predict_app.py "Instagram" --compare-competitors

# Quick analysis of Candy Crush without visualizations
python predict_app.py "Candy Crush" --no-visualize --simple

# Full analysis of TikTok with saved results
python predict_app.py "TikTok" --compare-competitors --save-results
```

### Training the Models

To train or retrain the prediction models with your own data:

```bash
python scripts/model.py
```

Options:
- `--use-lstm`: Include LSTM neural networks in training
- `--disable-advanced`: Disable advanced models like XGBoost and LightGBM
- `--data-file PATH`: Use a custom data file

## Example Output

```
============================================================
📊 APP LONGEVITY PREDICTION RESULTS
============================================================
App Name: Spotify
Platform: iOS

📈 Longevity Score: 0.8753
Category: Excellent
Expected Lifespan: 5+ years
Success Probability: Very High

Interpretation: This app shows strong indicators of long-term success and user retention.

📱 Key App Metrics:
----------------------------------------
• Rating: 4.8
• Downloads: 500000000
• Days Since Release: 3650
• Days Since Last Update: 12
• Positive Sentiment Ratio: 0.89
• In App Purchases: Yes

🔍 Top Contributing Factors:
----------------------------------------
• App rating of 4.8/5
• 89.0% positive sentiment in reviews
• Released 3650 days ago
• Last updated 12 days ago
• Updated every 14 days on average

💡 Recommendations:
----------------------------------------
• User Engagement (Potential engagement improvements): Consider adding features that encourage daily app usage, such as notifications, rewards, or social elements.
• Monetization (Revenue optimization): Review your monetization strategy compared to competitors in your category.

🥇 Market Position Analysis:
----------------------------------------
Market Position: 92th percentile in Music

Top Competitors:
• Apple Music - Rating: 4.7, Downloads: 450000000
• YouTube Music - Rating: 4.5, Downloads: 350000000
• Pandora - Rating: 4.3, Downloads: 200000000

📊 Visualization saved to: reports/Spotify_longevity.png

============================================================
Analysis Date: 2023-06-20
============================================================
```

## How It Works

The app longevity prediction system works in several steps:

1. **Data Collection**: Fetches app data from the iOS App Store and Google Play Store
2. **Feature Engineering**: Processes raw data into meaningful features
3. **Model Training**: Trains multiple machine learning models
4. **Prediction**: Applies the best-performing model to make predictions
5. **Analysis**: Identifies key factors and generates recommendations

## Results and Performance

The system achieves prediction performance with:
- RMSE (Root Mean Square Error): 0.1082
- R² (Coefficient of Determination): 0.9521
- MAE (Mean Absolute Error): 0.0831

Random Forest consistently performs as the best model for this task.

## Future Improvements

- Add a web interface for easier access
- Implement time-series forecasting for future trends
- Expand data collection to include more app stores
- Include user demographic analysis
- Add support for more languages and regions

## License

[MIT License](LICENSE)

## Acknowledgments

- This project uses data from the iOS App Store and Google Play Store
- Built with scikit-learn, pandas, matplotlib, and other open-source libraries
