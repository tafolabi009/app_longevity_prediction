import pandas as pd
import numpy as np
import pickle
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
from sklearn.feature_selection import SelectFromModel
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Dropout, Bidirectional, Attention, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
import lightgbm as lgb
import shap  # For model explainability
from joblib import dump, load
import json
import statsmodels.formula.api as smf

# Import visualization utilities
from scripts.visualization_utils import (
    setup_visualization_environment,
    create_feature_importance_plot,
    create_model_comparison_plot,
    create_predictions_plot
)

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

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

class AppLongevityModel:
    """App Longevity and Success Prediction Model"""
    
    def __init__(self, use_lstm=True, use_advanced_models=True):
        self.use_lstm = use_lstm
        self.use_advanced_models = use_advanced_models
        self.models = {}
        self.feature_importances = {}
        self.preprocessing = {}
        self.best_model = None
        self.model_metrics = {}
        
    def preprocess_data(self, df, test_size=0.2, random_state=42):
        """Preprocess data for training and testing"""
        
        print("Preprocessing data...")
        
        # Handle different types of data columns
        self.cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.cat_columns = [col for col in self.cat_columns if col not in ['app_id', 'app_name', 'keywords']]
        
        self.num_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Special handling for keywords and text data
        if 'keywords' in df.columns:
            keywords = df['keywords'].apply(lambda x: x if isinstance(x, list) else [])
            unique_keywords = set(kw for kwlist in keywords for kw in kwlist)
            
            for kw in unique_keywords:
                df[f'keyword_{kw}'] = df['keywords'].apply(lambda x: 1 if kw in x else 0)
        
        # Handle target variables
        target_variable = "rating"
        if target_variable not in df.columns:
            print(f"Warning: Target variable '{target_variable}' not found. Using first available numerical column.")
            target_variable = self.num_columns[0]
        
        X = df.drop(columns=['app_name', 'app_id', 'keywords', target_variable], errors='ignore')
        y = df[target_variable]
        
        # Convert categorical columns to dummies if they exist
        if self.cat_columns:
            X = pd.get_dummies(X, columns=self.cat_columns, drop_first=True)
        
        # Fill missing values
        X = X.fillna(0)  # Simple imputation for demonstration
        
        # Apply scaling
        self.scaler = RobustScaler()  # Using RobustScaler for better handling of outliers
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Save the train and test sets
        pd.DataFrame({'true_values': y_test}).to_csv("data/processed/test_ground_truth.csv", index=False)
        
        # Prepare LSTM data if needed
        if self.use_lstm:
            # Reshape for LSTM (assuming time-based features exist)
            sequence_length = 3  # Can be adjusted
            try:
                X_lstm_train, y_lstm_train = self._prepare_lstm_data(X_train, y_train, sequence_length)
                X_lstm_test, y_lstm_test = self._prepare_lstm_data(X_test, y_test, sequence_length)
                
                return (X_train, y_train, X_test, y_test), (X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test)
            except Exception as e:
                print(f"Error preparing LSTM data: {e}. Falling back to standard models.")
                self.use_lstm = False
        
        return (X_train, y_train, X_test, y_test), None
    
    def _prepare_lstm_data(self, X, y, sequence_length):
        """Format data for LSTM input"""
        X_seq = []
        y_seq = []
        
        for i in range(len(X) - sequence_length + 1):
            X_seq.append(X.iloc[i:i+sequence_length].values)
            y_seq.append(y.iloc[i+sequence_length-1])
        
        return np.array(X_seq), np.array(y_seq)
        
    def build_models(self, input_shape=None):
        """Build various prediction models"""
        
        print("Building models...")
        
        # Traditional ML models
        self.models['rf'] = RandomForestRegressor(n_estimators=100, random_state=42)
        self.models['gb'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.models['elasticnet'] = ElasticNet(random_state=42, alpha=0.1, l1_ratio=0.5)
        self.models['ridge'] = Ridge(alpha=1.0, random_state=42)
        
        # Advanced models if enabled
        if self.use_advanced_models:
            self.models['xgb'] = xgb.XGBRegressor(
                n_estimators=100, 
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            self.models['lgb'] = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        
        # Deep learning model if LSTM is enabled
        if self.use_lstm and input_shape is not None:
            # Create a more sophisticated neural network architecture
            input_layer = Input(shape=input_shape)
            
            # Bidirectional LSTM layers
            lstm_layer1 = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
            dropout1 = Dropout(0.2)(lstm_layer1)
            
            lstm_layer2 = Bidirectional(LSTM(32, return_sequences=True))(dropout1)
            dropout2 = Dropout(0.2)(lstm_layer2)
            
            # Attention mechanism
            attention_layer = Attention()([dropout2, dropout2])
            flattened = tf.keras.layers.Flatten()(attention_layer)
            
            # Dense layers
            dense1 = Dense(100, activation='relu')(flattened)
            dropout3 = Dropout(0.2)(dense1)
            dense2 = Dense(50, activation='relu')(dropout3)
            output = Dense(1, activation='linear')(dense2)
            
            nn_model = Model(inputs=input_layer, outputs=output)
            nn_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mean_squared_error',
                metrics=['mae', 'mse']
            )
            
            self.models['nn'] = nn_model
        
        # Create ensemble model from the base models
        estimators = [(name, model) for name, model in self.models.items() 
                     if name not in ['nn', 'ensemble']]
        
        if estimators:
            self.models['ensemble'] = VotingRegressor(
                estimators=estimators,
                weights=[1.0] * len(estimators)
            )
            
        return self.models
    
    def train_models(self, std_data, lstm_data=None):
        """Train all models on the provided data"""
        
        print("Training models...")
        X_train, y_train, X_test, y_test = std_data
        
        # Sanitize column names for XGBoost (remove [, ], < characters)
        sanitized_columns = {}
        for col in X_train.columns:
            if any(char in col for char in ['[', ']', '<', '>', ':', '"', '{', '}', '/', '\\', '|', '?', '*']):
                sanitized_col = col.replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') \
                                  .replace(':', '_').replace('"', '_').replace('{', '_').replace('}', '_') \
                                  .replace('/', '_').replace('\\', '_').replace('|', '_').replace('?', '_') \
                                  .replace('*', '_')
                sanitized_columns[col] = sanitized_col
        
        # Apply sanitized column names if needed
        if sanitized_columns:
            X_train_sanitized = X_train.copy()
            X_test_sanitized = X_test.copy()
            X_train_sanitized.columns = [sanitized_columns.get(col, col) for col in X_train.columns]
            X_test_sanitized.columns = [sanitized_columns.get(col, col) for col in X_test.columns]
        else:
            X_train_sanitized = X_train
            X_test_sanitized = X_test
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            if name == 'nn':
                if lstm_data is not None:
                    X_lstm_train, y_lstm_train, X_lstm_test, y_lstm_test = lstm_data
                    
                    # Callbacks for better training
                    callbacks = [
                        EarlyStopping(patience=10, restore_best_weights=True),
                        ReduceLROnPlateau(factor=0.5, patience=5),
                        ModelCheckpoint('models/best_nn_model.h5', save_best_only=True)
                    ]
                    
                    history = model.fit(
                        X_lstm_train, y_lstm_train,
                        validation_data=(X_lstm_test, y_lstm_test),
                        epochs=50,
                        batch_size=32,
                        callbacks=callbacks,
                        verbose=1
                    )
                    
                    # Plot training history
                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 2, 1)
                    plt.plot(history.history['loss'], label='Train')
                    plt.plot(history.history['val_loss'], label='Validation')
                    plt.title('Loss')
                    plt.legend()
                    
                    plt.subplot(1, 2, 2)
                    plt.plot(history.history['mae'], label='Train')
                    plt.plot(history.history['val_mae'], label='Validation')
                    plt.title('Mean Absolute Error')
                    plt.legend()
                    
                    plt.tight_layout()
                    plt.savefig('reports/nn_training_history.png')
                    
                    # Evaluate on test set
                    y_pred = model.predict(X_lstm_test)
                    self._evaluate_and_store_metrics(name, y_lstm_test, y_pred.flatten())
                    
                else:
                    print("Skipping NN model training - LSTM data not available")
            else:
                # Train standard ML models
                if name == 'lgb':
                    # Debug: print column names for LightGBM
                    print(f"LightGBM feature names: {X_train_sanitized.columns.tolist()}")
                    # Additional sanitization for LightGBM
                    try:
                        model.fit(X_train_sanitized, y_train)
                        y_pred = model.predict(X_test_sanitized)
                    except Exception as e:
                        print(f"LightGBM error: {str(e)}")
                        print("Skipping LightGBM model due to feature name compatibility issues")
                        continue
                elif name == 'xgb':
                if name in ['xgb', 'lgb']:
                    # Use sanitized column names for XGBoost and LightGBM
                    model.fit(X_train_sanitized, y_train)
                    y_pred = model.predict(X_test_sanitized)
                else:
                    # For other models, use original column names
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Evaluate and store metrics
                self._evaluate_and_store_metrics(name, y_test, y_pred)
                
                # Extract feature importance for tree-based models
                if name in ['rf', 'gb']:
                    self._extract_feature_importance(name, model, X_train.columns)
                elif name in ['xgb', 'lgb']:
                    self._extract_feature_importance(name, model, X_train_sanitized.columns)
        
        # Find the best model
        self._find_best_model()
        
        # Generate SHAP values for the best model
        if self.best_model in ['rf', 'gb']:
            self._generate_shap_values(self.models[self.best_model], X_test)
        elif self.best_model in ['xgb', 'lgb']:
            self._generate_shap_values(self.models[self.best_model], X_test_sanitized)
        
        return self.model_metrics
    
    def _evaluate_and_store_metrics(self, model_name, y_true, y_pred):
        """Evaluate model and store metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        median_ae = median_absolute_error(y_true, y_pred)
        
        self.model_metrics[model_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'median_ae': median_ae
        }
        
        print(f"{model_name} Metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        
        # Save predictions
        pd.DataFrame({
            'true': y_true,
            'predicted': y_pred
        }).to_csv(f"reports/{model_name}_predictions.csv", index=False)
        
        # Create improved prediction plot
        create_predictions_plot(model_name, y_true, y_pred)
    
    def _extract_feature_importance(self, model_name, model, feature_names):
        """Extract and visualize feature importance"""
        
        if model_name == 'rf' or model_name == 'gb':
            importance = model.feature_importances_
        elif model_name == 'xgb':
            importance = model.feature_importances_
        elif model_name == 'lgb':
            importance = model.feature_importances_
        else:
            return
            
        self.feature_importances[model_name] = dict(zip(feature_names, importance))
        
        # Use improved feature importance plot
        create_feature_importance_plot(model_name, feature_names, importance)
        
        # Save feature importances to file
        with open(f'reports/{model_name}_feature_importance.json', 'w') as f:
            json.dump(self.feature_importances[model_name], f, indent=2, cls=NumpyEncoder)
    
    def _find_best_model(self):
        """Find the best performing model based on RMSE"""
        
        if not self.model_metrics:
            print("No model metrics available")
            return None
            
        # Select best model based on RMSE
        self.best_model = min(self.model_metrics.items(), 
                             key=lambda x: x[1]['rmse'])[0]
        
        print(f"Best model: {self.best_model} with RMSE: {self.model_metrics[self.best_model]['rmse']:.4f}")
        
        # Save comparison of all models
        metrics_df = pd.DataFrame({
            model: {
                'RMSE': metrics['rmse'],
                'MAE': metrics['mae'],
                'R²': metrics['r2'],
                'MedianAE': metrics['median_ae']
            }
            for model, metrics in self.model_metrics.items()
        }).T
        
        metrics_df.to_csv('reports/model_comparison.csv')
        
        # Use improved model comparison visualization
        create_model_comparison_plot(self.model_metrics)
        
        return self.best_model
    
    def _generate_shap_values(self, model, X_sample):
        """Generate SHAP values for model explainability"""
        
        try:
            # Take a small sample to speed up SHAP computation
            X_sample_small = X_sample.iloc[:100] if len(X_sample) > 100 else X_sample
            
            # Create explainer
            explainer = shap.Explainer(model, X_sample_small)
            shap_values = explainer(X_sample_small)
            
            # Save SHAP values for later analysis - using NumpyEncoder for proper JSON serialization
            try:
                shap_data = {
                    'model_name': self.best_model,
                    'feature_names': list(X_sample_small.columns),
                    'mean_abs_shap': np.abs(shap_values.values).mean(0).tolist(),
                    'top_features': list(X_sample_small.columns[np.argsort(np.abs(shap_values.values).mean(0))[-10:]]),
                }
                
                with open('reports/shap_analysis.json', 'w') as f:
                    json.dump(shap_data, f, indent=2, cls=NumpyEncoder)
            except Exception as json_err:
                print(f"Error saving SHAP values to JSON: {str(json_err)}")
            
            # Plot SHAP summary with improved layout
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample_small, show=False, plot_size=(12, 8))
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust rect to avoid tight layout warnings
            plt.savefig('reports/shap_summary.png', bbox_inches='tight')
            plt.close()
            
            # Plot SHAP bar summary with improved layout
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample_small, plot_type="bar", show=False, plot_size=(12, 8))
            plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Adjust rect to avoid tight layout warnings
            plt.savefig('reports/shap_bar_summary.png', bbox_inches='tight')
            plt.close()
            
            print("SHAP analysis completed and saved to reports/")
        except Exception as e:
            print(f"Error generating SHAP values: {e}")
            import traceback
            traceback.print_exc()
    
    def save_models(self):
        """Save all models to disk"""
        
        print("Saving models...")
        
        for name, model in self.models.items():
            if name == 'nn':
                model.save(f'models/{name}_model.h5')
            else:
                dump(model, f'models/{name}_model.joblib')
                
        # Save the scaler
        dump(self.scaler, 'models/scaler.joblib')
        
        # Save metrics
        with open('models/metrics.json', 'w') as f:
            # Use custom encoder to handle NumPy data types
            json.dump(self.model_metrics, f, indent=2, cls=NumpyEncoder)
        
        print(f"All models saved to models/ directory. Best model: {self.best_model}")
        
    def predict(self, new_data, model_name=None):
        """Make predictions with the specified or best model"""
        
        if model_name is None:
            model_name = self.best_model
            
        if model_name not in self.models:
            print(f"Model {model_name} not found. Using {self.best_model} instead.")
            model_name = self.best_model
            
        model = self.models[model_name]
            
        # Preprocess new data
        if isinstance(new_data, pd.DataFrame):
            # Handle categorical columns
            if self.cat_columns:
                new_data = pd.get_dummies(new_data, columns=self.cat_columns, drop_first=True)
                
            # Ensure all columns match training data
            missing_cols = set(X_train.columns) - set(new_data.columns)
            for col in missing_cols:
                new_data[col] = 0
                
            # Reorder columns to match training data
            new_data = new_data[X_train.columns]
            
            # Scale the data
            new_data_scaled = self.scaler.transform(new_data)
            
            if model_name == 'nn' and self.use_lstm:
                # Need to reshape for LSTM
                sequence_length = 3  # Same as in training
                # Reshape logic would go here
                # ...
                return model.predict(new_data_lstm)
            else:
                return model.predict(new_data_scaled)
        else:
            print("Error: New data must be a pandas DataFrame")
            return None

    def predict_app_longevity(self, app_name, compare_with_competitors=False, visualize=True, explain=True):
        """
        Predict longevity for a new app by name with enhanced features:
        - Improved app discovery across platforms
        - Competitor analysis
        - Detailed explanations and visualizations
        - Longevity factors breakdown
        
        Args:
            app_name (str): Name of the app to predict longevity for
            compare_with_competitors (bool): Whether to compare with similar apps
            visualize (bool): Whether to generate visualizations
            explain (bool): Whether to provide detailed explanations
            
        Returns:
            dict: Prediction results with detailed analysis
        """
        try:
            # Import data collection module to fetch app data
            import sys
            sys.path.append('./scripts')
            from data_collection import fetch_app_store_data, fetch_play_store_data, calculate_feature_engineering
            
            print(f"Analyzing app: {app_name}")
            
            # Try to find the app in both stores with improved search
            ios_data = None
            android_data = None
            app_store_id = None
            play_store_id = None
            
            # For iOS, use iTunes search API with better error handling
            try:
                import requests
                # Encode app name for URL
                encoded_app_name = requests.utils.quote(app_name)
                search_url = f"https://itunes.apple.com/search?term={encoded_app_name}&entity=software&limit=5"
                
                response = requests.get(search_url)
                search_data = response.json()
                
                if search_data['resultCount'] > 0:
                    # Sort results by relevance (name similarity)
                    from difflib import SequenceMatcher
                    
                    def similarity(a, b):
                        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
                    
                    # Find the most relevant app
                    best_match = max(search_data['results'], 
                                    key=lambda x: similarity(x['trackName'], app_name))
                    
                    app_store_id = str(best_match['trackId'])
                    print(f"Found iOS app: {best_match['trackName']} (ID: {app_store_id})")
                    
                    ios_data = fetch_app_store_data(app_store_id)
                    if ios_data:
                        ios_data = calculate_feature_engineering(ios_data)
                else:
                    print(f"No iOS app found for '{app_name}'")
            except Exception as e:
                print(f"Error searching iOS app: {str(e)}")
            
            # For Android, use improved search approaches
            try:
                import requests
                from bs4 import BeautifulSoup
                
                # Try Google Play Store search
                search_term = app_name.replace(' ', '+')
                search_url = f"https://play.google.com/store/search?q={search_term}&c=apps"
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                try:
                    response = requests.get(search_url, headers=headers)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        # Try to extract package ID from the first result
                        app_links = soup.select('a[href^="/store/apps/details?id="]')
                        
                        if app_links:
                            # Extract package ID from the first result
                            href = app_links[0]['href']
                            import re
                            package_match = re.search(r'id=([^&]+)', href)
                            if package_match:
                                play_store_id = package_match.group(1)
                                print(f"Found Android app with package: {play_store_id}")
                                
                                # Fetch data
                                android_data = fetch_play_store_data(play_store_id)
                                if android_data:
                                    android_data = calculate_feature_engineering(android_data)
                except Exception as inner_e:
                    print(f"Error in Android web search: {str(inner_e)}")
                    
                # If web search failed, try common package patterns as fallback
                if not android_data:
                    package_guesses = [
                        f"com.{app_name.lower().replace(' ', '')}",
                        f"com.{app_name.lower().replace(' ', '.')}",
                        f"com.{app_name.lower().replace(' ', '_')}",
                        f"io.{app_name.lower().replace(' ', '')}",
                        f"org.{app_name.lower().replace(' ', '')}"
                    ]
                    
                    for package in package_guesses:
                        try:
                            test_data = fetch_play_store_data(package)
                            if test_data:
                                print(f"Found Android app with package: {package}")
                                play_store_id = package
                                android_data = calculate_feature_engineering(test_data)
                                break
                        except Exception:
                            continue
            except Exception as e:
                print(f"Error searching Android app: {str(e)}")
            
            # Prepare data for prediction
            platform = None
            app_data = None
            
            # Prefer platform with better data quality if both are available
            if ios_data and android_data:
                # Choose the platform with more features available
                ios_nulls = sum(1 for v in ios_data.values() if v is None)
                android_nulls = sum(1 for v in android_data.values() if v is None)
                
                if ios_nulls <= android_nulls:
                    app_data = ios_data
                    platform = "iOS"
                else:
                    app_data = android_data
                    platform = "Android"
                    
                print(f"Found app on both platforms, using {platform} data for prediction")
            elif ios_data:
                app_data = ios_data
                platform = "iOS"
            elif android_data:
                app_data = android_data
                platform = "Android"
            
            if not app_data:
                print(f"Could not find sufficient data for '{app_name}' on either platform")
                return {
                    'app_name': app_name,
                    'error': "Could not find app data. Please check the app name or try a more popular app."
                }
            
            # Convert to DataFrame for prediction
            app_df = pd.DataFrame([app_data])
            
            # Preprocess for model
            # Remove columns not used in training
            exclude_cols = ['app_name', 'app_id', 'keywords', 'reviews']
            for col in exclude_cols:
                if col in app_df.columns:
                    app_df = app_df.drop(columns=[col])
            
            # Load the best model
            if not hasattr(self, 'best_model') or not self.best_model:
                # Try to load the model metrics to determine best model
                try:
                    with open('models/metrics.json', 'r') as f:
                        metrics = json.load(f)
                    self.best_model = min(metrics.items(), key=lambda x: x[1]['rmse'])[0]
                except:
                    self.best_model = 'rf'  # Default to random forest if can't determine
            
            # Load feature importances
            feature_importances = {}
            try:
                with open(f'reports/{self.best_model}_feature_importance.json', 'r') as f:
                    feature_importances = json.load(f)
            except:
                print("Feature importance data not available")
            
            # Make prediction
            prediction = self.predict(app_df, self.best_model)
            predicted_value = float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction)
            
            # Gather key metrics for the app
            key_metrics = {
                'rating': app_data.get('rating', 'Unknown'),
                'downloads': app_data.get('downloads', 'Unknown'),
                'price': app_data.get('price', 'Unknown'),
                'size_mb': app_data.get('size_mb', 'Unknown'),
                'days_since_last_update': app_data.get('days_since_last_update', 'Unknown'),
                'days_since_release': app_data.get('days_since_release', 'Unknown'),
                'positive_sentiment_ratio': app_data.get('positive_sentiment_ratio', 'Unknown'),
                'in_app_purchases': app_data.get('has_in_app_purchases', False),
                'total_ratings': app_data.get('total_ratings', 'Unknown'),
            }
            
            # Build results dictionary
            results = {
                'app_name': app_name,
                'platform': platform,
                'store_id': app_store_id if platform == "iOS" else play_store_id,
                'predicted_longevity': predicted_value,
                'longevity_interpretation': self._interpret_longevity_score(predicted_value),
                'key_metrics': key_metrics,
                'date_analyzed': pd.Timestamp.now().strftime('%Y-%m-%d'),
            }
            
            # Add contributing factors if feature importances are available
            if feature_importances and explain:
                # Identify top contributing factors
                contributing_factors = []
                
                # Use non-null features from the app data
                available_features = {k: v for k, v in app_data.items() 
                                     if k in feature_importances and v is not None}
                
                # Sort by feature importance
                sorted_features = sorted(available_features.items(), 
                                        key=lambda x: feature_importances.get(x[0], 0), 
                                        reverse=True)
                
                # Take top 5 contributors
                for feature, value in sorted_features[:5]:
                    impact = "positive" if feature_importances[feature] > 0 else "negative"
                    contributing_factors.append({
                        'feature': feature,
                        'value': value,
                        'importance': feature_importances[feature],
                        'impact': impact,
                        'description': self._get_feature_description(feature, value)
                    })
                
                results['contributing_factors'] = contributing_factors
            
            # Add competitor analysis if requested
            if compare_with_competitors:
                try:
                    # Find similar apps data from our dataset
                    df = pd.read_csv('data/raw/app_data_final.csv')
                    
                    # Filter by platform and category if available
                    category = app_data.get('category')
                    if platform and category:
                        similar_apps = df[(df['platform'] == platform) & (df['category'] == category)]
                    elif platform:
                        similar_apps = df[df['platform'] == platform]
                    else:
                        similar_apps = df
                    
                    # Take top 5 similar apps by ratings or downloads
                    if len(similar_apps) > 0:
                        if 'rating' in similar_apps.columns:
                            top_competitors = similar_apps.sort_values('rating', ascending=False).head(5)
                        else:
                            top_competitors = similar_apps.sort_values('downloads', ascending=False).head(5)
                        
                        competitors = []
                        for _, app in top_competitors.iterrows():
                            competitors.append({
                                'name': app.get('app_name', 'Unknown'),
                                'rating': app.get('rating', 'Unknown'),
                                'downloads': app.get('downloads', 'Unknown'),
                                'known_longevity': app.get('longevity', 'Unknown')
                            })
                        
                        results['competitor_analysis'] = {
                            'category': category,
                            'competitors': competitors,
                            'market_position_percentile': self._calculate_market_position(app_data, similar_apps)
                        }
                except Exception as e:
                    print(f"Error performing competitor analysis: {str(e)}")
            
            # Generate visualizations if requested
            if visualize:
                try:
                    self._generate_longevity_visualization(app_name, predicted_value, key_metrics)
                    results['visualization_path'] = f'reports/{app_name.replace(" ", "_")}_longevity.png'
                except Exception as e:
                    print(f"Error generating visualization: {str(e)}")
            
            # Add recommendations based on the analysis
            if explain:
                results['recommendations'] = self._generate_recommendations(app_data, feature_importances)
            
            return results
        except Exception as e:
            print(f"Error predicting app longevity: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'app_name': app_name,
                'error': f"Error analyzing app: {str(e)}"
            }
            
    def _interpret_longevity_score(self, score):
        """Provide interpretation of the longevity score"""
        if score >= 0.8:
            return {
                'category': 'Excellent',
                'description': 'This app shows strong indicators of long-term success and user retention.',
                'expected_lifespan': '5+ years',
                'success_probability': 'Very High'
            }
        elif score >= 0.6:
            return {
                'category': 'Good',
                'description': 'This app has solid fundamentals and is likely to remain viable for years.',
                'expected_lifespan': '3-5 years',
                'success_probability': 'High'
            }
        elif score >= 0.4:
            return {
                'category': 'Average',
                'description': 'This app has moderate longevity indicators, typical of the average app.',
                'expected_lifespan': '1-3 years',
                'success_probability': 'Medium'
            }
        elif score >= 0.2:
            return {
                'category': 'Below Average',
                'description': 'This app shows some concerning metrics that may limit its lifespan.',
                'expected_lifespan': '6 months - 1 year',
                'success_probability': 'Low'
            }
        else:
            return {
                'category': 'Poor',
                'description': 'This app shows significant risk factors that suggest a short lifespan.',
                'expected_lifespan': 'Less than 6 months',
                'success_probability': 'Very Low'
            }
    
    def _get_feature_description(self, feature, value):
        """Get human-readable description of a feature's impact"""
        descriptions = {
            'rating': f"App rating of {value}/5",
            'days_since_last_update': f"Last updated {value} days ago",
            'days_since_release': f"Released {value} days ago" if value else "Release date unknown",
            'downloads': f"Approximately {value} downloads",
            'size_mb': f"App size of {value} MB",
            'number_of_reviews': f"{value} user reviews",
            'positive_sentiment_ratio': f"{value*100:.1f}% positive sentiment in reviews" if value else "Sentiment unknown",
            'update_frequency': f"Updated every {value} days on average" if value else "Update frequency unknown",
            'has_in_app_purchases': "Offers in-app purchases" if value else "No in-app purchases",
            'price': f"Priced at ${value}" if value else "Free app",
            'content_rating': f"Content rated for {value}",
            'total_ratings': f"{value} total ratings"
        }
        
        return descriptions.get(feature, f"{feature}: {value}")
    
    def _calculate_market_position(self, app_data, similar_apps):
        """Calculate percentile position against similar apps"""
        if 'rating' in similar_apps.columns and 'rating' in app_data:
            similar_ratings = similar_apps['rating'].dropna()
            if len(similar_ratings) > 0:
                return int(sum(similar_ratings <= app_data['rating']) / len(similar_ratings) * 100)
        
        if 'downloads' in similar_apps.columns and 'downloads' in app_data:
            similar_downloads = similar_apps['downloads'].dropna()
            if len(similar_downloads) > 0:
                return int(sum(similar_downloads <= app_data['downloads']) / len(similar_downloads) * 100)
        
        return None
    
    def _generate_longevity_visualization(self, app_name, prediction, metrics):
        """Generate visualization of the longevity prediction and key metrics"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 8))
            
            # Set style
            sns.set_style("whitegrid")
            
            # Main longevity gauge
            plt.subplot(2, 2, 1)
            gauge_colors = ['#FF4136', '#FF851B', '#FFDC00', '#2ECC40', '#0074D9']
            plt.pie(
                [0.2, 0.2, 0.2, 0.2, 0.2],
                colors=gauge_colors,
                startangle=90,
                counterclock=False,
                wedgeprops={'width': 0.3, 'edgecolor': 'w', 'linewidth': 2}
            )
            plt.title(f"Longevity Score: {prediction:.2f}", fontsize=14, fontweight='bold')
            
            # Arrow indicating the score
            angle = (prediction * 180) - 90
            arrow_length = 0.8
            plt.arrow(
                0, 0, 
                arrow_length * np.cos(np.radians(angle)), 
                arrow_length * np.sin(np.radians(angle)),
                head_width=0.1, head_length=0.1, fc='black', ec='black'
            )
            plt.text(0, -1.2, "Poor", ha='center', fontsize=12)
            plt.text(1.2, 0, "Excellent", ha='center', fontsize=12)
            plt.axis('equal')
            
            # Key metrics bar chart
            plt.subplot(2, 2, 2)
            metrics_to_plot = {k: v for k, v in metrics.items() 
                              if k in ['rating', 'positive_sentiment_ratio'] and v not in [None, 'Unknown']}
            
            if metrics_to_plot:
                keys = list(metrics_to_plot.keys())
                values = [float(metrics_to_plot[k]) if isinstance(metrics_to_plot[k], (int, float)) else 0 for k in keys]
                
                # Normalize values for comparison
                max_values = {
                    'rating': 5.0,
                    'positive_sentiment_ratio': 1.0,
                }
                
                normalized_values = [values[i] / max_values.get(keys[i], 1.0) for i in range(len(keys))]
                
                colors = ['#0074D9' if v > 0.6 else '#FF851B' if v > 0.3 else '#FF4136' for v in normalized_values]
                
                bars = plt.bar(keys, normalized_values, color=colors)
                plt.ylim(0, 1.0)
                plt.title("Key Quality Metrics (Normalized)", fontsize=14)
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    plt.text(
                        bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.05,
                        f"{values[i]:.2f}",
                        ha='center'
                    )
            else:
                plt.text(0.5, 0.5, "Insufficient metrics data", ha='center', va='center')
                plt.axis('off')
            
            # App info section
            plt.subplot(2, 2, 3)
            plt.axis('off')
            
            info_text = f"App: {app_name}\n"
            info_text += f"Platform: {metrics.get('platform', 'Unknown')}\n"
            
            # Add key date metrics if available
            if metrics.get('days_since_release') not in [None, 'Unknown']:
                release_date = pd.Timestamp.now() - pd.Timedelta(days=int(metrics['days_since_release']))
                info_text += f"Released: {release_date.strftime('%Y-%m-%d')}\n"
                
            if metrics.get('days_since_last_update') not in [None, 'Unknown']:
                last_update = pd.Timestamp.now() - pd.Timedelta(days=int(metrics['days_since_last_update']))
                info_text += f"Last Updated: {last_update.strftime('%Y-%m-%d')}\n"
                
            # Add downloads if available
            if metrics.get('downloads') not in [None, 'Unknown']:
                info_text += f"Downloads: {metrics['downloads']}\n"
                
            # Add price information
            price_info = f"${metrics['price']}" if metrics.get('price') not in [None, 'Unknown', 0] else "Free"
            info_text += f"Price: {price_info}\n"
            
            # Add in-app purchase info
            iap = "Yes" if metrics.get('in_app_purchases') else "No"
            info_text += f"In-App Purchases: {iap}"
            
            plt.text(0.1, 0.9, info_text, va='top', fontsize=12, linespacing=1.5)
            
            # Recommendations section
            plt.subplot(2, 2, 4)
            plt.axis('off')
            
            if prediction >= 0.7:
                advice = "• Strong fundamentals - focus on retention\n• Consider expansion to new platforms\n• Maintain regular update schedule"
            elif prediction >= 0.4:
                advice = "• Improve user ratings and reviews\n• Consider more frequent updates\n• Enhance engagement features\n• Optimize monetization strategy"
            else:
                advice = "• Address negative reviews urgently\n• Significant updates needed\n• Reconsider core app value proposition\n• Improve user experience and retention"
            
            plt.text(0.1, 0.9, "Recommendations:", va='top', fontsize=14, fontweight='bold')
            plt.text(0.1, 0.8, advice, va='top', fontsize=12, linespacing=1.5)
            
            # Main title
            plt.suptitle(f"App Longevity Analysis: {app_name}", fontsize=16, fontweight='bold', y=0.98)
            
            # Add timestamp
            plt.figtext(0.02, 0.02, f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}", fontsize=8)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            
            # Save the visualization
            safe_name = app_name.replace(" ", "_").replace("/", "_")
            plt.savefig(f'reports/{safe_name}_longevity.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
    
    def _generate_recommendations(self, app_data, feature_importances):
        """Generate specific recommendations based on app data"""
        recommendations = []
        
        # Extract key metrics
        rating = app_data.get('rating')
        days_since_update = app_data.get('days_since_last_update')
        update_frequency = app_data.get('update_frequency')
        positive_sentiment = app_data.get('positive_sentiment_ratio')
        
        # Rating-based recommendations
        if rating is not None:
            if rating < 3.5:
                recommendations.append({
                    'area': 'User Satisfaction',
                    'issue': 'Low app rating',
                    'recommendation': 'Address common complaints in reviews and consider a major update to improve user experience.'
                })
            elif rating < 4.0:
                recommendations.append({
                    'area': 'User Satisfaction',
                    'issue': 'Average app rating',
                    'recommendation': 'Focus on improving specific features mentioned in user reviews to increase ratings.'
                })
        
        # Update frequency recommendations
        if days_since_update is not None and days_since_update > 90:
            recommendations.append({
                'area': 'App Maintenance',
                'issue': 'Infrequent updates',
                'recommendation': 'Establish a regular update schedule to fix bugs and add new features.'
            })
        
        if update_frequency is not None and update_frequency > 60:
            recommendations.append({
                'area': 'Development Cycle',
                'issue': 'Long update cycle',
                'recommendation': 'Consider more frequent, smaller updates to maintain user engagement.'
            })
        
        # Sentiment-based recommendations
        if positive_sentiment is not None and positive_sentiment < 0.6:
            recommendations.append({
                'area': 'User Sentiment',
                'issue': 'Negative user sentiment',
                'recommendation': 'Analyze user reviews to identify pain points and prioritize addressing them.'
            })
        
        # Generic recommendations if we don't have enough data
        if len(recommendations) < 2:
            recommendations.append({
                'area': 'User Engagement',
                'issue': 'Potential engagement improvements',
                'recommendation': 'Consider adding features that encourage daily app usage, such as notifications, rewards, or social elements.'
            })
            
            recommendations.append({
                'area': 'Monetization',
                'issue': 'Revenue optimization',
                'recommendation': 'Review your monetization strategy compared to competitors in your category.'
            })
        
        return recommendations

def main():
    # Handle Jupyter/Colab environment
    import sys
    
    # Handle Colab/Jupyter environment by modifying sys.argv
    # This removes Jupyter-specific arguments like -f kernel files
    if any(arg.startswith('-f') for arg in sys.argv):
        # We're likely in Jupyter/Colab - filter out those arguments
        filtered_args = [arg for arg in sys.argv if not arg.startswith('-f') and not arg.endswith('.json')]
        sys.argv = filtered_args
    
    parser = argparse.ArgumentParser(description="Train and evaluate app longevity prediction models")
    parser.add_argument("--use-lstm", action="store_true", help="Use LSTM in the model")
    parser.add_argument("--disable-advanced", action="store_true", help="Disable advanced models like XGBoost")
    parser.add_argument("--data-file", type=str, default="data/raw/app_data_final.csv", 
                       help="Path to the data file")
    parser.add_argument("--predict-app", type=str, help="Predict longevity for an app by name")
    parser.add_argument("--compare-competitors", action="store_true", 
                       help="Compare with competitor apps when predicting")
    parser.add_argument("--no-visualize", action="store_true",
                       help="Disable visualization generation")
    parser.add_argument("--simple", action="store_true",
                       help="Simple output without detailed explanations")
                       
    # Ignore unknown arguments to handle Jupyter/Colab environment
    args, unknown = parser.parse_known_args()
    
    # Setup visualization environment
    setup_visualization_environment()
    
    # Check if we should predict for a specific app
    if args.predict_app:
        try:
            # Create required directories
            os.makedirs("reports", exist_ok=True)
            os.makedirs("models", exist_ok=True)
            
            # Try to load existing model
            loaded_model = AppLongevityModel()
            
            # Try to load the best model
            model_files = os.listdir('models') if os.path.exists('models') else []
            if not model_files or not any(f.endswith('.joblib') for f in model_files):
                print("No trained models found. Training new models first...")
                # Load data and train models
                df = pd.read_csv(args.data_file)
                std_data, lstm_data = loaded_model.preprocess_data(df)
                loaded_model.build_models()
                loaded_model.train_models(std_data, lstm_data)
                loaded_model.save_models()
            else:
                # Load the best model
                try:
                    with open('models/metrics.json', 'r') as f:
                        metrics = json.load(f)
                    best_model_name = min(metrics.items(), key=lambda x: x[1]['rmse'])[0]
                    print(f"Loading best model: {best_model_name}")
                    
                    if best_model_name == 'nn':
                        model = load_model(f'models/{best_model_name}_model.h5')
                    else:
                        model = load(f'models/{best_model_name}_model.joblib')
                    
                    loaded_model.best_model = best_model_name
                    loaded_model.models = {best_model_name: model}
                    loaded_model.scaler = load('models/scaler.joblib')
                except Exception as e:
                    print(f"Error loading models: {str(e)}. Training new models...")
                    # Load data and train models as fallback
                    df = pd.read_csv(args.data_file)
                    std_data, lstm_data = loaded_model.preprocess_data(df)
                    loaded_model.build_models()
                    loaded_model.train_models(std_data, lstm_data)
                    loaded_model.save_models()
                
            # Get prediction parameters
            visualize = not args.no_visualize
            explain = not args.simple
            
            # Predict for the app
            print(f"\n{'-'*60}")
            print(f"📱 Analyzing app: {args.predict_app}")
            print(f"{'-'*60}")
            
            results = loaded_model.predict_app_longevity(
                args.predict_app,
                compare_with_competitors=args.compare_competitors,
                visualize=visualize,
                explain=explain
            )
            
            if results:
                if 'error' in results:
                    print(f"\n⚠️ Error: {results['error']}")
                    return
                
                # Display results in a nice format
                print(f"\n{'='*60}")
                print(f"📊 APP LONGEVITY PREDICTION RESULTS")
                print(f"{'='*60}")
                print(f"App Name: {results['app_name']}")
                print(f"Platform: {results['platform']}")
                
                # Longevity score with interpretation
                longevity = results['predicted_longevity']
                interpretation = results['longevity_interpretation']
                
                print(f"\n📈 Longevity Score: {longevity:.4f}")
                print(f"Category: {interpretation['category']}")
                print(f"Expected Lifespan: {interpretation['expected_lifespan']}")
                print(f"Success Probability: {interpretation['success_probability']}")
                print(f"\nInterpretation: {interpretation['description']}")
                
                # Key metrics section
                print(f"\n📱 Key App Metrics:")
                print(f"{'-'*40}")
                for feature, value in results['key_metrics'].items():
                    if value not in [None, 'Unknown']:
                        print(f"• {feature.replace('_', ' ').title()}: {value}")
                
                # Contributing factors
                if 'contributing_factors' in results:
                    print(f"\n🔍 Top Contributing Factors:")
                    print(f"{'-'*40}")
                    for factor in results['contributing_factors']:
                        print(f"• {factor['description']}")
                
                # Recommendations
                if 'recommendations' in results:
                    print(f"\n💡 Recommendations:")
                    print(f"{'-'*40}")
                    for rec in results['recommendations']:
                        print(f"• {rec['area']} ({rec['issue']}): {rec['recommendation']}")
                
                # Competitor analysis
                if 'competitor_analysis' in results:
                    print(f"\n🥇 Market Position Analysis:")
                    print(f"{'-'*40}")
                    competition = results['competitor_analysis']
                    
                    if 'market_position_percentile' in competition and competition['market_position_percentile'] is not None:
                        print(f"Market Position: {competition['market_position_percentile']}th percentile in {competition['category']}")
                    
                    print("\nTop Competitors:")
                    for comp in competition['competitors'][:3]:  # Show top 3
                        print(f"• {comp['name']} - Rating: {comp['rating']}, Downloads: {comp['downloads']}")
                
                # Visualization
                if 'visualization_path' in results:
                    print(f"\n📊 Visualization saved to: {results['visualization_path']}")
                
                print(f"\n{'='*60}")
                print(f"Analysis Date: {results['date_analyzed']}")
                print(f"{'='*60}")
            
            return
        except Exception as e:
            import traceback
            print(f"Error predicting app longevity: {str(e)}")
            traceback.print_exc()
            # Continue with regular training if prediction fails
    
    # Create model directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    # Load data
    try:
        df = pd.read_csv(args.data_file)
        print(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"Error: {args.data_file} not found. Please run data_collection.py first.")
        return
    
    # Initialize model
    use_lstm = args.use_lstm
    use_advanced_models = not args.disable_advanced
    
    model = AppLongevityModel(use_lstm=use_lstm, use_advanced_models=use_advanced_models)
    
    # Preprocess data
    std_data, lstm_data = model.preprocess_data(df)
    
    # Determine input shape for LSTM
    input_shape = None
    if lstm_data is not None:
        X_lstm_train = lstm_data[0]
        input_shape = X_lstm_train.shape[1:]
    
    # Build models
    model.build_models(input_shape)
    
    # Train models
    model.train_models(std_data, lstm_data)
    
    # Save models
    model.save_models()
    
    print("Model training and evaluation complete. Results saved to reports/ directory.")

if __name__ == "__main__":
    main()
