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
            if any(char in col for char in ['[', ']', '<']):
                sanitized_col = col.replace('[', '_').replace(']', '_').replace('<', '_')
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
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_name} - Actual vs Predicted')
        plt.tight_layout()
        plt.savefig(f'reports/{model_name}_predictions.png')
        plt.close()
    
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
        
        # Sort for visualization
        indices = np.argsort(importance)[-20:]  # Top 20 features
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Feature Importances - {model_name}')
        plt.tight_layout()
        plt.savefig(f'reports/{model_name}_feature_importance.png')
        plt.close()
        
        # Save feature importances to file
        with open(f'reports/{model_name}_feature_importance.json', 'w') as f:
            json.dump(self.feature_importances[model_name], f, indent=2)
    
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
        
        # Visualize model comparison
        plt.figure(figsize=(12, 8))
        
        metrics = ['rmse', 'mae', 'r2']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, metric in enumerate(metrics):
            values = [m[metric] for m in self.model_metrics.values()]
            x = range(len(self.model_metrics))
            plt.bar([p + i*0.25 for p in x], values, width=0.25, 
                   color=colors[i], label=metric.upper())
        
        plt.xticks([p + 0.25 for p in range(len(self.model_metrics))], 
                  list(self.model_metrics.keys()))
        plt.legend()
        plt.title('Model Comparison')
        plt.tight_layout()
        plt.savefig('reports/model_comparison.png')
        plt.close()
        
        return self.best_model
    
    def _generate_shap_values(self, model, X_sample):
        """Generate SHAP values for model explainability"""
        
        try:
            # Take a small sample to speed up SHAP computation
            X_sample_small = X_sample.iloc[:100] if len(X_sample) > 100 else X_sample
            
            # Create explainer
            explainer = shap.Explainer(model, X_sample_small)
            shap_values = explainer(X_sample_small)
            
            # Plot SHAP summary
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample_small, show=False)
            plt.tight_layout()
            plt.savefig('reports/shap_summary.png')
            plt.close()
            
            # Plot SHAP bar summary
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample_small, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig('reports/shap_bar_summary.png')
            plt.close()
            
            print("SHAP analysis completed and saved to reports/")
        except Exception as e:
            print(f"Error generating SHAP values: {e}")
    
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

    def predict_app_longevity(self, app_name):
        """Predict longevity for a new app by name"""
        try:
            # Import data collection module to fetch app data
            import sys
            sys.path.append('./scripts')
            from data_collection import fetch_app_store_data, fetch_play_store_data, calculate_feature_engineering
            
            print(f"Fetching data for app: {app_name}")
            
            # Try to find the app in both stores
            ios_data = None
            android_data = None
            
            # For iOS, try to search by app name
            try:
                # Use iTunes search API to find app ID by name
                import requests
                search_url = f"https://itunes.apple.com/search?term={app_name.replace(' ', '+')}&entity=software&limit=1"
                response = requests.get(search_url)
                search_data = response.json()
                
                if search_data['resultCount'] > 0:
                    app_id = str(search_data['results'][0]['trackId'])
                    print(f"Found iOS app ID: {app_id}")
                    ios_data = fetch_app_store_data(app_id)
                    if ios_data:
                        ios_data = calculate_feature_engineering(ios_data)
                else:
                    print(f"No iOS app found for '{app_name}'")
            except Exception as e:
                print(f"Error searching iOS app: {e}")
            
            # For Android, try to search by app name
            try:
                # Try common package patterns
                package_guesses = [
                    f"com.{app_name.lower().replace(' ', '')}",
                    f"com.{app_name.lower().replace(' ', '.')}"
                ]
                
                for package in package_guesses:
                    try:
                        android_data = fetch_play_store_data(package)
                        if android_data:
                            print(f"Found Android app with package: {package}")
                            android_data = calculate_feature_engineering(android_data)
                            break
                    except:
                        continue
                    
                if not android_data:
                    print(f"No Android app found for '{app_name}'")
            except Exception as e:
                print(f"Error searching Android app: {e}")
            
            # Combine or select data for prediction
            app_data = ios_data if ios_data else android_data
            
            if not app_data:
                print(f"Could not find app data for '{app_name}' on either platform")
                return None
            
            # Convert to DataFrame for prediction
            app_df = pd.DataFrame([app_data])
            
            # Preprocess for model
            # Remove columns not used in training
            for col in ['app_name', 'app_id', 'keywords', 'reviews']:
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
            
            # Make prediction
            prediction = self.predict(app_df, self.best_model)
            
            results = {
                'app_name': app_name,
                'platform': app_data.get('platform', 'Unknown'),
                'predicted_longevity': float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction),
                'key_features': {
                    'rating': app_data.get('rating'),
                    'downloads': app_data.get('downloads'),
                    'days_since_release': app_data.get('days_since_release'),
                    'positive_sentiment_ratio': app_data.get('positive_sentiment_ratio')
                }
            }
            
            return results
        except Exception as e:
            print(f"Error predicting app longevity: {e}")
            return None

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
                       
    # Ignore unknown arguments to handle Jupyter/Colab environment
    args, unknown = parser.parse_known_args()
    
    # Check if we should predict for a specific app
    if args.predict_app:
        try:
            # Try to load existing model
            loaded_model = AppLongevityModel()
            
            # Try to load the best model
            model_files = os.listdir('models')
            if not model_files or not any(f.endswith('.joblib') for f in model_files):
                print("No trained models found. Training new models first...")
                # Load data and train models
                df = pd.read_csv(args.data_file)
                loaded_model.preprocess_data(df)
                loaded_model.build_models()
                std_data, lstm_data = loaded_model.preprocess_data(df)
                loaded_model.train_models(std_data, lstm_data)
                loaded_model.save_models()
            else:
                # Load the best model
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
                
            # Predict for the app
            results = loaded_model.predict_app_longevity(args.predict_app)
            if results:
                print("\nApp Longevity Prediction Results:")
                print(f"App Name: {results['app_name']}")
                print(f"Platform: {results['platform']}")
                print(f"Predicted Longevity Score: {results['predicted_longevity']:.4f}")
                print("\nKey Features:")
                for feature, value in results['key_features'].items():
                    print(f"- {feature}: {value}")
            return
        except Exception as e:
            print(f"Error predicting app longevity: {e}")
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
