import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Attention, Concatenate
from tensorflow.keras.callbacks import EarlyStopping  # For Colab stability

def create_hybrid_model(X_train, y_train, input_shape, use_lstm=True):
    """Creates a hybrid ensemble of traditional ML and deep learning models."""

    # 1. Traditional Models
    rf = RandomForestRegressor(random_state=42, n_estimators=50)  # Reduced n_estimators
    gb = GradientBoostingRegressor(random_state=42, n_estimators=50)  # Reduced n_estimators
    ada = AdaBoostRegressor(random_state=42, n_estimators=30)  # Reduced n_estimators
    lr = LinearRegression()

    estimators = [('rf', rf), ('gb', gb), ('ada', ada), ('lr', lr)]
    weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights to start

    if use_lstm:
        # 2. Neural Network with LSTM and Attention
        input_layer = Input(shape=input_shape)
        lstm_layer = LSTM(32, return_sequences=True)(input_layer)  # Reduced LSTM units
        attention_layer = Attention()([lstm_layer, lstm_layer])
        lstm_output = tf.keras.layers.Flatten()(attention_layer)
        dense_1 = Dense(64, activation='relu')(lstm_output)  # Reduced Dense units
        dense_2 = Dense(32, activation='relu')(dense_1)
        nn_output = Dense(1, activation='linear')(dense_2)

        neural_network = Model(inputs=input_layer, outputs=input_layer)
        neural_network.compile(optimizer='adam', loss='mean_squared_error')

        estimators.append(('nn', neural_network))
        weights = [0.15, 0.15, 0.15, 0.15, 0.4]  # Increased weight for NN

        # Train the neural network (can be adjusted)
    else:
        neural_network = None

    ensemble = VotingRegressor(estimators=estimators, weights=weights)

    if use_lstm:
        ensemble.fit(X_train[:, 0, :], y_train)  # Use only non-sequential for VotingRegressor
    else:
        ensemble.fit(X_train, y_train)

    return ensemble, neural_network

def train_and_evaluate_model(df, use_lstm=True):
    """Trains and evaluates the hybrid model with cross-validation."""

    # 1. Data Preparation
    X = df.drop(columns=["app_name", "rating"], errors='ignore')
    y = df["rating"]
    X = X.fillna(0)

    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    if use_lstm:
        # Ensure proper shape for LSTM input
        sequence_length = 5  # Adjust as needed
        X_lstm = []
        y_lstm = []
        app_ids = df['app_id'].unique()

        for app_id in app_ids:
            app_data = df[df['app_id'] == app_id][numerical_cols].values
            if len(app_data) >= sequence_length:
                for i in range(len(app_data) - sequence_length + 1):
                    X_lstm.append(app_data[i:i + sequence_length])
                    y_lstm.append(df[df['app_id'] == app_id]['rating'].values[i + sequence_length - 1])

        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        if X_lstm.size == 0 or y_lstm.size == 0:  # Handle empty data
            print("Error: No data available for LSTM training.")
            return

        input_shape = (X_lstm.shape[1], X_lstm.shape[2])
        X = X_lstm
        y = y_lstm
    else:
        input_shape = (X.shape[1],)

    # 2. Cross-Validation and Hyperparameter Tuning
    kf = KFold(n_splits=3, shuffle=True, random_state=42)  # Reduced n_splits for speed
    mse_scores = []
    r2_scores = []

    for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # a. Create Model
        ensemble_model, neural_network = create_hybrid_model(X_train, y_train, input_shape, use_lstm)

        # b. Hyperparameter Tuning (Illustrative - Expand This!)
        param_grid_rf = {'n_estimators': [30, 50], 'max_depth': [4, None]}  # Reduced search space
        grid_search_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf,
                                      cv=2, scoring='neg_mean_squared_error', verbose=1)
        if use_lstm:
            grid_search_rf.fit(X_train[:, 0, :], y_train)
        else:
            grid_search_rf.fit(X_train, y_train)
        if ensemble_model.named_estimators_ and 'rf' in ensemble_model.named_estimators_:
            ensemble_model.named_estimators_['rf'] = grid_search_rf.best_estimator_
            print(f"Fold {fold + 1} - Best RF params: {grid_search_rf.best_params_}")

        # c. Model Evaluation
        if use_lstm:
            y_pred = ensemble_model.predict(X_val[:, 0, :])
        else:
            y_pred = ensemble_model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

        print(f"Fold {fold + 1} - MSE: {mse:.4f}, R^2: {r2:.4f}")

    print(f"\nAvg. MSE: {np.mean(mse_scores):.4f}, Avg. R^2: {np.mean(r2_scores):.4f}")

    # 3. Final Training and Saving
    final_ensemble_model, final_nn_model = create_hybrid_model(X, y, input_shape, use_lstm)
    pickle.dump(final_ensemble_model, open("models/final_app_ensemble.pkl", "wb"))
    if use_lstm and final_nn_model is not None:
        final_nn_model.save("models/final_nn_model.h5")  # Save Keras model

def main():
    try:
        df = pd.read_csv("data/raw/app_data_final.csv")
    except FileNotFoundError:
        print("Error: data/raw/app_data_final.csv not found. Please run data_collection.py first.")
        return

    # --- Choose LSTM or not ---
    use_lstm = True  # Set to False to disable LSTM

    train_and_evaluate_model(df, use_lstm)

if __name__ == "__main__":
    main()