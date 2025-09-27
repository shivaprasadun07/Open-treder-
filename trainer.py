import pandas as pd
import numpy as np
import joblib
import os
import yaml
from ta import add_all_ta_features

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# --- Main Function to Run the Training ---
def run_training():
    """
    Main function to load data, process features, train models,
    and save the final artifacts.
    """
    print("--- Starting Model Training Process ---")

    # 1. Load Configuration
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        feature_names = config['model']['feature_names']
        print("✅ Configuration loaded.")
    except FileNotFoundError:
        print("❌ FATAL: config.yaml not found. Make sure it's in the same directory.")
        return

    # 2. Load and Combine Data
    # We will combine data from all pairs to train a more general model.
    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    if not data_files:
        print("❌ FATAL: No data files found in the 'data/' folder. Run download_data.py first.")
        return
    
    all_data = pd.concat([pd.read_csv(os.path.join('data', f)) for f in data_files])
    all_data = all_data.sort_values('timestamp').reset_index(drop=True)
    print(f"✅ Data loaded and combined. Total rows: {len(all_data)}")

    # 3. Feature Engineering and Target Creation
    print("🔬 Engineering features and creating target variable...")
    features, target = create_features_and_target(all_data, feature_names)
    print("✅ Features and target created.")

    # 4. Data Splitting and Scaling
    print("Spliting and scaling data...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(features, target)
    print("✅ Data split and scaled.")
    
    # Create the 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the scaler immediately - it's needed for the bot
    scaler_path = config['model']['scaler_path']
    joblib.dump(scaler, scaler_path)
    print(f"✅ Data scaler saved to {scaler_path}")

    # 5. Train Random Forest Model
    print("\n--- Training Random Forest Model ---")
    rf_model = train_random_forest(X_train, y_train)
    print("Evaluating Random Forest model...")
    y_pred_rf = rf_model.predict(X_test)
    print(classification_report(y_test, y_pred_rf, target_names=['SHORT', 'HOLD', 'LONG']))
    
    # Save the RF model
    rf_model_path = config['model']['rf_model_path']
    joblib.dump(rf_model, rf_model_path)
    print(f"✅ Random Forest model saved to {rf_model_path}")

    # 6. Train LSTM Model
    print("\n--- Training LSTM Model ---")
    # We need to reshape the data for the LSTM model
    X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Convert target to categorical for LSTM (short, hold, long)
    y_train_cat = to_categorical(y_train, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    lstm_model = train_lstm(X_train_lstm, y_train_cat, len(feature_names))
    print("Evaluating LSTM model...")
    loss, accuracy = lstm_model.evaluate(X_test_lstm, y_test_cat, verbose=0)
    print(f"LSTM Model Test Accuracy: {accuracy*100:.2f}%")
    
    # Save the LSTM model weights
    lstm_weights_path = config['model']['lstm_weights_path']
    lstm_model.save_weights(lstm_weights_path)
    print(f"✅ LSTM model weights saved to {lstm_weights_path}")
    
    print("\n--- 🎉 Training Complete! Your bot's brain is ready. ---")


def create_features_and_target(df, feature_names):
    """
    Adds technical indicators to the dataframe and creates the target variable.
    """
    # Add all technical analysis features
    df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
    
    # --- Create the Target Variable (The "Answer" the model needs to learn) ---
    # We want to predict if the price will go up or down in the near future.
    # Let's define "future" as the next 5 candles (5 * 15 = 75 minutes).
    future_periods = 5
    # Calculate the percentage change in price in the future
    df['future_return'] = df['close'].pct_change(periods=-future_periods).shift(-future_periods)
    
    # Define thresholds for what we consider a "long", "short", or "hold" signal.
    # If the price goes up by more than 0.2% we'll call it a LONG signal.
    # If it goes down by more than 0.2% we'll call it a SHORT signal.
    # Anything in between is a HOLD.
    long_threshold = 0.002
    short_threshold = -0.002
    
    # Create the target column based on the thresholds
    df['target'] = 2 # Default to HOLD
    df.loc[df['future_return'] > long_threshold, 'target'] = 1 # LONG
    df.loc[df['future_return'] < short_threshold, 'target'] = 0 # SHORT
    
    # Drop rows with missing values that were created during feature calculation
    df = df.dropna()
    
    # Select only the features specified in config.yaml
    features = df[feature_names]
    target = df['target']
    
    return features, target


def split_and_scale_data(features, target):
    """
    Splits data into training and testing sets and scales the features.
    """
    # We split the data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    
    # Scaling is very important for neural networks.
    # It normalizes the data so that all features have a similar scale.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_random_forest(X_train, y_train):
    """
    Trains a RandomForestClassifier model.
    """
    model = RandomForestClassifier(
        n_estimators=100,      # Number of trees in the forest
        random_state=42,
        n_jobs=-1              # Use all available CPU cores
    )
    model.fit(X_train, y_train)
    return model


def train_lstm(X_train, y_train, n_features):
    """
    Builds and trains an LSTM neural network.
    """
    model = Sequential([
        LSTM(64, input_shape=(n_features, 1), return_sequences=True),
        Dropout(0.2), # Dropout helps prevent overfitting
        LSTM(32),
        Dropout(0.2),
        Dense(3, activation='softmax') # 3 outputs: SHORT, HOLD, LONG
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # We train the model for 10 epochs (passes over the data).
    # This might take a few minutes depending on your computer.
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    
    return model

# --- This makes the script runnable from the command line ---
if __name__ == "__main__":
    run_training()