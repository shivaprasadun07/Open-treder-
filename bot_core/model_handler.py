import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from ta import add_all_ta_features

class HybridModel:
    def __init__(self, config):
        self.rf_model = joblib.load(config['rf_model_path'])
        self.scaler = joblib.load(config['scaler_path'])
        self.feature_names = config['feature_names']
        self.lstm_model = self._build_lstm()
        self.lstm_model.load_weights(config['lstm_weights_path'])

    def _build_lstm(self):
        # This defines the structure of the LSTM model.
        # It must match the structure used in the trainer.py script.
        model = Sequential([
            LSTM(64, input_shape=(len(self.feature_names), 1), return_sequences=True),
            LSTM(32),
            Dense(3, activation='softmax') # 3 outputs for Short, Long, Hold
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        return model

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds all technical indicators and selects the ones the model was trained on."""
        df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume", fillna=True)
        # Ensure all required columns exist, fill with 0 if not
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        return df[self.feature_names]

    def predict(self, features: pd.DataFrame) -> tuple[float, float]:
        """Returns (long_probability, short_probability)"""
        # Step 1: Scale the incoming features
        scaled_features = self.scaler.transform(features)

        # Step 2: Reshape data for the LSTM model
        lstm_input = scaled_features.reshape(scaled_features.shape[0], scaled_features.shape[1], 1)

        # Step 3: Get predictions from both models.
        # They both return 3 probabilities: [prob_short, prob_long, prob_hold]
        rf_proba = self.rf_model.predict_proba(scaled_features)[0] 
        lstm_proba = self.lstm_model.predict(lstm_input, verbose=0)[0] 

        # Step 4: Correctly pull out the probabilities for SHORT (index 0) and LONG (index 1)
        # and average them together.
        final_prob_short = (rf_proba[0] + lstm_proba[0]) / 2
        final_prob_long = (rf_proba[1] + lstm_proba[1]) / 2

        # Step 5: Return the final probabilities
        return final_prob_long, final_prob_short