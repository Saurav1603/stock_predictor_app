import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# --- Configuration ---
SEQUENCE_LENGTH = 60 # Use last 60 days of prices to predict the next day
TRAIN_SPLIT_PERCENT = 0.8
EPOCHS = 50 # Keep epochs lower for faster UI demo, can be increased
BATCH_SIZE = 32

# --- Helper Functions ---
def load_data(ticker, start_date, end_date):
    """Loads stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None
    return data

def preprocess_data(data):
    """Prepares data for LSTM model: scaling and sequencing."""
    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(SEQUENCE_LENGTH, len(scaled_prices)):
        X.append(scaled_prices[i-SEQUENCE_LENGTH:i, 0])
        y.append(scaled_prices[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler, scaled_prices # Return scaled_prices for next day prediction

def split_data(X, y):
    """Splits data into training and testing sets chronologically."""
    split_index = int(len(X) * TRAIN_SPLIT_PERCENT)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    """Builds the LSTM model."""
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test, progress_bar_placeholder=None):
    """Trains the LSTM model."""
    # Simple callback to update progress (can be more sophisticated)
    class ProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, total_epochs, placeholder):
            super().__init__()
            self.total_epochs = total_epochs
            self.placeholder = placeholder
            self.current_epoch = 0

        def on_epoch_end(self, epoch, logs=None):
            self.current_epoch += 1
            if self.placeholder:
                progress_percent = self.current_epoch / self.total_epochs
                self.placeholder.progress(progress_percent, text=f"Training Epoch {self.current_epoch}/{self.total_epochs}")

    callbacks_list = []
    if progress_bar_placeholder:
        callbacks_list.append(ProgressCallback(EPOCHS, progress_bar_placeholder))

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        verbose=0, # Set to 0 to avoid printing to console, Streamlit will handle updates
        callbacks=callbacks_list
    )
    return history

def make_predictions(model, X_test, scaler):
    """Makes predictions and inverse transforms them."""
    predicted_prices_scaled = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices_scaled)
    return predicted_prices

def calculate_rmse(actual, predicted):
    """Calculates Root Mean Squared Error."""
    return np.sqrt(np.mean((predicted - actual)**2))

def plot_predictions(actual_prices, predicted_prices, ticker):
    """Plots actual vs. predicted prices."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(actual_prices, color='blue', label='Actual Stock Price')
    ax.plot(predicted_prices, color='red', linestyle='--', label='Predicted Stock Price')
    ax.set_title(f'{ticker} Stock Price Prediction')
    ax.set_xlabel('Time (Days in Test Set)')
    ax.set_ylabel('Stock Price (USD)')
    ax.legend()
    ax.grid(True)
    return fig

def plot_loss(history):
    """Plots training and validation loss."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_title('Model Loss During Training')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.legend()
    ax.grid(True)
    return fig

def predict_next_day(model, full_scaled_data, scaler):
    """Predicts the next day's price (highly speculative)."""
    last_sequence_scaled = full_scaled_data[-SEQUENCE_LENGTH:]
    last_sequence_scaled = np.reshape(last_sequence_scaled, (1, SEQUENCE_LENGTH, 1))
    next_day_prediction_scaled = model.predict(last_sequence_scaled)
    next_day_prediction = scaler.inverse_transform(next_day_prediction_scaled)
    return next_day_prediction[0,0]

# Import tensorflow here to avoid potential issues with Streamlit's execution order
import tensorflow as tf