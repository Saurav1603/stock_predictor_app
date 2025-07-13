import streamlit as st # This should be one of the first imports
import datetime
import numpy as np
import pandas as pd
from stock_predictor_model import ( # Your model imports
    load_data, preprocess_data, split_data,
    build_model, train_model, make_predictions,
    calculate_rmse, plot_predictions, plot_loss,
    predict_next_day, SEQUENCE_LENGTH
)

# --- Page Configuration - THIS MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
# --- NO OTHER st.command() BEFORE THIS LINE ---


# --- Now other Streamlit UI elements can follow ---
st.title("📈 Stock Price Predictor")
st.caption("Predict future stock prices using LSTMs. Educational purposes only.")

# --- Sidebar for Inputs ---
st.sidebar.header("Input Parameters")
# ... (rest of your app.py code)

# Default values
default_ticker = "AAPL"
default_start_date = datetime.date(2015, 1, 1)
default_end_date = datetime.date.today() - datetime.timedelta(days=1) # Yesterday

ticker = st.sidebar.text_input("Stock Ticker", value=default_ticker).upper()
start_date = st.sidebar.date_input("Start Date", value=default_start_date)
end_date = st.sidebar.date_input("End Date", value=default_end_date, max_value=datetime.date.today() - datetime.timedelta(days=1))


run_button = st.sidebar.button("🚀 Run Prediction")

# --- Main Area for Outputs ---
if run_button:
    if not ticker:
        st.error("Please enter a stock ticker.")
    elif start_date >= end_date:
        st.error("Error: Start date must be before end date.")
    elif (end_date - start_date).days < SEQUENCE_LENGTH + 10: # Ensure enough data for sequence + some test
        st.error(f"Date range too short. Need at least {SEQUENCE_LENGTH + 10} days for processing.")
    else:
        with st.status(f"Processing {ticker} data...", expanded=True) as status:
            try:
                # 1. Load Data
                st.write("💾 Loading data...")
                data = load_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if data is None or data.empty:
                    st.error(f"Could not load data for {ticker}. Check ticker or date range.")
                    status.update(label="Data loading failed!", state="error", expanded=False)
                    st.stop()
                if len(data) < SEQUENCE_LENGTH:
                    st.error(f"Not enough historical data for ticker {ticker} in the selected range to form even one sequence of {SEQUENCE_LENGTH} days. Please select an earlier start date or a different ticker.")
                    status.update(label="Data loading failed!", state="error", expanded=False)
                    st.stop()

                st.success(f"Data loaded successfully for {ticker} ({len(data)} days).")
                st.dataframe(data.tail(), use_container_width=True)

                # 2. Preprocess Data
                st.write("⚙️ Preprocessing data...")
                X, y, scaler, full_scaled_data = preprocess_data(data) # Capture full_scaled_data
                if len(X) == 0: # This check might be redundant if len(data) < SEQUENCE_LENGTH is caught above
                    st.error("Not enough data to create sequences after preprocessing. Try an earlier start date.")
                    status.update(label="Preprocessing failed!", state="error", expanded=False)
                    st.stop()

                # 3. Split Data
                X_train, X_test, y_train, y_test = split_data(X, y)
                st.write(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

                if X_train.shape[0] == 0 or X_test.shape[0] == 0:
                    st.error("Not enough data to create training and testing sets. Please expand the date range.")
                    status.update(label="Data splitting failed!", state="error", expanded=False)
                    st.stop()


                # 4. Build Model
                st.write("🧠 Building LSTM model...")
                model = build_model(input_shape=(X_train.shape[1], 1))

                # 5. Train Model
                st.write("💪 Training model... (this may take a few minutes)")
                progress_bar_placeholder = st.progress(0, text="Initializing training...")
                history = train_model(model, X_train, y_train, X_test, y_test, progress_bar_placeholder)
                progress_bar_placeholder.progress(1.0, text="Training Complete!")
                st.success("Model training complete!")

                # 6. Make Predictions
                st.write("📊 Making predictions...")
                predicted_prices = make_predictions(model, X_test, scaler)
                actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

                # 7. Evaluate
                rmse = calculate_rmse(actual_prices, predicted_prices)
                st.metric(label="Root Mean Squared Error (RMSE) on Test Set", value=f"${rmse:.2f}")

                # 8. Predict Next Day (Speculative)
                last_actual_price_series = data['Close'].iloc[-1]
                last_actual_date = data.index[-1].strftime('%Y-%m-%d')
                next_day_pred = predict_next_day(model, full_scaled_data, scaler) # next_day_pred is a float

                # Ensure last_actual_price is a float for the f-string calculation
                if isinstance(last_actual_price_series, (pd.Series, pd.DataFrame)):
                    last_actual_price_float = last_actual_price_series.item()
                elif isinstance(last_actual_price_series, (int, float, np.number)):
                    last_actual_price_float = float(last_actual_price_series)
                else:
                    st.warning(f"Could not determine numeric type for last_actual_price: {type(last_actual_price_series)}. Using 0 for delta calculation.")
                    last_actual_price_float = 0.0 # Fallback, though ideally this shouldn't happen

                if last_actual_price_float != 0: # Avoid division by zero
                    delta_value = next_day_pred - last_actual_price_float
                    delta_percentage = (delta_value / last_actual_price_float) * 100
                    delta_str = f"${delta_value:.2f} ({delta_percentage:.2f}%)"
                    delta_color_val = "neutral" if delta_value == 0 else ("inverse" if delta_value < 0 else "normal")
                else:
                    delta_str = "N/A (last price was 0 or unavailable)"
                    delta_color_val = "off"


                st.metric(label=f"Predicted Next Trading Day's Close (after {last_actual_date})",
                          value=f"${next_day_pred:.2f}",
                          delta=delta_str,
                          delta_color=delta_color_val)


                status.update(label="Prediction complete!", state="complete", expanded=False)

                # --- Display Results ---
                st.subheader("Prediction Results")
                fig_pred = plot_predictions(actual_prices, predicted_prices, ticker)
                st.pyplot(fig_pred)

                st.subheader("Model Training History")
                fig_loss = plot_loss(history)
                st.pyplot(fig_loss)

                st.info("Remember: This model is for educational purposes. Past performance is not indicative of future results. Do not use for financial decisions.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                status.update(label="Error during processing!", state="error", expanded=True)
                import traceback
                st.error(traceback.format_exc())

else:
    st.info("Enter parameters in the sidebar and click 'Run Prediction'.")
    st.markdown(f"""
    **How this works:**
    1.  **Data Loading:** Fetches historical stock data from Yahoo Finance.
    2.  **Preprocessing:**
        *   Uses only the 'Close' price.
        *   Scales prices to a range of 0-1 (MinMaxScaler).
        *   Creates sequences of past data (uses {SEQUENCE_LENGTH} days of prices to predict the next).
    3.  **Model:** A Long Short-Term Memory (LSTM) neural network is used.
    4.  **Training:** The model learns patterns from the historical sequences.
    5.  **Prediction:** Predicts prices on a test set (unseen data) and the next potential trading day.
    6.  **Evaluation:** Root Mean Squared Error (RMSE) measures the difference.

    **Disclaimer:** Stock market prediction is highly complex and speculative. This tool is a simplified demonstration and should **NOT** be used for making real investment decisions.
    """)
