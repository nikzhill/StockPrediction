# verify_predictions.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import joblib

# ----------------------------
# Configuration
# ----------------------------
TICKERS = ["AAPL", "GOOGL", "MSFT", "TSLA", "INFY.NS"]  # add/remove as needed
MODEL_DIR = "models"
LOG_FILE = "prediction_verification.csv"


# ----------------------------
# Predict next-day close for a ticker
# ----------------------------
def make_prediction(ticker):
    model_path = os.path.join(MODEL_DIR, f"{ticker}_xgboost.pkl")
    if not os.path.exists(model_path):
        print(f"⚠️ Model not found for {ticker}")
        return None

    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"❌ Error loading model for {ticker}: {e}")
        return None

    # Fetch last 90 days stock data
    end = datetime.today()
    start = end - timedelta(days=90)
    df = yf.download(ticker, start=start, end=end)
    if df.empty:
        return None
    df.reset_index(inplace=True)

    # Simple feature engineering (must match training)
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df = df.dropna()

    if df.empty:
        return None

    latest = df.iloc[-1]

    try:
        features = latest[["SMA_10", "EMA_10", "Close"]].values.reshape(1, -1)
    except KeyError:
        return None

    predicted_close = model.predict(features)[0]

    return {
        "date": latest["Date"].strftime("%Y-%m-%d"),
        "stock": ticker,
        "predicted_close": round(predicted_close, 2),
        "actual_close": None,
        "error": None,
        "mape(%)": None,
    }


# ----------------------------
# Update missing actuals
# ----------------------------
def update_actuals(df):
    updated_rows = []
    for idx, row in df[df["actual_close"].isna()].iterrows():
        ticker = row["stock"]
        next_day = (
            datetime.strptime(row["date"], "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        actual_df = yf.download(ticker, start=next_day, end=next_day)
        if not actual_df.empty:
            actual_close = round(actual_df["Close"].iloc[0], 2)
            error = actual_close - row["predicted_close"]
            mape = abs(error) / actual_close * 100

            df.at[idx, "actual_close"] = actual_close
            df.at[idx, "error"] = round(error, 2)
            df.at[idx, "mape(%)"] = round(mape, 2)
            updated_rows.append((row["stock"], row["date"]))
    return df, updated_rows


# ----------------------------
# Main
# ----------------------------
# Load existing log
if os.path.exists(LOG_FILE):
    log_df = pd.read_csv(LOG_FILE)
else:
    log_df = pd.DataFrame(
        columns=["date", "stock", "predicted_close", "actual_close", "error", "mape(%)"]
    )

# 1️⃣ Make new predictions for today
new_rows = []
for ticker in TICKERS:
    pred = make_prediction(ticker)
    if (
        pred
        and not (
            (log_df["date"] == pred["date"]) & (log_df["stock"] == pred["stock"])
        ).any()
    ):
        new_rows.append(pred)

if new_rows:
    log_df = pd.concat([log_df, pd.DataFrame(new_rows)], ignore_index=True)

# 2️⃣ Try to fill missing actuals
log_df, updated = update_actuals(log_df)

# Save
log_df.to_csv(LOG_FILE, index=False)

print("✅ Predictions logged.")
if new_rows:
    print("➕ Added new predictions for:", ", ".join([r["stock"] for r in new_rows]))
if updated:
    print("📈 Updated actuals for:", ", ".join([f"{s} ({d})" for s, d in updated]))
