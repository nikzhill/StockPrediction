# app.py - FastAPI backend for AI Stock Prediction
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
import os
import yfinance as yf
from typing import Dict, Any

app = FastAPI(title="AI Stock Prediction API")

# Allow frontend (React/Vite) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data/processed"
MODEL_DIR = "models"


# ------------------------------
# Utility Functions
# ------------------------------
def load_data(ticker: str):
    file_path = f"{DATA_DIR}/processed_{ticker}_stock_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        return df
    return None


def load_sentiment():
    path = "data/sentiment_output.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


def load_model(ticker: str):
    model_path = f"{MODEL_DIR}/{ticker}_xgboost.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def get_latest_price(ticker: str):
    stock = yf.Ticker(ticker)
    df_live = stock.history(period="1d", interval="1m")
    if not df_live.empty:
        return float(df_live["Close"].iloc[-1])
    return None


# ------------------------------
# API Endpoints
# ------------------------------


@app.get("/api/stock/{ticker}")
def get_stock_data(ticker: str) -> Dict[str, Any]:
    df = load_data(ticker)
    if df is None or df.empty:
        return {"error": f"No data found for {ticker}"}

    close = df["Close"].iloc[-1] if "Close" in df.columns else None
    volume = df["Volume"].iloc[-1] if "Volume" in df.columns else None
    rsi = df["RSI"].iloc[-1] if "RSI" in df.columns else None

    return {
        "ticker": ticker,
        "latest": {
            "close": close,
            "volume": volume,
            "rsi": rsi,
        },
        "chartData": df[["Date", "Close"]].tail(60).to_dict(orient="records"),
    }


@app.get("/api/sentiment/{ticker}")
def get_sentiment(ticker: str):
    df = load_sentiment()
    if df is None or df.empty:
        return {"error": "No sentiment data found"}

    df_ticker = df[df["ticker"] == ticker] if "ticker" in df.columns else df
    if df_ticker.empty:
        return {"error": f"No sentiment found for {ticker}"}

    last = df_ticker.iloc[-1]
    return {
        "ticker": ticker,
        "sentiment": {
            "compound": float(last.get("sentiment_compound", 0)),
            "positive": float(last.get("sentiment_positive", 0)),
            "negative": float(last.get("sentiment_negative", 0)),
        },
    }


@app.get("/api/predict/{ticker}")
def predict_stock(ticker: str):
    df = load_data(ticker)
    model = load_model(ticker)

    if df is None or model is None:
        return {"error": f"No data/model found for {ticker}"}

    latest = df.iloc[-1:].copy()
    close_price = latest["Close"].iloc[0] if "Close" in latest.columns else None

    # Collect features
    features = []
    for col in df.columns:
        if any(ind in col for ind in ["SMA", "EMA", "RSI"]):
            features.append(col)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            norm_col = f"{col}_Norm"
            latest[norm_col] = (latest[col] - df[col].mean()) / df[col].std()
            features.append(norm_col)
    for col in ["sentiment_compound", "sentiment_positive", "sentiment_negative"]:
        if col in df.columns:
            features.append(col)

    if hasattr(model, "feature_names_in_"):
        X_pred = latest.reindex(columns=model.feature_names_in_, fill_value=0)
    else:
        X_pred = latest[features].fillna(0)

    prediction = float(model.predict(X_pred)[0])
    change = prediction - close_price if close_price else None
    pct = (change / close_price * 100) if close_price else None

    return {
        "ticker": ticker,
        "prediction": {
            "tomorrow": prediction,
            "change": change,
            "percent": pct,
            "confidence": np.random.uniform(0.7, 0.95),  # placeholder
        },
    }
