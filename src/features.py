# src/features.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_price_data(ticker: str, config_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    ensure_dir(processed_dir)
    file_path = processed_dir / f"{ticker}_prices.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Processed price data not found: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df

def load_sentiment_data(ticker: str, config_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    sentiment_dir = Path(cfg["paths"]["sentiment_dir"])
    file_path = sentiment_dir / f"{ticker}_sentiment_daily.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Sentiment CSV not found: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df

def compute_technical_indicators(df: pd.DataFrame, config_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    ma_windows = cfg["features"]["ma_windows"]
    rsi_period = cfg["features"]["rsi_period"]
    atr_period = cfg["features"]["atr_period"]
    
    # Moving averages
    for window in ma_windows:
        df[f"ma_{window}"] = df["close"].rolling(window).mean()
    
    # RSI
    df["rsi"] = RSIIndicator(df["close"], rsi_period).rsi()
    
    # ATR
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"], atr_period).average_true_range()
    
    # Lag features
    lag_days = cfg["features"]["lag_days"]
    df["close_lag"] = df["close"].shift(lag_days)
    
    return df

def merge_features_sentiment(ticker: str, config_path: str = "config.yaml") -> pd.DataFrame:
    df_prices = load_price_data(ticker, config_path)
    df_prices = compute_technical_indicators(df_prices, config_path)
    df_sent = load_sentiment_data(ticker, config_path)
    
    # Merge on date
    df = pd.merge(df_prices, df_sent, on="date", how="left")
    
    # Fill missing sentiment with neutral
    df["mean_compound"].fillna(0.0, inplace=True)
    df["sentiment_label"].fillna("neutral", inplace=True)
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate features with technical + sentiment for a ticker")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    df_features = merge_features_sentiment(args.ticker, args.config)
    output_path = Path(load_config(args.config)["paths"]["processed_dir"]) / f"{args.ticker}_features.csv"
    df_features.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
