# src/model.py
import os
from pathlib import Path
import pandas as pd
import joblib
import json
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def load_features(ticker: str, config_path: str = "config.yaml") -> pd.DataFrame:
    cfg = load_config(config_path)
    processed_dir = Path(cfg["paths"]["processed_dir"])
    file_path = processed_dir / f"{ticker}_features.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Features CSV not found: {file_path}")
    df = pd.read_csv(file_path, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df

def train_xgboost(df: pd.DataFrame, config_path: str = "config.yaml") -> dict:
    cfg = load_config(config_path)
    params = cfg["model"]["xgboost_params"]
    test_size = cfg["model"]["test_size"]
    random_state = cfg["model"]["random_state"]
    
    # Target: next day's close price
    df["target"] = df["close"].shift(-1)
    df = df.dropna(subset=["target"])
    
    X = df.drop(columns=["date", "target", "sentiment_label"])  # features only
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    metrics = {
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,  # manual sqrt for compatibility
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred)
    }
    
    # Save model and metrics
    models_dir = Path(cfg["paths"]["models_dir"])
    ensure_dir(models_dir)
    model_file = models_dir / cfg["model"]["model_filename"].split("/")[-1]
    metrics_file = models_dir / cfg["model"]["metrics_filename"].split("/")[-1]
    
    joblib.dump(model, model_file)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model saved to {model_file}")
    print(f"Metrics saved to {metrics_file}")
    print(f"Metrics: {metrics}")
    
    return {"model": model, "metrics": metrics}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train XGBoost model for stock prediction")
    parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    
    df_features = load_features(args.ticker, args.config)
    train_xgboost(df_features, args.config)
