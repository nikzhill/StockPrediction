# test_pipeline.py - Check pipeline health
import os
import pandas as pd
import joblib

# Define project folders
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
SENTIMENT_DIR = "data"
MODELS_DIR = "models"

# Stocks you expect to track
STOCKS = ["AAPL", "TSLA", "MSFT", "GOOGL", "INFY.NS", "TCS.NS"]


def check_raw_data():
    print("\n🔍 Checking Raw Data...")
    if not os.path.exists(RAW_DIR):
        return "❌ raw folder missing"
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    return f"✅ Found {len(files)} raw CSVs" if files else "⚠️ No raw CSVs found"


def check_processed_data():
    print("\n🔍 Checking Processed Data...")
    if not os.path.exists(PROCESSED_DIR):
        return "❌ processed folder missing"
    results = []
    for stock in STOCKS:
        fname = f"processed_{stock}_stock_data.csv"
        fpath = os.path.join(PROCESSED_DIR, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                if "Date" in df.columns and "Close" in df.columns:
                    results.append(f"✅ {stock} processed OK ({len(df)} rows)")
                else:
                    results.append(f"⚠️ {stock} missing Date/Close column")
            except Exception as e:
                results.append(f"❌ {stock} could not read CSV: {e}")
        else:
            results.append(f"⚠️ {stock} processed file missing")
    return "\n".join(results)


def check_sentiment():
    print("\n🔍 Checking Sentiment Data...")
    fpath = os.path.join(SENTIMENT_DIR, "sentiment_output.csv")
    if os.path.exists(fpath):
        try:
            df = pd.read_csv(fpath)
            if "date" in df.columns and "compound" in df.columns:
                return f"✅ Sentiment data OK ({len(df)} rows)"
            else:
                return "⚠️ Sentiment file missing required columns"
        except Exception as e:
            return f"❌ Sentiment CSV could not be read: {e}"
    return "⚠️ sentiment_output.csv missing"


def check_models():
    print("\n🔍 Checking Models...")
    if not os.path.exists(MODELS_DIR):
        return "❌ models folder missing"
    results = []
    for stock in STOCKS:
        fname = f"{stock}_xgboost.pkl"
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.exists(fpath):
            try:
                model = joblib.load(fpath)
                results.append(f"✅ {stock} model loaded")
            except Exception as e:
                results.append(f"❌ {stock} model load failed: {e}")
        else:
            results.append(f"⚠️ {stock} model missing")
    return "\n".join(results)


def main():
    print("🚦 Pipeline Health Check\n")

    print(check_raw_data())
    print(check_processed_data())
    print(check_sentiment())
    print(check_models())

    print(
        "\n📌 If everything shows ✅, you’re ready to run: streamlit run dashboard.py"
    )


if __name__ == "__main__":
    main()
