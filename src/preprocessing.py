import os
import pandas as pd

DATA_DIR = "data/raw_fixed"
OUTPUT_DIR = "data/technical"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def preprocess_file(input_path, output_path):
    try:
        # --- Read CSV, flexible header handling ---
        df = pd.read_csv(input_path, header=0)

        # If first row contains actual column names, reset header
        if "Date" not in df.columns and df.iloc[0].dtype == object:
            df.columns = df.iloc[0]
            df = df.drop([0]).reset_index(drop=True)

        # --- Auto-detect date column ---
        date_col = None
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                date_col = col
                break

        if not date_col:
            # fallback: use first column
            date_col = df.columns[0]
            print(f"⚠️ No 'Date' column detected, using first column '{date_col}' as Date")

        # Convert to datetime
        df["Date"] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=["Date"])

        # --- Convert numeric columns safely ---
        for col in df.columns:
            if col != "Date":
                series = df[col]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                series = series.astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(series, errors="coerce")

        # --- Fill missing values ---
        df = df.ffill().bfill()

        # --- Technical indicators ---
        if "Close" in df.columns and df["Close"].notna().any():
            df["SMA_10"] = df["Close"].rolling(10).mean()
            df["SMA_50"] = df["Close"].rolling(50).mean()
            df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
            df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
            df["RSI"] = compute_rsi(df["Close"], 14)
        else:
            print(f"⚠️ No valid 'Close' column in {input_path}, skipping indicators.")

        # --- Add dummy sentiment columns ---
        for col in ["sentiment_compound", "sentiment_positive", "sentiment_negative",
                    "sentiment_strength", "positive_ratio", "negative_ratio"]:
            if col not in df.columns:
                df[col] = 0

        # --- Save processed file ---
        df.to_csv(output_path, index=False)
        print(f"✅ Processed {input_path} → {output_path}")

    except Exception as e:
        print(f"⚠️ Error processing {input_path}: {e}")

def main():
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv") and "_news" not in file:
            input_path = os.path.join(DATA_DIR, file)
            ticker = file.replace(".csv", "")
            output_path = os.path.join(OUTPUT_DIR, f"{ticker}_technical.csv")
            preprocess_file(input_path, output_path)

if __name__ == "__main__":
    main()
