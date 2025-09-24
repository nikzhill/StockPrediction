import yfinance as yf
import os

# -------------------------------
# Folder to save raw CSVs
# -------------------------------
raw_folder = "data/raw"
os.makedirs(raw_folder, exist_ok=True)

# -------------------------------
# List of tickers
# -------------------------------
tickers = ["AAPL", "TSLA", "MSFT", "GOOGL", "INFY.NS", "TCS.NS"]

# -------------------------------
# Download each ticker
# -------------------------------
for ticker in tickers:
    try:
        print(f"[Info] Downloading {ticker}...")
        df = yf.download(ticker, start="2023-01-01", progress=False)
        if df.empty:
            print(f"[Warning] No data returned for {ticker}")
            continue
        # Reset index to get 'Date' column
        df.reset_index(inplace=True)
        # Save CSV
        file_path = os.path.join(raw_folder, f"{ticker}.csv")
        df.to_csv(file_path, index=False)
        print(f"[Success] Saved {file_path} ({len(df)} rows)")
    except Exception as e:
        print(f"[Error] Failed to download {ticker}: {e}")
