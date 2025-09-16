import yfinance as yf
import os
import pandas as pd

# 1️⃣ Ensure the folder exists
os.makedirs("data/raw", exist_ok=True)

# 2️⃣ List of NSE stock tickers (add more if needed)
# NSE tickers in Yahoo Finance usually have ".NS" suffix
nse_tickers = [
    "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "HINDUNILVR.NS", "LT.NS", "AXISBANK.NS"
]

# 3️⃣ Date range
start_date = "2023-01-01"
end_date = "2025-09-13"

# 4️⃣ Download each stock and save as CSV
for ticker in nse_tickers:
    try:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        if not df.empty:
            # Keep only required columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            # Save CSV in data/raw/
            filename = f"data/raw/{ticker.replace('.NS','')}.csv"
            df.to_csv(filename)
            print(f"Saved {filename}")
        else:
            print(f"No data for {ticker}")
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")

print("✅ All done!")
