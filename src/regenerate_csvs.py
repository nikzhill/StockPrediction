import os
import yfinance as yf
import pandas as pd

# Folder to save processed CSVs
processed_folder = "data/processed/"
os.makedirs(processed_folder, exist_ok=True)

# List of tickers
tickers = ["AAPL", "GOOGL", "INFY.NS", "MSFT", "TCS.NS", "TSLA"]

# Date range
start_date = "2023-01-01"
end_date = "2025-09-23"

for ticker in tickers:
    print(f"Fetching data for {ticker}...")
    try:
        # Use Ticker().history() which is more reliable for single tickers
        t = yf.Ticker(ticker)
        df = t.history(start=start_date, end=end_date, auto_adjust=False)

        if df.empty:
            print(f"[⚠️] No data for {ticker}, skipping.")
            continue

        # Reset index to get date as a column
        df.reset_index(inplace=True)

        # Create processed dataframe
        processed_df = pd.DataFrame(
            {
                "date": df["Date"].dt.strftime("%Y-%m-%d"),
                "open": df["Open"],
                "high": df["High"],
                "low": df["Low"],
                "close": df["Close"],
                "volume": df["Volume"],
                "price": df["Close"],
            }
        )

        # Normalized columns
        for col in ["open", "high", "low", "close", "volume", "price"]:
            processed_df[f"{col}_norm"] = (
                processed_df[col] - processed_df[col].min()
            ) / (processed_df[col].max() - processed_df[col].min())

        # Save CSV
        save_path = os.path.join(processed_folder, f"processed_{ticker}_stock_data.csv")
        processed_df.to_csv(save_path, index=False)
        print(f"[✅] Saved processed CSV for {ticker} at {save_path}")

    except Exception as e:
        print(f"[❌] Failed for {ticker}: {e}")

print("All tickers processed.")
