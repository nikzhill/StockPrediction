import yfinance as yf
import os

# List of stock tickers to fetch
TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "INFY.NS", "TCS.NS"]  # you can add/remove here

# Output folder
DATA_DIR = "data"

def fetch_and_save_data(ticker, period="1y", interval="1d"):
    """Fetch stock data and save as CSV in /data folder."""
    try:
        # Download stock data
        df = yf.download(ticker, period=period, interval=interval)

        # Create /data folder if not exists
        os.makedirs(DATA_DIR, exist_ok=True)

        # File path
        file_path = os.path.join(DATA_DIR, f"{ticker}_stock_data.csv")

        # Save to CSV
        df.to_csv(file_path)

        print(f"✅ Data for {ticker} saved to {file_path}")

    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")

if __name__ == "__main__":
    for ticker in TICKERS:
        fetch_and_save_data(ticker)
