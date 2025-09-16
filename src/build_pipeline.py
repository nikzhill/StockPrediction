import os
import subprocess
import pandas as pd
import sys

# Define paths
RAW_NEWS_DIR = "data/raw"
SENTIMENT_DIR = "data/sentiment"
TECHNICAL_DIR = "data/technical"
OUTPUT_DIR = "data/processed"

TICKERS = ["AAPL", "TSLA", "MSFT", "GOOGL", "INFY.NS", "TCS.NS"]

os.makedirs(SENTIMENT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for ticker in TICKERS:
    print(f"\n🚀 Processing {ticker}...")

    # Step 1: Run sentiment agent if raw news file exists
    raw_news_file = os.path.join(RAW_NEWS_DIR, f"{ticker}_news.csv")
    sentiment_file = os.path.join(SENTIMENT_DIR, f"{ticker}_sentiment.csv")

    if os.path.exists(raw_news_file):
        print(f"   📰 Running sentiment agent for {ticker}...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "src/sentiment_agent.py",
                    "--input-csv", raw_news_file,
                    "--ticker", ticker,
                    "--config", "config.yaml",
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Sentiment agent failed for {ticker}: {e}")
    else:
        print(f"   ⚠️ No raw news file found for {ticker}, skipping sentiment.")

    # Step 2: Merge stock technicals + sentiment
    stock_file = os.path.join(TECHNICAL_DIR, f"{ticker}_technical.csv")
    if not os.path.exists(stock_file):
        print(f"   ❌ Stock file not found for {ticker}, skipping...")
        continue

    stock = pd.read_csv(stock_file)
    stock.columns = stock.columns.str.strip()

    # Normalize date
    if "date" in stock.columns:
        stock["date"] = pd.to_datetime(stock["date"]).dt.date
    elif "Date" in stock.columns:
        stock["date"] = pd.to_datetime(stock["Date"]).dt.date

    if os.path.exists(sentiment_file):
        sent = pd.read_csv(sentiment_file)
        sent.columns = sent.columns.str.strip()

        if "date" in sent.columns:
            sent["date"] = pd.to_datetime(sent["date"]).dt.date
        elif "Date" in sent.columns:
            sent["date"] = pd.to_datetime(sent["Date"]).dt.date

        merged = pd.merge(stock, sent, on="date", how="left")
        print(f"   ✅ Merged stock + sentiment for {ticker}: {len(merged)} rows")
    else:
        merged = stock
        print(f"   ⚠️ No sentiment file for {ticker}, saved stock only.")

    # Save processed file
    output_file = os.path.join(OUTPUT_DIR, f"processed_{ticker}_stock_data.csv")
    merged.to_csv(output_file, index=False)
    print(f"   💾 Saved {output_file}")
0