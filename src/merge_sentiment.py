import os
import pandas as pd

TECH_DIR = "data/technical"
SENTIMENT_DIR = "data/sentiment"
MERGED_DIR = "data/merged"

os.makedirs(MERGED_DIR, exist_ok=True)

def merge_technical_sentiment(ticker):
    tech_path = os.path.join(TECH_DIR, f"{ticker}_technical.csv")
    sentiment_path = os.path.join(SENTIMENT_DIR, f"{ticker}_sentiment.csv")
    output_path = os.path.join(MERGED_DIR, f"{ticker}_merged.csv")

    if not os.path.exists(tech_path):
        print(f"⚠️ Technical file not found for {ticker}")
        return

    df_tech = pd.read_csv(tech_path)

    if os.path.exists(sentiment_path):
        df_sent = pd.read_csv(sentiment_path)
        df_sent["Date"] = pd.to_datetime(df_sent["Date"], errors="coerce")
        df_merged = pd.merge(df_tech, df_sent, on="Date", how="left")
        # Fill missing sentiment columns if any
        for col in ["sentiment_compound", "sentiment_positive", "sentiment_negative",
                    "sentiment_strength", "positive_ratio", "negative_ratio"]:
            if col not in df_merged.columns:
                df_merged[col] = 0
        df_merged = df_merged.fillna(0)
    else:
        print(f"⚠️ Sentiment file not found for {ticker}, keeping dummy sentiment columns")
        df_merged = df_tech

    df_merged.to_csv(output_path, index=False)
    print(f"✅ Merged {ticker} → {output_path}")

def main():
    files = [f for f in os.listdir(TECH_DIR) if f.endswith("_technical.csv")]
    tickers = [f.replace("_technical.csv", "") for f in files]

    for ticker in tickers:
        merge_technical_sentiment(ticker)

if __name__ == "__main__":
    main()
