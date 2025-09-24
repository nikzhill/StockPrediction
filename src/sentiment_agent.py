import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
import contractions

# Ensure Vader lexicon is downloaded
nltk.download("vader_lexicon")


class SentimentAgent:
    def __init__(self, input_csv, output_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.analyzer = SentimentIntensityAnalyzer()

        # 📈 Extend with finance-specific words (stronger weights)
        finance_lexicon = {
            "beat": 3.0,
            "beats": 3.0,
            "beating": 3.0,
            "miss": -2.0,
            "missed": -2.0,
            "misses": -2.0,
            "surge": 3.0,
            "soar": 3.0,
            "plummet": -3.5,
            "crash": -3.5,
            "downgrade": -2.5,
            "upgrade": 2.5,
            "outperform": 2.2,
            "underperform": -2.2,
            "guidance": 0.8,
            "cut": -2.0,
            "raise": 2.0,
            "layoffs": -2.5,
            "buyback": 2.0,
            "bullish": 3.5,
            "bearish": -3.5,
            "profit": 2.5,
            "loss": -2.5,
            "growth": 2.5,
            "decline": -2.5,
            "strong": 2.5,
            "weak": -2.5,
        }
        self.analyzer.lexicon.update(finance_lexicon)

    def validate_input(self, df):
        if "date" not in df.columns or "text" not in df.columns:
            raise ValueError("Input CSV must contain 'date' and 'text' columns")
        return df

    def preprocess_text(self, s):
        if pd.isna(s):
            return ""
        s = str(s)
        s = re.sub(r"http\S+", "", s)  # remove URLs
        s = re.sub(r"\$[A-Za-z]{1,5}", "", s)  # remove $AAPL style tickers
        s = re.sub(r"@\w+", "", s)  # remove mentions
        s = contractions.fix(s)  # expand isn't → is not
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def analyze_sentiment(self, text):
        return self.analyzer.polarity_scores(text)

    def run_pipeline(self):
        print("📥 Loading input CSV...")
        df = pd.read_csv(self.input_csv)

        # Validate structure
        df = self.validate_input(df)
        print(f"✅ Loaded {len(df)} rows of news data")

        # Clean text
        df["text"] = df["text"].apply(self.preprocess_text)

        # Apply sentiment analysis
        print("🔎 Running sentiment analysis...")
        sentiment_scores = df["text"].apply(self.analyze_sentiment)

        sentiment_df = pd.DataFrame(list(sentiment_scores))
        df = pd.concat([df, sentiment_df], axis=1)

        # Save headline-level sentiment
        detailed_output = self.output_csv.replace(".csv", "_detailed.csv")
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        df.to_csv(detailed_output, index=False)
        print(f"📝 Detailed sentiment saved to {detailed_output}")

        # Aggregate by date (daily average sentiment)
        print("📊 Aggregating daily sentiment...")
        daily_sentiment = (
            df.groupby("date")[["neg", "neu", "pos", "compound"]].mean().reset_index()
        )

        # Save aggregated output
        daily_sentiment.to_csv(self.output_csv, index=False)
        print(f"✅ Daily sentiment saved to {self.output_csv}")


if __name__ == "__main__":
    input_path = "data/merged/input.csv"
    output_path = "data/sentiment_output.csv"

    agent = SentimentAgent(input_path, output_path)
    agent.run_pipeline()
