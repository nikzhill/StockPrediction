import pandas as pd
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure Vader lexicon is downloaded
nltk.download("vader_lexicon")


class SentimentAgent:
    def __init__(self, input_csv, output_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.analyzer = SentimentIntensityAnalyzer()

    def validate_input(self, df):
        # Ensure required columns exist
        if "date" not in df.columns or "text" not in df.columns:
            raise ValueError("Input CSV must contain 'date' and 'text' columns")
        return df

    def preprocess_text(self, text):
        """Clean text before sentiment analysis"""
        if pd.isna(text):
            return ""
        # Remove extra whitespace
        return str(text).strip()

    def analyze_sentiment(self, text):
        """Return sentiment scores for a given text"""
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

        # Aggregate by date (daily average sentiment)
        print("📊 Aggregating daily sentiment...")
        daily_sentiment = df.groupby("date")[["neg", "neu", "pos", "compound"]].mean().reset_index()

        # Save output
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        daily_sentiment.to_csv(self.output_csv, index=False)

        print(f"✅ Sentiment analysis complete! Results saved to {self.output_csv}")


if __name__ == "__main__":
    # Example usage
    input_path = "data/merged/input.csv"      # Input file: must have 'date' and 'text'
    output_path = "data/sentiment_output.csv"  # Output file

    agent = SentimentAgent(input_path, output_path)
    agent.run_pipeline()
