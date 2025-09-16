# src/sentiment_agent.py
import os
import re
import sys
import json
import yaml
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Callable, Optional, Dict

import pandas as pd
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Global AI hook placeholder
AI_HOOK: Optional[Callable] = None


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def init_sentiment(vader_download: bool = True) -> SentimentIntensityAnalyzer:
    if vader_download:
        nltk.download("vader_lexicon", quiet=True)
        nltk.download("punkt", quiet=True)
    return SentimentIntensityAnalyzer()


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove urls
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_to_sentences(text: str) -> List[str]:
    try:
        from nltk import sent_tokenize
        return sent_tokenize(text)
    except LookupError:
        return text.split(".")


def score_texts_vader(
    df: pd.DataFrame,
    analyzer: SentimentIntensityAnalyzer,
    ai_hook: Optional[Callable] = None,
) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        text = preprocess_text(row["text"])
        if ai_hook:
            enriched = ai_hook(text)
            if "enriched_text" in enriched:
                text = enriched["enriched_text"]
        sentences = split_to_sentences(text)
        for sent in sentences:
            if not sent.strip():
                continue
            if ai_hook:
                enriched = ai_hook(sent)
                if enriched.get("override_sentiment"):
                    scores = enriched["override_sentiment"]
                else:
                    scores = analyzer.polarity_scores(sent)
            else:
                scores = analyzer.polarity_scores(sent)
            records.append(
                {
                    "date": pd.to_datetime(row["date"]).date(),
                    "ticker": row.get("ticker", None),
                    "source": row.get("source", None),
                    "sentence": sent,
                    **scores,
                }
            )
    return pd.DataFrame(records)


def aggregate_daily(sent_df: pd.DataFrame, method: str = "mean") -> pd.DataFrame:
    if sent_df.empty:
        return pd.DataFrame()
    group_cols = ["date", "ticker", "source"]
    agg = sent_df.groupby(group_cols).agg(
        n_items=("compound", "count"),
        mean_compound=("compound", "mean"),
        median_compound=("compound", "median"),
        std_compound=("compound", "std"),
        pos_mean=("pos", "mean"),
        neg_mean=("neg", "mean"),
        neu_mean=("neu", "mean"),
    ).reset_index()
    agg["sentiment_label"] = np.where(
        agg["mean_compound"] >= 0.05, "positive",
        np.where(agg["mean_compound"] <= -0.05, "negative", "neutral")
    )
    return agg


def save_sentiment(df: pd.DataFrame, out_path: Path):
    ensure_dir(out_path.parent)
    df.to_csv(out_path, index=False)


def run_pipeline(
    ticker: Optional[str] = None,
    input_csv: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    ai_hook: Optional[Callable] = None,
    config_path: str = "config.yaml",
) -> Dict:
    cfg = load_config(config_path)
    sentiment_dir = Path(cfg["paths"]["sentiment_dir"])
    ensure_dir(sentiment_dir)

    analyzer = init_sentiment(cfg["sentiment"].get("vader_lexicon_download", True))

    if input_csv:
        df = pd.read_csv(input_csv)
    else:
        raise ValueError("No input CSV provided for sentiment analysis")

    if "date" not in df.columns or "text" not in df.columns:
        raise ValueError("Input CSV must contain 'date' and 'text' columns")

    df["date"] = pd.to_datetime(df["date"])

    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["date"] <= pd.to_datetime(end_date)]

    sent_df = score_texts_vader(df, analyzer, ai_hook)
    daily_df = aggregate_daily(sent_df)

    raw_out = sentiment_dir / f"{ticker or 'unknown'}_sentiment_raw.csv"
    daily_out = sentiment_dir / f"{ticker or 'unknown'}_sentiment_daily.csv"
    save_sentiment(sent_df, raw_out)
    save_sentiment(daily_df, daily_out)

    stats = {
        "ticker": ticker,
        "n_texts": len(df),
        "n_sentences": len(sent_df),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "output_files": {"raw": str(raw_out), "daily": str(daily_out)},
    }
    meta_out = sentiment_dir / f"{ticker or 'unknown'}_sentiment_meta.json"
    with open(meta_out, "w") as f:
        json.dump(stats, f, indent=2)

    return {"raw_df": sent_df, "daily_df": daily_df, "stats": stats}


def register_ai_hook(hook_fn: Callable):
    global AI_HOOK
    AI_HOOK = hook_fn


def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis with VADER")
    parser.add_argument("--input-csv", type=str, required=True, help="Path to CSV with date,text columns")
    parser.add_argument("--ticker", type=str, default=None, help="Ticker symbol")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    result = run_pipeline(
        ticker=args.ticker,
        input_csv=args.input_csv,
        start_date=args.start,
        end_date=args.end,
        ai_hook=AI_HOOK,
        config_path=args.config,
    )
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    main()
