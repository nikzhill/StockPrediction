import pandas as pd

df = pd.read_csv("./processed_AAPL_with_sentiment.csv", skiprows=[1, 2])

# Parse the actual date column
df["date"] = pd.to_datetime(
    df["price"], errors="coerce"
)  # 'price' column is actually holding dates

# Rename columns properly
df = df.rename(columns={"price": "date", "close": "price"})

# Drop invalid rows
df = df.dropna(subset=["date"])

print(df.head())
