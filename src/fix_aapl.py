import pandas as pd

# Load the corrupted AAPL file
input_file = "data/processed/processed_AAPL_stock_data.csv"
output_file = "data/processed/processed_AAPL_stock_data_clean.csv"

# Read CSV normally
df = pd.read_csv(input_file)

# Drop rows where the second column contains junk like "Ticker" or "Date"
df = df[~df.iloc[:, 1].astype(str).isin(["Ticker", "Date"])]

# Ensure date column is parsed correctly
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
elif "price" in df.columns:
    # Some of your files had "price" accidentally holding dates
    df = df.rename(columns={"price": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Drop rows with invalid dates
df = df.dropna(subset=["date"])

# Save cleaned file with the same expected name (overwrite original if safe)
df.to_csv(output_file, index=False)

print(f"✅ Cleaned AAPL file saved to {output_file} with {len(df)} rows.")
