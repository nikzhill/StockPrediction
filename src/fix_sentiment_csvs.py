import os
import pandas as pd

RAW_SENT_DIR = "data/sentiment_raw"
FIXED_SENT_DIR = "data/sentiment_fixed"
os.makedirs(FIXED_SENT_DIR, exist_ok=True)

# Expected final columns
COLUMNS = [
    "Date",
    "Ticker",
    "sentiment_compound",
    "sentiment_positive",
    "sentiment_negative",
    "sentiment_strength",
    "positive_ratio",
    "negative_ratio"
]

def fix_sentiment_file(file_path, output_path):
    try:
        df = pd.read_csv(file_path, header=None)  # read without headers
        # First row often contains headers/ticker names
        first_row = df.iloc[0].tolist()
        data_rows = df.iloc[1:].reset_index(drop=True)

        # Determine number of sentiment columns per ticker
        # Usually after 'Date' column, the sentiment columns follow
        # We'll assume last 7 columns are sentiment features
        if "Date" in first_row:
            date_idx = first_row.index("Date")
            ticker = first_row[0]  # first column is ticker
        else:
            date_idx = len(first_row) - 7
            ticker = first_row[0]

        # Reshape the data: each row corresponds to Date + Ticker + sentiment columns
        fixed_rows = []
        for idx, row in data_rows.iterrows():
            row_list = row.tolist()
            # Skip empty rows
            if all(pd.isna(x) for x in row_list):
                continue
            # Grab last 7 values as sentiment, first non-NaN as Date
            date_val = row_list[date_idx] if date_idx < len(row_list) else None
            sentiment_vals = row_list[-7:] if len(row_list) >= 7 else [0]*7
            fixed_rows.append([date_val, ticker] + sentiment_vals)

        fixed_df = pd.DataFrame(fixed_rows, columns=COLUMNS)
        # Convert Date column to datetime
        fixed_df["Date"] = pd.to_datetime(fixed_df["Date"], errors="coerce")
        fixed_df = fixed_df.dropna(subset=["Date"])

        # Save fixed CSV
        fixed_df.to_csv(output_path, index=False)
        print(f"✅ Fixed sentiment file: {file_path} → {output_path}")

    except Exception as e:
        print(f"⚠️ Error fixing {file_path}: {e}")


def main():
    for file in os.listdir(RAW_SENT_DIR):
        if file.endswith(".csv"):
            input_path = os.path.join(RAW_SENT_DIR, file)
            output_path = os.path.join(FIXED_SENT_DIR, file)
            fix_sentiment_file(input_path, output_path)

if __name__ == "__main__":
    main()
