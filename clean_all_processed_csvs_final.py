import pandas as pd
import os
from datetime import datetime
import shutil

# Path to processed CSV folder
processed_folder = "data/processed"

# Backup folder
backup_folder = os.path.join(processed_folder, "backup")
os.makedirs(backup_folder, exist_ok=True)

# Date range to keep
start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp.today()

# Manual mapping for known problematic files
date_column_mapping = {
    "processed_GOOGL_stock_data.csv": "Date",
    "processed_GOOGL_with_sentiment.csv": "Date",
    "processed_INFY.NS_stock_data.csv": "Date",
    "processed_INFY.NS_with_sentiment.csv": "Date",
    "processed_MSFT_stock_data.csv": "Date",
    "processed_MSFT_with_sentiment.csv": "Date",
    "processed_TCS.NS_stock_data.csv": "Date",
    "processed_TCS.NS_with_sentiment.csv": "Date",
    "processed_TSLA_stock_data.csv": "Date",
    "processed_TSLA_with_sentiment.csv": "Date",
}

# List all CSV files in the folder
csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

if not csv_files:
    print("[Info] No CSV files found in", processed_folder)

for file_name in csv_files:
    file_path = os.path.join(processed_folder, file_name)

    # Backup original
    shutil.copy(file_path, os.path.join(backup_folder, file_name))

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # clean column names

    # --- Determine date column ---
    date_col = date_column_mapping.get(file_name, None)

    # If no mapping, try auto-detect
    if date_col is None:
        for col in df.columns:
            try:
                pd.to_datetime(df[col].dropna().iloc[:5])
                date_col = col
                break
            except Exception:
                continue

    if date_col is None:
        print(f"[Error] No datetime-like column detected in {file_name}, skipping.")
        continue

    # Convert to datetime, drop invalid
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    # Filter only valid date range
    df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]

    # Sort by date ascending
    df = df.sort_values(date_col).reset_index(drop=True)

    # Save cleaned CSV (overwrite)
    df.to_csv(file_path, index=False)
    print(f"[Success] Cleaned {file_name} ({date_col} detected): {len(df)} rows")
