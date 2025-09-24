import pandas as pd
import os
import shutil

# -------------------------------
# Paths
# -------------------------------
processed_folder = "data/processed"
raw_folder = "data/raw"
backup_folder = os.path.join(processed_folder, "backup")
os.makedirs(backup_folder, exist_ok=True)

# -------------------------------
# Date range filter
# -------------------------------
start_date = pd.Timestamp("2023-01-01")
end_date = pd.Timestamp.today()

# -------------------------------
# List all processed CSVs
# -------------------------------
csv_files = [f for f in os.listdir(processed_folder) if f.endswith(".csv")]

if not csv_files:
    print("[Info] No CSV files found in", processed_folder)

for file_name in csv_files:
    proc_path = os.path.join(processed_folder, file_name)
    df = pd.read_csv(proc_path)
    df.columns = df.columns.str.strip().str.lower()  # normalize column names

    # -------------------------------
    # Backup original
    # -------------------------------
    shutil.copy(proc_path, os.path.join(backup_folder, file_name))

    # -------------------------------
    # Ensure date column exists
    # -------------------------------
    date_col = None
    # Check if 'date' or 'datetime' exists
    for col in df.columns:
        if "date" in col or "time" in col:
            date_col = col
            break

    # If missing, try to add from raw CSV
    if date_col is None:
        ticker = None
        for t in ["AAPL", "TSLA", "MSFT", "GOOGL", "INFY.NS", "TCS.NS"]:
            if t.lower() in file_name.lower():
                ticker = t
                break

        if ticker is not None:
            raw_file = os.path.join(raw_folder, f"{ticker}.csv")
            if os.path.exists(raw_file):
                raw_df = pd.read_csv(raw_file)
                # Find date column in raw
                raw_date_col = None
                for col in raw_df.columns:
                    if "date" in col.lower() or "time" in col.lower():
                        raw_date_col = col
                        break
                if raw_date_col:
                    df["date"] = pd.to_datetime(raw_df[raw_date_col])
                    date_col = "date"
                    print(
                        f"[Info] Added date column to {file_name} from raw {ticker}.csv"
                    )
                else:
                    print(f"[Warning] No date column found in raw {ticker}.csv")
            else:
                print(f"[Warning] Raw CSV for {ticker} not found")
        else:
            print(f"[Warning] Could not determine ticker for {file_name}")

    # -------------------------------
    # If date column exists, clean CSV
    # -------------------------------
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        df = df[(df[date_col] >= start_date) & (df[date_col] <= end_date)]
        df = df.sort_values(date_col).reset_index(drop=True)
        df.to_csv(proc_path, index=False)
        print(f"[Success] Cleaned {file_name} ({date_col}): {len(df)} rows")
    else:
        print(f"[Skipped] {file_name}: No date column to clean")
