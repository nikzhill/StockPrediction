import os
import pandas as pd

RAW_DIR = "data/raw"
FIXED_DIR = "data/raw_fixed"
os.makedirs(FIXED_DIR, exist_ok=True)

# Mapping of possible variations to standard column names
COLUMN_MAP = {
    "date": "Date",
    "timestamp": "Date",
    "trade_date": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj close": "Close",
    "volume": "Volume",
}

def fix_csv(file_path, output_path):
    try:
        df = pd.read_csv(file_path)
        new_cols = []
        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in COLUMN_MAP:
                new_cols.append(COLUMN_MAP[col_lower])
            else:
                new_cols.append(col.strip())
        df.columns = new_cols
        df.to_csv(output_path, index=False)
        print(f"✅ Fixed {file_path} → {output_path}")
    except Exception as e:
        print(f"⚠️ Error fixing {file_path}: {e}")

def main():
    for file in os.listdir(RAW_DIR):
        if file.endswith(".csv"):
            input_path = os.path.join(RAW_DIR, file)
            output_path = os.path.join(FIXED_DIR, file)
            fix_csv(input_path, output_path)

if __name__ == "__main__":
    main()
