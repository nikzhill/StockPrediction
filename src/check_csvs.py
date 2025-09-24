import os
import pandas as pd

# Folder containing processed CSVs
processed_folder = "data/processed/"

# Columns to check for numeric data
numeric_cols = ["price", "close", "high", "low", "open", "volume"]

# Loop through all CSVs
for file in os.listdir(processed_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(processed_folder, file)
        df = pd.read_csv(file_path)

        # Strip column names of spaces just in case
        df.columns = df.columns.str.strip()

        # Check if any numeric columns have at least one non-NaN value
        empty_cols = [
            col for col in numeric_cols if col in df.columns and df[col].isna().all()
        ]

        if empty_cols:
            print(f"[⚠️] {file} has empty columns: {empty_cols}")
        else:
            print(f"[✅] {file} looks good!")
