# src/debug_data.py
import os
import pandas as pd

DATA_DIR = "data/processed"

def debug_processed_data():
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            file_path = os.path.join(DATA_DIR, file)
            print(f"\n🔎 Inspecting: {file}")
            df = pd.read_csv(file_path)

            # Show first few rows
            print(df.head())

            # Show column data types
            print("\nData Types:")
            print(df.dtypes)

            # Check if any columns are object type
            obj_cols = df.select_dtypes(include=["object"]).columns
            if len(obj_cols) > 0:
                print(f"⚠️ Found object columns: {list(obj_cols)}")
            else:
                print("✅ All columns are numeric (or datetime for Date).")

            print("-" * 50)

if __name__ == "__main__":
    debug_processed_data()
