"""
data_processing.py
Cleans raw sentiment dataset and saves a processed version for time series analysis.
"""

import pandas as pd
import os

RAW_PATH = "data/raw/Sentiment dataset.csv"
PROCESSED_PATH = "data/processed/cleaned_sentiment.csv"


def clean_data(input_path=RAW_PATH, output_path=PROCESSED_PATH):
    # Load raw data
    df = pd.read_csv(input_path)

    # Debug: show columns in the dataset
    print("📌 Columns found in dataset before cleaning:", df.columns.tolist())
    print("🔎 Sample before cleaning:")
    print(df.head(), "\n")

    # Normalize column names (lowercase, strip spaces)
    df.columns = df.columns.str.strip().str.lower()

    # Drop unnamed/junk columns
    df = df.loc[:, ~df.columns.str.contains("^unnamed")]

    print("📌 Columns after normalization:", df.columns.tolist())

    # Drop unnecessary text/meta columns if present
    drop_cols = ["text", "sentiment", "user", "platform", "hashtags", "country", "year", "month", "day", "hour"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Drop duplicates
    df = df.drop_duplicates()

    # Identify timestamp column
    if "timestamp" not in df.columns:
        raise ValueError("❌ No timestamp column found in dataset after cleaning!")

    # Clean timestamp column
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Set timestamp as index
    df = df.set_index("timestamp").sort_index()

    # Columns available for aggregation
    agg_dict = {}
    if "retweets" in df.columns:
        agg_dict["retweets"] = "sum"
    if "likes" in df.columns:
        agg_dict["likes"] = "sum"

    if not agg_dict:
        raise ValueError("❌ No 'likes' or 'retweets' columns found for aggregation!")

    # Resample to daily frequency
    df = df.resample("D").agg(agg_dict)

    # Fill missing days with 0
    df = df.fillna(0)

    print("✅ Sample after cleaning (daily aggregated):")
    print(df.head(), "\n")

    # Save processed dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)

    print(f"✅ Processed data saved at {output_path}")


if __name__ == "__main__":
    clean_data()
