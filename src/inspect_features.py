"""
Feature inspection utility for nba-game-predictor
Generates a list of all features used in the model.
"""
from .utils import fetch_nba_data, get_feature_columns
from .nba_data import process_nba_data
import pandas as pd
import os


def save_feature_list():
    """
    Load data, process it, and save the feature list to a text file.
    Used for documentation and feature verification.
    """
    print("Loading data to extract feature names...")
    data_path = os.path.join("data", "nba_2008-2025.csv")
    df = fetch_nba_data(data_path)
    
    if df is None:
        print("Error: Could not load data.")
        return

    # Process data to generate feature columns
    processed_df = process_nba_data(df)
    feature_cols, target = get_feature_columns(processed_df)
    feature_cols.sort()

    output_file = os.path.join("data", "feature_list.txt")
    
    with open(output_file, "w") as f:
        f.write(f"Target Column: {target}\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Features: {len(feature_cols)}\n")
        f.write("=" * 60 + "\n\n")
        
        for col in feature_cols:
            f.write(f"{col}\n")
    
    print(f"Feature list saved to {output_file}")


if __name__ == "__main__":
    save_feature_list()
