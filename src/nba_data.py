"""
NBA data processing module for Over/Under prediction.
Orchestrates feature engineering pipeline.
"""
import pandas as pd
from .features import (
    calculate_team_stats, 
    calculate_rolling_features, 
    calculate_pace_features,
    calculate_trend_features, 
    calculate_season_features, 
    calculate_matchup_features,
    create_over_under_features
)
from .utils import (
    fetch_nba_data, 
    get_feature_columns, 
    prepare_train_test_split
)

def process_nba_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Main processing function that orchestrates all feature engineering.
    
    Parameters:
    data (pd.DataFrame): Raw NBA game data.
    
    Returns:
    pd.DataFrame: Fully processed data ready for ML modeling.
    """
    print("Processing NBA data for Over/Under prediction...")
    
    # Step 1: Calculate team-centric stats
    print("  Calculating team statistics...")
    team_games = calculate_team_stats(data)
    
    # Step 2: Calculate rolling features
    print("  Calculating rolling window features...")
    team_games = calculate_rolling_features(team_games)
    
    # Step 3: Calculate pace features
    print("  Calculating pace features...")
    team_games = calculate_pace_features(team_games)
    
    # Step 4: Calculate trend features
    print("  Calculating trend features...")
    team_games = calculate_trend_features(team_games)
    
    # Step 5: Calculate season features
    print("  Calculating season features...")
    team_games = calculate_season_features(team_games)
    
    # Step 6: Create matchup features
    print("  Creating matchup features...")
    processed_df = calculate_matchup_features(data, team_games)
    
    # Step 7: Create over/under specific features
    print("  Creating over/under prediction features...")
    processed_df = create_over_under_features(processed_df)
    
    # Drop rows with too many NaN values (early season games)
    initial_rows = len(processed_df)
    processed_df = processed_df.dropna(thresh=len(processed_df.columns) * 0.5)
    print(f"  Dropped {initial_rows - len(processed_df)} rows with insufficient data")
    
    print(f"Processing complete. Final dataset: {len(processed_df)} games, {len(processed_df.columns)} features")
    
    return processed_df


if __name__ == "__main__":
    # Example usage
    # Assuming script is run from project root, or handle relative paths
    import os
    source_path = os.path.join("data", "nba_2008-2025.csv")
    
    # Fetch raw data
    nba_data = fetch_nba_data(source_path)
    
    if nba_data is not None:
        print(f"Loaded {len(nba_data)} games from {source_path}")
        print(f"Seasons: {nba_data['season'].min()} to {nba_data['season'].max()}")
        
        # Process data with all features
        processed_data = process_nba_data(nba_data)
        
        # Show sample of processed data
        print("\nSample of processed features:")
        print(processed_data.head())
        
        # Get feature info
        feature_cols, target_col = get_feature_columns(processed_data)
        print(f"\nTotal features for modeling: {len(feature_cols)}")
        print(f"Target column: {target_col}")
        
        # Prepare train/test split (uses last 5 seasons before test by default)
        X_train, X_test, y_train, y_test, train_df, test_df = prepare_train_test_split(processed_data)
        
        # Show target distribution (regression stats)
        print(f"\nTarget stats (training) - {target_col}:")
        print(f"  Mean: {y_train.mean():.1f}")
        print(f"  Std:  {y_train.std():.1f}")
        print(f"  Min:  {y_train.min():.0f}")
        print(f"  Max:  {y_train.max():.0f}")
        
        # Compare with Vegas line (on training data)
        vegas_mae = (train_df['actual_total'] - train_df['total']).abs().mean()
        print(f"\nVegas line baseline MAE (training): {vegas_mae:.2f} points")
