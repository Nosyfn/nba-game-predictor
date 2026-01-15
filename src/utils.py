import pandas as pd
from typing import Tuple, List, Optional

from .config import EXCLUDE_FROM_FEATURES


def fetch_nba_data(source_path: str) -> pd.DataFrame:
    """
    Fetch NBA data from the given CSV file path.
    
    Parameters:
    source_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: A DataFrame containing the NBA data.
    """
    try:
        data = pd.read_csv(source_path)
        return data
    except Exception as e:
        print(f"Make sure you have put csv file in the correct location. Error: {e}")
        return None


def fetch_nba_data_from_db() -> Optional[pd.DataFrame]:
    """
    Fetch NBA data from the SQLite database.
    
    Returns:
    pd.DataFrame: A DataFrame containing the NBA data, or None if empty.
    """
    from .database import get_all_games
    
    try:
        data = get_all_games()
        if data.empty:
            print("No games found in database. Please run seed_database.py first.")
            return None
        return data
    except Exception as e:
        print(f"Error fetching data from database: {e}")
        return None


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], str]:
    """
    Get the feature columns and target column for ML modeling.
    Uses actual_total as target for regression (predicting total points).
    
    Parameters:
    df (pd.DataFrame): Processed dataframe.
    
    Returns:
    Tuple[List[str], str]: Feature column names and target column name.
    """
    # Use centralized exclude list from config
    feature_cols = [col for col in df.columns if col not in EXCLUDE_FROM_FEATURES]
    target_col = 'actual_total'  # Regression target: predict total points scored
    
    return feature_cols, target_col


def prepare_train_test_split(df: pd.DataFrame, test_seasons: List[int] = None,
                             train_seasons: int = 5) -> Tuple:
    """
    Prepare train/test split based on seasons (time-based split).
    Uses a rolling window of recent seasons for training to avoid data drift.
    
    Parameters:
    df (pd.DataFrame): Processed dataframe.
    test_seasons (List[int]): Defaults to [2024, 2025].
    train_seasons (int):  e.g 2019-2023 if testing on 2024-2025).
    
    Returns:
    Tuple: X_train, X_test, y_train, y_test, train_df, test_df
    """
    if test_seasons is None:
        test_seasons = [2024, 2025]  # Use last 2 seasons for testing
    
    feature_cols, target_col = get_feature_columns(df)
    
    # Calculate training season range (rolling window before test)
    min_test_season = min(test_seasons)
    train_season_start = min_test_season - train_seasons
    train_season_end = min_test_season - 1
    valid_train_seasons = list(range(train_season_start, train_season_end + 1))
    
    train_df = df[df['season'].isin(valid_train_seasons)]
    test_df = df[df['season'].isin(test_seasons)]
    
    
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    
    print(f"Training set: {len(X_train)} games (seasons {train_season_start}-{train_season_end})")
    print(f"Test set: {len(X_test)} games (seasons {min(test_seasons)}-{max(test_seasons)})")
    
    return X_train, X_test, y_train, y_test, train_df, test_df
